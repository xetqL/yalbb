#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <string>
#include <mpi.h>
#include <random>
#include <set>


#include <liblj/nbody_io.hpp>
#include <liblj/neighborhood.hpp>

#include "../includes/spatial_elements.hpp"
#include "../includes/geometric_load_balancer.hpp"
#include "../includes/physics.hpp"
#include "../includes/ljpotential.hpp"

static int rank;
static int nproc;
static MPI_Datatype pairtype;
static MPI_Datatype intpairtype;

/**
 * \subsection{Problem partitioning}
 *
 * Divide the problem more-or-less evenly among processors.  Every
 * processor at least gets [[num_each]] = $\lfloor n/p \rfloor$
 * particles, and the first few processors each get one more.
 * !! Each processor assigns particles for each processor
 **/
void partition_problem(std::vector<int>& iparts, std::vector<int>& counts, int npart) {
    int num_each = npart / nproc; /* Each processor has the same number of particle */
    int num_left = npart - num_each*nproc; /* How many particles are not computed by me */
    iparts[0] = 0;

    for (int i = 0; i < nproc; ++i) {
        counts[i] = num_each + (i < num_left ? 1 : 0);
        iparts[i + 1] = iparts[i] + counts[i];
    }
}

void init_particles_random_v(int n, std::vector<float>& v, sim_param_t* params) {
    float T0 = params->T0;
    for (int i = 0; i < n; ++i) {
        double R = T0 * std::sqrt(-2 * std::log(drand48()));
        double T = 2 * M_PI * drand48();
        v[2 * i + 0] = (float) (R * std::cos(T));
        v[2 * i + 1] = (float) (R * std::sin(T));
    }
}

void run_box(FILE* fp, /* Output file (at 0) */
        int npframe, /* Steps per frame */
        int nframes, /* Frames */
        float dt, /* Time step */
        std::vector<float>& x, /* Global position vec */
        std::vector<float>& xlocal, /* Local part of position */
        std::vector<float>& vlocal, /* Local part of velocity */
        int* iparts,
        int* counts,
        const sim_param_t* params) /* Simulation params */ {

    std::vector<float> alocal(xlocal.size(), 0.0);

    /* r_m = 3.2 * sig */
    double rm = 3.2 * std::sqrt(params->sig_lj);

    /* number of cell in a row*/
    int M = (int) (params->simsize / rm);
    std::vector<int> head(M * M), plklist(params->npart);

    int n = x.size() / 2;
    int nlocal = xlocal.size() / 2;

    float simsize = params->simsize;
    float lsub;
    float lcell = simsize;

    /* size of cell */
    lsub = lcell / ((float) M);

    if (fp) {
        write_header(fp, n, simsize);
        write_frame_data(fp, n, &x[0]);
    }

    switch (params->computation_method) {
    case 1: //brute force
        compute_forces(n, x, iparts[rank], iparts[rank + 1], xlocal, alocal, params);
        break;
    case 2: // cell linked list method
        create_cell_linkedlist(M, lsub, n, &x[0], &plklist[0], &head[0]);
        compute_forces(n, M, lsub, x, iparts[rank], iparts[rank + 1], xlocal, alocal, head, plklist, params);
        break;
    }

    clock_t begin = clock();

    for (int frame = 1; frame < nframes; ++frame) {
        for (int i = 0; i < npframe; ++i) {

            leapfrog1(nlocal, dt, &xlocal[0], &vlocal[0], &alocal[0]);

            apply_reflect(nlocal, &xlocal[0], &vlocal[0], &alocal[0], simsize);

            MPI_Allgatherv(&xlocal[0], nlocal, pairtype, &x[0], counts, iparts, pairtype, MPI_COMM_WORLD);

            switch (params->computation_method) {
            case 1:
                compute_forces(n, x, iparts[rank], iparts[rank + 1], xlocal, alocal, params);
                break;
            case 2:
                create_cell_linkedlist(M, lsub, n, &x[0], &plklist[0], &head[0]);
                compute_forces(n, M, lsub, x, iparts[rank], iparts[rank + 1], xlocal, alocal, head, plklist, params);
                break;
            }

            leapfrog2(nlocal, dt, &vlocal[0], &alocal[0]);
        }
        if (fp) {
            clock_t end = clock();
            double time_spent = (double) (end - begin) / CLOCKS_PER_SEC;
            write_frame_data(fp, n, &x[0]);
            printf("Frame [%d] completed in %f seconds\n", frame, time_spent);
            begin = clock();
        }
    }
}

int main(int argc, char** argv) {
    sim_param_t params;
    FILE* fp = NULL;
    int npart, npart_wu;
    int nlocal;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Type_vector(1, 2, 1, MPI_FLOAT, &pairtype);
    MPI_Type_vector(1, 2, 1, MPI_INT, &intpairtype);

    MPI_Type_commit(&pairtype);
    MPI_Type_commit(&intpairtype);

    if (get_params(argc, argv, &params) != 0) {
        MPI_Finalize();
        return -1;
    }

    if(params.world_size != (size_t) nproc) {
        if(rank == 0) printf("Size of world does not match the expected world size: World=%d, Expected=%d\n", nproc, params.world_size);
        MPI_Finalize();
        return -1;
    }
    
    std::vector<float> x(2 * params.npart, 0);
    std::vector<float> v(2 * params.npart, 0);
    std::vector<liblj::work_unit_t> w;
    std::vector<float> wser(4 * params.npart, 0);
    std::vector<elements::Element<2>> elements(params.npart);

    partitioning::geometric::SeqSpatialBisection<2> rcb_partitioner;
    load_balancing::geometric::GeometricLoadBalancer<2> load_balancer(rcb_partitioner, MPI_COMM_WORLD);
    std::array<std::pair<double, double>, 2> domain_boundary =
    {
        std::make_pair(0.0, params.simsize), std::make_pair(0.0, params.simsize)
    };

    /* Get file handle and initialize everything on P0 */
    if (rank == 0) {
        fp = fopen(params.fname, "w");

        npart_wu = liblj::init_work_unit_random<std::normal_distribution<double>>(params.npart, w, 0.0, 0.0, params.simsize, params.simsize, 0.5, 0.1);

        if (npart_wu < params.npart) {
            fprintf(stderr, "Could not generate %d particles; trying %d\n", params.npart, npart);
        }
        npart = npart_wu;
        x = *liblj::serialize_work_units_position(w);
        init_particles_random_v(v.size() / 2, v, &params);
        elements::transform(elements, &x.front(), &v.front());
    }

    MPI_Bcast(&npart,         1,  MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&x.front(), npart, pairtype, 0, MPI_COMM_WORLD);

    load_balancer.load_balance(elements, domain_boundary);

    double t1 = MPI_Wtime();

    params.npart = npart;

    std::vector<int> iparts(nproc), counts(nproc);

    partition_problem(iparts, counts, x.size() / 2);

    std::vector<float> xlocal(&x[2 * iparts[rank]], &x[2 * iparts[rank] + 2 * counts[rank]]);

    xlocal.shrink_to_fit();

    std::vector<float> vlocal(xlocal.size(), 0.0);

    nlocal = xlocal.size();

    init_particles_random_v(nlocal / 2, vlocal, &params);

    run_box(fp, params.npframe, params.nframes, params.dt, x, xlocal, vlocal, &iparts[0], &counts[0], &params);

    double t2 = MPI_Wtime();

    if (fp) fclose(fp);
    if (rank == 0) printf("Simulation finished: %f seconds\n", (t2-t1));

    MPI_Finalize();
    return 0;
}