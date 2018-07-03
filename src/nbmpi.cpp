#include <string>
#include <mpi.h>
#include <random>

#include <zoltan.h>

#include "../includes/runners/simulator.hpp"
#include "../includes/initial_conditions.hpp"
#include "../includes/nbody_io.hpp"
#include "../includes/params.hpp"

int main(int argc, char** argv) {
    constexpr int DIMENSION = 3;
    sim_param_t params;
    FILE* fp = NULL;
    int rank, nproc, dim;
    float ver;
    MESH_DATA<DIMENSION> mesh_data;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    if (get_params(argc, argv, &params) != 0) {
        MPI_Finalize();
        return -1;
    }

    if (rank == 0 && params.record) {
        fp = fopen(params.fname, "w");
    }

    int changes, numGidEntries, numLidEntries, numImport, numExport;
    ZOLTAN_ID_PTR importGlobalGids, importLocalGids, exportGlobalGids, exportLocalGids;
    int *importProcs, *importToPart, *exportProcs, *exportToPart;

    double t1 = MPI_Wtime();
    partitioning::CommunicationDatatype datatype = elements::register_datatype<DIMENSION>();

    int rc = Zoltan_Initialize(argc, argv, &ver);

    if(rc != ZOLTAN_OK){
        MPI_Finalize();
        exit(0);
    }

    if(rank == 0){
        initial_condition::lennard_jones::RejectionCondition<DIMENSION> condition(&(mesh_data.els),
                                                                                  params.sig_lj,
                                                                                  params.sig_lj*params.sig_lj,
                                                                                  params.T0,
                                                                                  0, 0, 0,
                                                                                  params.simsize,
                                                                                  params.simsize,
                                                                                  params.simsize);
        constexpr int NB_CLUSTERS = 6;

        std::array<int, NB_CLUSTERS> clusters;
        std::fill(clusters.begin(), clusters.end(), params.npart / NB_CLUSTERS);
        initial_condition::lennard_jones::RandomElementsInNClustersGenerator<DIMENSION, NB_CLUSTERS>
                elements_generator(clusters, params.seed, 100000);
        elements_generator.generate_elements(mesh_data.els, params.npart, &condition);
    }

    if (rank == 0) {
        std::cout << "==============================================" << std::endl;
        std::cout << "= Simulation is starting now...                 " << std::endl;
        std::cout << "= Parameters: " << std::endl;
        std::cout << "= Particles: " << params.npart << std::endl;
        std::cout << "= Seed: " << params.seed << std::endl;
        std::cout << "= PEs: " << params.world_size << std::endl;
        std::cout << "= Simulation size: " << params.simsize << std::endl;
        std::cout << "= Number of time-steps: " << params.nframes * params.npframe << std::endl;
        std::cout << "= Initial conditions: " << std::endl;
        std::cout << "= SIG:" << params.sig_lj << std::endl;
        std::cout << "= EPS:  " << params.eps_lj << std::endl;
        std::cout << "= Borders: collisions " << std::endl;
        std::cout << "= Gravity:  " << params.G << std::endl;
        std::cout << "= Temperature: " << params.T0 << std::endl;
        std::cout << "==============================================" << std::endl;
    }

    auto zz = zoltan_create_wrapper();
    zoltan_fn_init<DIMENSION>(zz, &mesh_data);

    rc = Zoltan_LB_Partition(zz,                 // input (all remaining fields are output)
                             &changes,           // 1 if partitioning was changed, 0 otherwise
                             &numGidEntries,     // Number of integers used for a global ID
                             &numLidEntries,     // Number of integers used for a local ID
                             &numImport,         // Number of vertices to be sent to me
                             &importGlobalGids,  // Global IDs of vertices to be sent to me
                             &importLocalGids,   // Local IDs of vertices to be sent to me
                             &importProcs,       // Process rank for source of each incoming vertex
                             &importToPart,      // New partition for each incoming vertex
                             &numExport,         // Number of vertices I must send to other processes
                             &exportGlobalGids,  // Global IDs of the vertices I must send
                             &exportLocalGids,   // Local IDs of the vertices I must send
                             &exportProcs,       // Process to which I send each of the vertices
                             &exportToPart);     // Partition to which each vertex will belong
    double xmin, ymin, zmin, xmax, ymax, zmax;

    std::vector<partitioning::geometric::Domain<DIMENSION>> domain_boundaries(nproc);

    for(int part = 0; part < nproc; ++part) {
        Zoltan_RCB_Box(zz, part, &dim, &xmin, &ymin, &zmin, &xmax, &ymax, &zmax);
        auto domain = partitioning::geometric::borders_to_domain<DIMENSION>(xmin, ymin, zmin,
                                                                            xmax, ymax, zmax, params.simsize);
        domain_boundaries[part] = domain;
    }

    load_balancing::geometric::migrate_zoltan<DIMENSION>(mesh_data.els, numImport, numExport,
                                                         exportProcs, exportGlobalGids, datatype, MPI_COMM_WORLD);

    std::shared_ptr<decision_making::Policy> lb_policy = std::make_shared<decision_making::RandomPolicy>(0.001, params.seed);

    simulate<DIMENSION>(fp, &mesh_data, zz,  lb_policy, &params, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    double t2 = MPI_Wtime();

    Zoltan_LB_Free_Part(&importGlobalGids, &importLocalGids,
                        &importProcs,      &importToPart);
    Zoltan_LB_Free_Part(&exportGlobalGids, &exportLocalGids,
                        &exportProcs,      &exportToPart);
    Zoltan_Destroy(&zz);

    if (fp) fclose(fp);

    if (rank == 0) printf("Simulation finished in %f seconds\n", (t2 - t1));

    MPI_Finalize();
    return 0;
}
