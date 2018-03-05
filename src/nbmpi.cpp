#include <string>
#include <mpi.h>
#include <random>

#include <librl/agents/RLAgentFactory.hpp>
#include <librl/approximators/FunctionApproximator.hpp>
#include <librl/approximators/MLPActionValueApproximator.hpp>
#include <librl/policies/Policies.hpp>
#include <zoltan.h>

#include "../includes/box_runner.hpp"
#include "../includes/zoltan_fn.hpp"

int main(int argc, char** argv) {
    sim_param_t params;
    FILE* fp = NULL;
    int npart;
    int rank, nproc, dim;
    float ver;
    MESH_DATA mesh_data;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    if (get_params(argc, argv, &params) != 0) {
        MPI_Finalize();
        return -1;
    }

    if(params.world_size != (size_t) nproc) {
        std::cout << "World size does not match the expected world size: World=" <<nproc<<" Expected="<< params.world_size << std::endl;
        MPI_Finalize();
        exit(0);
    }
    if (rank == 0) {
        fp = fopen(params.fname, "w");
    }
    //std::vector<elements::Element<2>> elements(params.npart);

    int rc = Zoltan_Initialize(argc, argv, &ver);

    if(rc != ZOLTAN_OK){
        MPI_Finalize();
        exit(0);
    }

    auto zz = zoltan_create_wrapper();

    init_mesh_data(rank, nproc, &mesh_data, &params);

    zoltan_fn_init(zz, &mesh_data);

    int changes, numGidEntries, numLidEntries, numImport, numExport;
    ZOLTAN_ID_PTR importGlobalGids, importLocalGids, exportGlobalGids, exportLocalGids;
    int *importProcs, *importToPart, *exportProcs, *exportToPart;

    zoltan_fn_init(zz, &mesh_data);
    rc = Zoltan_LB_Partition(zz,                 /* input (all remaining fields are output) */
                             &changes,           /* 1 if partitioning was changed, 0 otherwise */
                             &numGidEntries,     /* Number of integers used for a global ID */
                             &numLidEntries,     /* Number of integers used for a local ID */
                             &numImport,         /* Number of vertices to be sent to me */
                             &importGlobalGids,  /* Global IDs of vertices to be sent to me */
                             &importLocalGids,   /* Local IDs of vertices to be sent to me */
                             &importProcs,       /* Process rank for source of each incoming vertex */
                             &importToPart,      /* New partition for each incoming vertex */
                             &numExport,         /* Number of vertices I must send to other processes*/
                             &exportGlobalGids,  /* Global IDs of the vertices I must send */
                             &exportLocalGids,   /* Local IDs of the vertices I must send */
                             &exportProcs,       /* Process to which I send each of the vertices */
                             &exportToPart);     /* Partition to which each vertex will belong */

    Zoltan_LB_Free_Part(&importGlobalGids, &importLocalGids,
                        &importProcs, &importToPart);
    Zoltan_LB_Free_Part(&exportGlobalGids, &exportLocalGids,
                        &exportProcs, &exportToPart);

    double xmin, ymin, zmin, xmax, ymax, zmax;
    std::vector<partitioning::geometric::Domain<2>> domain_boundaries(nproc);
    for(int part = 0; part < nproc; ++part){
        Zoltan_RCB_Box(zz, part, &dim, &xmin, &ymin, &zmin, &xmax, &ymax, &zmax);
        auto domain = partitioning::geometric::borders_to_domain<2>(xmin, ymin, zmin, xmax, ymax, zmax, params.simsize);
        domain_boundaries[part] = domain;
    }
    partitioning::CommunicationDatatype datatype = elements::register_datatype<2>();

    load_balancing::geometric::migrate_particles<2>(mesh_data.els, domain_boundaries, datatype, MPI_COMM_WORLD);
        double t1 = MPI_Wtime();

    //run_box<2>(fp, params.npframe, params.nframes, params.dt, elements, domain_boundaries, load_balancer, &params);
    zoltan_run_box(fp, &mesh_data, zz, &params, MPI_COMM_WORLD);

    Zoltan_Destroy(&zz);

    MPI_Barrier(MPI_COMM_WORLD);

    double t2 = MPI_Wtime();

    if (fp) fclose(fp);

    if (rank == 0) printf("Simulation finished in %f seconds\n", (t2 - t1));

    MPI_Finalize();
    return 0;
}