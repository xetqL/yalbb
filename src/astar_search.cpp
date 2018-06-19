//
// Created by xetql on 23.05.18.
//

#include <string>
#include <mpi.h>
#include <random>

#include <zoltan.h>
#include "../includes/box_runner.hpp"
#include "../includes/initial_conditions.hpp"

int main(int argc, char **argv) {
    constexpr int DIMENSION = 3;
    sim_param_t params;
    FILE *fp = NULL;
    int rank, nproc, dim;
    float ver;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    if (get_params(argc, argv, &params) != 0) {
        MPI_Finalize();
        return -1;
    }

    params.world_size = nproc;

    if (rank == 0 && params.record) {
        fp = fopen(params.fname, "w");
    }

    int changes, numGidEntries, numLidEntries, numImport, numExport;
    ZOLTAN_ID_PTR importGlobalGids, importLocalGids, exportGlobalGids, exportLocalGids;
    int *importProcs, *importToPart, *exportProcs, *exportToPart;

    partitioning::CommunicationDatatype datatype = elements::register_datatype<DIMENSION>();

    params.verbose = false;
    MESH_DATA<DIMENSION> mesh_data_original;

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

    int rc = Zoltan_Initialize(argc, argv, &ver);
    if (rc != ZOLTAN_OK) {
        MPI_Finalize();
        exit(0);
    }

    MESH_DATA<DIMENSION> _mesh_data;
    if(rank == 0){
        initial_condition::lennard_jones::RejectionCondition<DIMENSION> condition(&(_mesh_data.els),
                                                                                  params.sig_lj,
                                                                                  params.sig_lj*params.sig_lj,
                                                                                  params.T0,
                                                                                  0, 0, 0,
                                                                                  params.simsize,
                                                                                  params.simsize,
                                                                                  params.simsize);
        constexpr int NB_CLUSTERS = 1;
        std::array<int, NB_CLUSTERS> clusters;
        std::fill(clusters.begin(), clusters.end(), params.npart / NB_CLUSTERS);
        initial_condition::lennard_jones::RandomElementsInNClustersGenerator<DIMENSION, NB_CLUSTERS>
                elements_generator(clusters, params.seed, 100000);
        elements_generator.generate_elements(_mesh_data.els, params.npart, &condition);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MESH_DATA<DIMENSION> mesh_data = _mesh_data;

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

    for (int part = 0; part < nproc; ++part) {
        Zoltan_RCB_Box(zz, part, &dim, &xmin, &ymin, &zmin, &xmax, &ymax, &zmax);
        auto domain = partitioning::geometric::borders_to_domain<DIMENSION>(xmin, ymin, zmin,
                                                                            xmax, ymax, zmax, params.simsize);
        domain_boundaries[part] = domain;
    }

    load_balancing::geometric::migrate_zoltan<DIMENSION>(mesh_data.els, numImport, numExport,
                                                         exportProcs, exportGlobalGids, datatype, MPI_COMM_WORLD);
#ifdef IDASTAR_IMPL
    if(!rank) std::cout << "IDA* + Custom memory opt. implementation of A*" << std::endl;
    auto res = IDAstar_runner<DIMENSION>(&mesh_data, zz, &params, MPI_COMM_WORLD);
#else
#ifdef ASTAR_IMPL
    if(!rank) std::cout << "Standard implementation of A*" << std::endl;
    auto res = Astar_runner<DIMENSION>(&mesh_data, zz, &params, MPI_COMM_WORLD);
#else
    if(!rank) std::cout << "Standard implementation of A*" << std::endl;
    auto res = Astar_runner<DIMENSION>(&mesh_data, zz, &params, MPI_COMM_WORLD);
#endif
#endif
    std::ofstream dataset;
    const std::string DATASET_FILENAME = "lj_dataset-" + std::to_string(params.seed) +
                                         "-" + std::to_string(params.world_size) +
                                         "-" + std::to_string(params.npart) +
                                         "-" + std::to_string((params.T0)) +
                                         "-" + std::to_string((params.G)) +
                                         "-" + std::to_string((params.simsize)) +
                                         "-" + std::to_string((params.eps_lj)) +
                                         "-" + std::to_string((params.sig_lj)) + ".data";
    metric::io::write_dataset(dataset, DATASET_FILENAME, res, rank, 0);

    MPI_Barrier(MPI_COMM_WORLD);

    Zoltan_LB_Free_Part(&importGlobalGids, &importLocalGids,
                        &importProcs, &importToPart);
    Zoltan_LB_Free_Part(&exportGlobalGids, &exportLocalGids,
                        &exportProcs, &exportToPart);
    Zoltan_Destroy(&zz);

    if (fp) fclose(fp);

    MPI_Finalize();
    return 0;
}
