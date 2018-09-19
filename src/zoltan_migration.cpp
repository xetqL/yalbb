//
// Created by xetql on 9/18/18.
//
#include <string>
#include <mpi.h>
#include <random>

#include <zoltan.h>
#include "../includes/zoltan_fn.hpp"
#include "../includes/initial_conditions.hpp"
#include "../includes/nbody_io.hpp"
#include "../includes/params.hpp"


int main(int argc, char** argv){
    constexpr int DIMENSION = 3;
    sim_param_t params;
    int rank, nproc, dim;
    float ver;
    MESH_DATA<DIMENSION> original_data, mesh_data;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    if (get_params(argc, argv, &params) != 0) {
        MPI_Finalize();
        return -1;
    }

    std::vector<int> caca;
    caca.push_back(10);
    caca.push_back(20);
    ZOLTAN_ID_PTR bite = (ZOLTAN_ID_PTR) &caca[0];
    std::cout << bite[0] <<std::endl;
    params.world_size = (unsigned int) nproc;

    if(!rank){
        std::shared_ptr<initial_condition::lennard_jones::RejectionCondition<DIMENSION>> condition;
        const int MAX_TRIAL = 100000;
        int NB_CLUSTERS;
        std::vector<int> clusters;
        using ElementGeneratorCfg = std::pair<std::shared_ptr<initial_condition::RandomElementsGenerator<DIMENSION>>, int>;

        condition = std::make_shared<initial_condition::lennard_jones::RejectionCondition<DIMENSION>>(
                &(mesh_data.els), params.sig_lj, params.sig_lj * params.sig_lj, params.T0, 0, 0, 0,
                params.simsize, params.simsize, params.simsize
        );
        auto gen = std::make_shared<initial_condition::lennard_jones::UniformRandomElementsGenerator<DIMENSION>>(params.seed, MAX_TRIAL);
        gen->generate_elements(mesh_data.els, params.npart, condition);
    }

    auto zz = zoltan_create_wrapper(true);

    zoltan_fn_init<DIMENSION>(zz, &mesh_data, true);

    int changes, numGidEntries, numLidEntries, numImport, numExport;
    ZOLTAN_ID_PTR importGlobalGids, importLocalGids, exportGlobalGids, exportLocalGids;
    int *importProcs, *importToPart, *exportProcs, *exportToPart;

    std::for_each(mesh_data.els.cbegin(), mesh_data.els.cend(), [&rank](auto p){std::cout << rank << " before: "<< p << std::endl;});

    MPI_Barrier(MPI_COMM_WORLD);
    Zoltan_LB_Partition(zz,                 // input (all remaining fields are output)
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

    std::for_each(mesh_data.els.cbegin(), mesh_data.els.cend(), [&rank](auto p){std::cout << rank << " after: "<< p << std::endl;});

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;

}