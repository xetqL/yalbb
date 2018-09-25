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
#include "../includes/ljpotential.hpp"


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
    std::unordered_map<long long, std::unique_ptr<std::vector<elements::Element<3> > > > plklist;

    std::array<elements::ElementRealType, 3> p1 = {39.972660,39.971722,39.992611}, v1 = {0,0,0}, a1 = v1;
    std::array<elements::ElementRealType, 3> p2 = {39.988892, 39.982147 , 39.991638};
    long long nb_c = std::ceil(40.0/(3.5f * params.sig_lj));
    long long pos1 = position_to_cell<3>(p1, 3.5f*params.sig_lj, nb_c, nb_c);
    long long pos2 = position_to_cell<3>(p2, 3.5f*params.sig_lj, nb_c, nb_c);
    std::vector<elements::Element<3>> l, r;
    l.push_back(elements::Element<3>::createc(p1, v1, 0, 0));
    std::cout << "number of cell = "<< (long long) std::pow(nb_c, 3) << std::endl;
    std::cout << "pos = "<< pos1 << std::endl;
    std::cout << "pos = "<< pos2 << std::endl;

    int err = lennard_jones::create_cell_linkedlist(1143, dto<elements::ElementRealType >(0.035), l, r, plklist);




    return 0;

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