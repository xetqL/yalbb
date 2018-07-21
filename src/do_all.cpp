//
// Created by xetql on 7/21/18.
//
#include <string>
#include <mpi.h>
#include <random>

#include <zoltan.h>

#include "../includes/runners/simulator.hpp"
#include "../includes/initial_conditions.hpp"
#include "../includes/nbody_io.hpp"
#include "../includes/params.hpp"


template<int N>
std::vector<partitioning::geometric::Domain<N>>
retrieve_domain_boundaries(Zoltan_Struct *zz, int nproc, sim_param_t *params) {
    int dim;
    double xmin, ymin, zmin, xmax, ymax, zmax;
    std::vector<partitioning::geometric::Domain<N>> domain_boundaries(nproc);
    for (int part = 0; part < nproc; ++part) {
        Zoltan_RCB_Box(zz, part, &dim, &xmin, &ymin, &zmin, &xmax, &ymax, &zmax);
        auto domain = partitioning::geometric::borders_to_domain<N>(xmin, ymin, zmin, xmax, ymax, zmax,
                                                                    params->simsize);
        domain_boundaries[part] = domain;
    }
    return domain_boundaries;
}

int main(int argc, char **argv) {
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

    params.world_size = (unsigned int) nproc;

    //Define the output simulation name
    const std::string SIMULATION_STR_NAME = std::to_string(params.seed) +
                                            "-" + std::to_string(params.nframes) +"x"+ std::to_string(params.npframe) +
                                            "-" + std::to_string(params.world_size) +
                                            "-" + std::to_string(params.npart) +
                                            "-" + std::to_string((params.T0)) +
                                            "-" + std::to_string((params.G)) +
                                            "-" + std::to_string((params.simsize)) +
                                            "-" + std::to_string((params.eps_lj)) +
                                            "-" + std::to_string((params.sig_lj)) +
                                            "-" + std::to_string(params.dt) +
                                            "_" + std::to_string(params.particle_init_conf);

    int changes, numGidEntries, numLidEntries, numImport, numExport;
    ZOLTAN_ID_PTR importGlobalGids, importLocalGids, exportGlobalGids, exportLocalGids;
    int *importProcs, *importToPart, *exportProcs, *exportToPart;

    partitioning::CommunicationDatatype datatype = elements::register_datatype<DIMENSION>();

    int rc = Zoltan_Initialize(argc, argv, &ver);
    if (rc != ZOLTAN_OK) {
        MPI_Finalize();
        exit(0);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////START PARITCLE INITIALIZATION///////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    if (rank == 0) {
        initial_condition::lennard_jones::RejectionCondition<DIMENSION> condition(&(mesh_data.els),
                                                                                  params.sig_lj,
                                                                                  params.sig_lj * params.sig_lj,
                                                                                  params.T0,
                                                                                  0, 0, 0,
                                                                                  params.simsize,
                                                                                  params.simsize,
                                                                                  params.simsize);
        const int MAX_TRIAL = 100000;
        int NB_CLUSTERS;
        std::vector<int> clusters;
        using ElementGeneratorCfg = std::pair<std::shared_ptr<initial_condition::RandomElementsGenerator<DIMENSION>>, int>;
        std::queue<ElementGeneratorCfg> elements_generators;
        switch (params.particle_init_conf) {
            case 1: //uniformly distributed
                elements_generators.push(std::make_pair(
                        std::make_shared<initial_condition::lennard_jones::UniformRandomElementsGenerator<DIMENSION>>(
                                params.seed, MAX_TRIAL), params.npart));
                break;
            case 2: //Half full half empty
                elements_generators.push(std::make_pair(
                        std::make_shared<initial_condition::lennard_jones::HalfLoadedRandomElementsGenerator<DIMENSION>>(
                                params.simsize / 2, false, params.seed, MAX_TRIAL), params.npart));
                break;
            case 3: //Wall of particle
                elements_generators.push(std::make_pair(
                        std::make_shared<initial_condition::lennard_jones::ParticleWallElementsGenerator<DIMENSION>>(
                                params.simsize / 2, false, params.seed, MAX_TRIAL), params.npart));
                break;
            case 4: //cluster(s)
                NB_CLUSTERS = 1;
                clusters.resize(NB_CLUSTERS);
                std::fill(clusters.begin(), clusters.end(), params.npart);
                elements_generators.push(std::make_pair(
                        std::make_shared<initial_condition::lennard_jones::RandomElementsInNClustersGenerator<DIMENSION>>(
                                clusters, params.seed, MAX_TRIAL), params.npart));
                break;
            case 5: //custom various density
                NB_CLUSTERS = 2;
                clusters.resize(NB_CLUSTERS);
                std::fill(clusters.begin(), clusters.end(), params.npart / 4);
                elements_generators.push(std::make_pair(
                        std::make_shared<initial_condition::lennard_jones::RandomElementsInNClustersGenerator<DIMENSION>>(
                                clusters, params.seed, MAX_TRIAL), params.npart / 4));
                elements_generators.push(std::make_pair(
                        std::make_shared<initial_condition::lennard_jones::HalfLoadedRandomElementsGenerator<DIMENSION>>(
                                params.simsize / 10, false, params.seed, MAX_TRIAL), 3 * params.npart / 4));
                break;
            default:
                elements_generators.push(std::make_pair(
                        std::make_shared<initial_condition::lennard_jones::UniformRandomElementsGenerator<DIMENSION>>(
                                params.seed, MAX_TRIAL), params.npart));
        }
        while (!elements_generators.empty()) {
            ElementGeneratorCfg el_gen = elements_generators.front();
            el_gen.first->generate_elements(mesh_data.els, el_gen.second, &condition);
            elements_generators.pop();
            std::cout << el_gen.second << std::endl;
        }
    }
    original_data = mesh_data; //copy data elsewhere for future use

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////FINISHED PARITCLE INITIALIZATION///////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    const std::string DATASET_FILENAME = SIMULATION_STR_NAME+".dataset";
    auto zz = zoltan_create_wrapper();

    zoltan_fn_init<DIMENSION>(zz, &mesh_data);
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

    std::vector<partitioning::geometric::Domain<DIMENSION>>
            domain_boundaries = retrieve_domain_boundaries<DIMENSION>(zz, nproc, &params);
    load_balancing::geometric::migrate_zoltan<DIMENSION>(mesh_data.els, numImport, numExport,
                                                         exportProcs, exportGlobalGids, datatype, MPI_COMM_WORLD);
    Zoltan_LB_Free_Part(&importGlobalGids, &importLocalGids, &importProcs, &importToPart);
    Zoltan_LB_Free_Part(&exportGlobalGids, &exportLocalGids, &exportProcs, &exportToPart);

    if (!rank) std::cout << "Run A* simulation to get optimal path..." << std::endl;
    auto astar_optimal_paths = Astar_runner<DIMENSION>(&mesh_data, zz, &params, MPI_COMM_WORLD);

    std::ofstream dataset;
    if(!rank && file_exists(DATASET_FILENAME)) {
        std::remove(DATASET_FILENAME.c_str());
    }
    for (auto const &solution : astar_optimal_paths)
        metric::io::write_dataset(dataset, DATASET_FILENAME, solution, rank,
                                  (*(std::next(solution.end(), -1)))->cost());

    Zoltan_Destroy(&zz); //destroy A* for zoltan
    MPI_Barrier(MPI_COMM_WORLD);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////FINISHED A* COMPUTATION, START COMPARISON WITH HEURISTICS//////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if (!rank) std::cout << "Start to compare best path with heuristics..." << std::endl;

    std::shared_ptr<decision_making::Policy> lb_policy;
    const std::string RESULT_FILENAME = SIMULATION_STR_NAME+".result";
    if(!rank && file_exists(RESULT_FILENAME)) {
        std::remove(RESULT_FILENAME.c_str());
    }
    for (unsigned int lb_policy_idx = 1; lb_policy_idx <= 6; ++lb_policy_idx) {
        mesh_data = original_data; //recover data from the clean copy
        switch (lb_policy_idx) {
            case 1:
                lb_policy = std::make_shared<decision_making::RandomPolicy>(0.1, params.seed);
                break;
            case 2://TODO: threshold should be a parameter
                lb_policy = std::make_shared<decision_making::ThresholdHeuristicPolicy>(0.6);
                break;
            case 3:
                lb_policy = std::make_shared<decision_making::PeriodicPolicy>(25);
                break;
            case 4://load the file created above
                lb_policy = std::make_shared<decision_making::PeriodicPolicy>(100);
                break;
            case 5://load the file created above
                lb_policy = std::make_shared<decision_making::InFilePolicy>(
                        DATASET_FILENAME, params.nframes, params.npframe);
                break;
            default:
                lb_policy = std::make_shared<decision_making::NoLBPolicy>();
        }

        zz = zoltan_create_wrapper();
        zoltan_fn_init<DIMENSION>(zz, &mesh_data);
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

        domain_boundaries = retrieve_domain_boundaries<DIMENSION>(zz, nproc, &params);

        load_balancing::geometric::migrate_zoltan<DIMENSION>(mesh_data.els, numImport, numExport,
                                                             exportProcs, exportGlobalGids, datatype, MPI_COMM_WORLD);
        Zoltan_LB_Free_Part(&importGlobalGids, &importLocalGids, &importProcs, &importToPart);
        Zoltan_LB_Free_Part(&exportGlobalGids, &exportLocalGids, &exportProcs, &exportToPart);

        auto time_spent = simulate<DIMENSION>(NULL, &mesh_data, zz,  lb_policy, &params, MPI_COMM_WORLD);

        if(!rank) {
            std::ofstream result;
            result.open(RESULT_FILENAME, std::ofstream::app | std::ofstream::out);
            result << time_spent << std::endl;
            result.close();
        }
        Zoltan_Destroy(&zz);
        MPI_Barrier(MPI_COMM_WORLD); //finished with this LB heuristic... go next
    }

    MPI_Finalize();
    return 0;
}