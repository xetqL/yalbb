#include <string>
#include <mpi.h>
#include <random>

#include "../includes/runners/simulator.hpp"
#include "../includes/initial_conditions.hpp"


int main(int argc, char** argv) {

    constexpr int DIMENSION = 3;
    sim_param_t params;
    FILE* fp = NULL;
    int rank, nproc, dim;
    float ver;
    MESH_DATA<DIMENSION> mesh_data;

    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    if (get_params(argc, argv, &params) != 0) {
        MPI_Finalize();
        return -1;
    }

    MPI_Bcast(&params.seed, 1, MPI_INT, 0, MPI_COMM_WORLD);

    params.world_size = nproc;

    if (rank == 0 && params.record) {
        fp = fopen(params.fname, "w");
    }

    int changes, numGidEntries, numLidEntries, numImport, numExport;
    ZOLTAN_ID_PTR importGlobalGids, importLocalGids, exportGlobalGids, exportLocalGids;
    int *importProcs, *importToPart, *exportProcs, *exportToPart;

    CommunicationDatatype datatype = elements::register_datatype<DIMENSION>();

    int rc = Zoltan_Initialize(argc, argv, &ver);

    if(rc != ZOLTAN_OK) {
        MPI_Finalize();
        exit(0);
    }

    params.simsize = std::ceil(params.simsize / params.rc) * params.rc;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////START PARITCLE INITIALIZATION///////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if (rank == 0) {
        const std::string IMPORT_FILENAME
                = std::to_string(params.npart) + "-" +
                  std::to_string(params.particle_init_conf) + "-" +
                  std::to_string(params.simsize) + ".particles";
        if(file_exists(IMPORT_FILENAME)) {
            std::cout << "importing from file ..." << std::endl;
            elements::import_from_file<DIMENSION, elements::ElementRealType >(IMPORT_FILENAME, mesh_data.els);
            std::cout << "Done !" << std::endl;
        } else {
            std::cout << "Generating data ..." << std::endl;
            std::shared_ptr<initial_condition::lennard_jones::RejectionCondition<DIMENSION>> condition;
            const int MAX_TRIAL = 100000;
            int NB_CLUSTERS;

            std::vector<int> clusters;
            using ElementGeneratorCfg = std::pair<std::shared_ptr<initial_condition::RandomElementsGenerator<DIMENSION>>, int>;
            std::queue<ElementGeneratorCfg> elements_generators;
            switch (params.particle_init_conf) {
                case 1: //uniformly distributed
                    condition = std::make_shared<initial_condition::lennard_jones::RejectionCondition<DIMENSION>>(
                            &(mesh_data.els), params.sig_lj, params.sig_lj * params.sig_lj, params.T0, 0, 0, 0,
                            params.simsize, params.simsize, params.simsize, &params
                    );
                    elements_generators.push(std::make_pair(
                            std::make_shared<initial_condition::lennard_jones::UniformRandomElementsGenerator<DIMENSION>>(
                                    params.seed, MAX_TRIAL), params.npart));
                    break;
                case 2: //Half full half empty
                    condition = std::make_shared<initial_condition::lennard_jones::RejectionCondition<DIMENSION>>(
                            &(mesh_data.els), params.sig_lj, params.sig_lj * params.sig_lj, params.T0, 0, 0, 0,
                            params.simsize, params.simsize, params.simsize, &params
                    );
                    elements_generators.push(std::make_pair(
                            std::make_shared<initial_condition::lennard_jones::HalfLoadedRandomElementsGenerator<DIMENSION>>(
                                    params.simsize / 2, false, params.seed, MAX_TRIAL), params.npart));
                    break;
                case 3: //Wall of particle
                    condition = std::make_shared<initial_condition::lennard_jones::RejectionCondition<DIMENSION>>(
                            &(mesh_data.els), params.sig_lj, params.sig_lj * params.sig_lj, params.T0, 0, 0, 0,
                            params.simsize, params.simsize, params.simsize, &params
                    );
                    elements_generators.push(std::make_pair(
                            std::make_shared<initial_condition::lennard_jones::ParticleWallElementsGenerator<DIMENSION>>(
                                    params.simsize * 0.99, false, params.seed, MAX_TRIAL), params.npart));
                    break;
                case 4: //cluster(s)
                    condition = std::make_shared<initial_condition::lennard_jones::RejectionCondition<DIMENSION>>(
                            &(mesh_data.els), params.sig_lj, 6.25 * params.sig_lj * params.sig_lj, params.T0, 0, 0, 0,
                            params.simsize, params.simsize, params.simsize, &params
                    );
                    NB_CLUSTERS = 1;
                    clusters.resize(NB_CLUSTERS);
                    std::fill(clusters.begin(), clusters.end(), params.npart);
                    elements_generators.push(std::make_pair(
                            std::make_shared<initial_condition::lennard_jones::RandomElementsInNClustersGenerator<DIMENSION>>(
                                    clusters, params.seed, MAX_TRIAL), params.npart));
                    break;
                case 5: //custom various density
                    condition = std::make_shared<initial_condition::lennard_jones::RejectionCondition<DIMENSION>>(
                            &(mesh_data.els), params.sig_lj, params.sig_lj * params.sig_lj, params.T0, 0, 0, 0,
                            params.simsize, params.simsize, params.simsize, &params
                    );
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
                case 6: //custom various density
                    condition = std::make_shared<initial_condition::lennard_jones::RejectionCondition<DIMENSION>>(
                            &(mesh_data.els), params.sig_lj, params.sig_lj * params.sig_lj, params.T0, 0, 0, 0,
                            params.simsize, params.simsize, params.simsize, &params
                    );
                    NB_CLUSTERS = 1;
                    clusters.resize(NB_CLUSTERS);
                    std::fill(clusters.begin(), clusters.end(), params.npart);
                    elements_generators.push(std::make_pair(
                            std::make_shared<initial_condition::lennard_jones::RandomElementsInNClustersGenerator<DIMENSION>>(
                                    clusters, params.seed, MAX_TRIAL), params.npart));
                    break;
                default:
                    MPI_Finalize();
                    throw std::runtime_error("Unknown particle distribution.");
            }
            while (!elements_generators.empty()) {
                ElementGeneratorCfg el_gen = elements_generators.front();
                el_gen.first->generate_elements(mesh_data.els, el_gen.second, condition);
                elements_generators.pop();
                std::cout << el_gen.second << std::endl;
            }
            std::cout << "Done !" << std::endl;
        }
    }


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////FINISHED PARITCLE INITIALIZATION///////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    if (rank == 0) {
        std::cout << "==============================================" << std::endl;
        std::cout << "= Simulation is starting now...                 " << std::endl;
        std::cout << "= Parameters: " << std::endl;
        std::cout << "= Particles: " << params.npart << std::endl;
        std::cout << "= Seed: " << params.seed << std::endl;
        std::cout << "= PEs: " << params.world_size << std::endl;
        std::cout << "= Simulation size: " << params.simsize << std::endl;
        std::cout << "= Number of time-steps: " << params.nframes << "x" << params.npframe << std::endl;
        std::cout << "= Initial conditions: " << std::endl;
        std::cout << "= SIG:" << params.sig_lj << std::endl;
        std::cout << "= EPS:  " << params.eps_lj << std::endl;
        std::cout << "= Borders: collisions " << std::endl;
        std::cout << "= Gravity:  " << params.G << std::endl;
        std::cout << "= Temperature: " << params.T0 << std::endl;
        std::cout << "==============================================" << std::endl;
    }

    auto zz = zoltan_create_wrapper(ENABLE_AUTOMATIC_MIGRATION);

    zoltan_fn_init<DIMENSION>(zz, &mesh_data, ENABLE_AUTOMATIC_MIGRATION);

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

    Zoltan_LB_Free_Part(&importGlobalGids, &importLocalGids, &importProcs, &importToPart);
    Zoltan_LB_Free_Part(&exportGlobalGids, &exportLocalGids, &exportProcs, &exportToPart);

    using namespace decision_making;
    PolicyRunner<NoLBPolicy> lb_policy;

    PAR_START_TIMER(time_spent, MPI_COMM_WORLD);
    simulate<DIMENSION>(nullptr, &mesh_data, zz,  std::move(lb_policy), &params, MPI_COMM_WORLD, ENABLE_AUTOMATIC_MIGRATION);
    PAR_END_TIMER(time_spent, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    Zoltan_LB_Free_Part(&importGlobalGids, &importLocalGids,
                        &importProcs,      &importToPart);
    Zoltan_LB_Free_Part(&exportGlobalGids, &exportLocalGids,
                        &exportProcs,      &exportToPart);
    Zoltan_Destroy(&zz);

    if (fp) fclose(fp);

    MPI_Finalize();

    return 0;

}
