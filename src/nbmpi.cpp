#include <string>
#include <mpi.h>
#include <random>

#include "../includes/runners/simulator.hpp"
#include "../includes/initial_conditions.hpp"
#include "../includes/runners/shortest_path.hpp"

int main(int argc, char** argv) {

    constexpr int DIMENSION = 3;
    sim_param_t params;
    int rank, nproc;
    float ver;
    MESH_DATA<DIMENSION> mesh_data;

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm APP_COMM;
    MPI_Comm_dup(MPI_COMM_WORLD, &APP_COMM);

    if (get_params(argc, argv, &params) != 0) {
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    params.world_size = nproc;
    params.simsize = std::ceil(params.simsize / params.rc) * params.rc;
    MPI_Bcast(&params.seed, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        print_params(params);
    }

    if(Zoltan_Initialize(argc, argv, &ver) != ZOLTAN_OK) {
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }
    auto zz = zoltan_create_wrapper(APP_COMM, ENABLE_AUTOMATIC_MIGRATION);

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
            elements::import_from_file<DIMENSION, Real>(IMPORT_FILENAME, mesh_data.els);
            std::cout << "Done !" << std::endl;
        } else {
            std::cout << "Generating data ..." << std::endl;
            std::shared_ptr<initial_condition::lj::RejectionCondition<DIMENSION>> condition;
            const int MAX_TRIAL = 100000;
            int NB_CLUSTERS;
            std::vector<int> clusters;
            using ElementGeneratorCfg = std::pair<std::shared_ptr<initial_condition::RandomElementsGenerator<DIMENSION>>, int>;
            std::queue<ElementGeneratorCfg> elements_generators;
            switch (params.particle_init_conf) {
                case 1: //uniformly distributed
                    condition = std::make_shared<initial_condition::lj::RejectionCondition<DIMENSION>>(
                            &(mesh_data.els), params.sig_lj, params.sig_lj * params.sig_lj, params.T0, 0, 0, 0,
                            params.simsize, params.simsize, params.simsize, &params
                    );
                    elements_generators.push(std::make_pair(
                            std::make_shared<initial_condition::lj::UniformRandomElementsGenerator<DIMENSION>>(
                                    params.seed, MAX_TRIAL), params.npart));
                    break;
                case 2: //Half full half empty
                    condition = std::make_shared<initial_condition::lj::RejectionCondition<DIMENSION>>(
                            &(mesh_data.els), params.sig_lj, params.sig_lj * params.sig_lj, params.T0, 0, 0, 0,
                            params.simsize, params.simsize, params.simsize, &params
                    );
                    elements_generators.push(std::make_pair(
                            std::make_shared<initial_condition::lj::HalfLoadedRandomElementsGenerator<DIMENSION>>(
                                    params.simsize / 2, false, params.seed, MAX_TRIAL), params.npart));
                    break;
                case 3: //Wall of particle
                    condition = std::make_shared<initial_condition::lj::RejectionCondition<DIMENSION>>(
                            &(mesh_data.els), params.sig_lj, params.sig_lj * params.sig_lj, params.T0, 0, 0, 0,
                            params.simsize, params.simsize, params.simsize, &params
                    );
                    elements_generators.push(std::make_pair(
                            std::make_shared<initial_condition::lj::ParticleWallElementsGenerator<DIMENSION>>(
                                    params.simsize * 0.99, false, params.seed, MAX_TRIAL), params.npart));
                    break;
                case 4: //cluster(s)
                    condition = std::make_shared<initial_condition::lj::RejectionCondition<DIMENSION>>(
                            &(mesh_data.els), params.sig_lj, 6.25 * params.sig_lj * params.sig_lj, params.T0, 0, 0, 0,
                            params.simsize, params.simsize, params.simsize, &params
                    );
                    NB_CLUSTERS = 1;
                    clusters.resize(NB_CLUSTERS);
                    std::fill(clusters.begin(), clusters.end(), params.npart);
                    elements_generators.push(std::make_pair(
                            std::make_shared<initial_condition::lj::RandomElementsInNClustersGenerator<DIMENSION>>(
                                    clusters, params.seed, MAX_TRIAL), params.npart));
                    break;
                case 5: //custom various density
                    condition = std::make_shared<initial_condition::lj::RejectionCondition<DIMENSION>>(
                            &(mesh_data.els), params.sig_lj, params.sig_lj * params.sig_lj, params.T0, 0, 0, 0,
                            params.simsize, params.simsize, params.simsize, &params
                    );
                    NB_CLUSTERS = 2;
                    clusters.resize(NB_CLUSTERS);
                    std::fill(clusters.begin(), clusters.end(), params.npart / 4);
                    elements_generators.push(std::make_pair(
                            std::make_shared<initial_condition::lj::RandomElementsInNClustersGenerator<DIMENSION>>(
                                    clusters, params.seed, MAX_TRIAL), params.npart / 4));
                    elements_generators.push(std::make_pair(
                            std::make_shared<initial_condition::lj::HalfLoadedRandomElementsGenerator<DIMENSION>>(
                                    params.simsize / 10, false, params.seed, MAX_TRIAL), 3 * params.npart / 4));
                    break;
                case 6: //custom various density
                    condition = std::make_shared<initial_condition::lj::RejectionCondition<DIMENSION>>(
                            &(mesh_data.els), params.sig_lj, params.sig_lj * params.sig_lj, params.T0, 0, 0, 0,
                            params.simsize, params.simsize, params.simsize, &params
                    );
                    NB_CLUSTERS = 1;
                    clusters.resize(NB_CLUSTERS);
                    std::fill(clusters.begin(), clusters.end(), params.npart);
                    elements_generators.push(std::make_pair(
                            std::make_shared<initial_condition::lj::RandomElementsInNClustersGenerator<DIMENSION>>(
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

    auto original_data = mesh_data;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////FINISHED PARITCLE INITIALIZATION///////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    using namespace decision_making;

    /* Experiment 2 */
    {
        mesh_data = original_data;
        auto load_balancer = Zoltan_Copy(zz);

        PAR_START_TIMER(lb_time_spent, APP_COMM);
        Zoltan_Do_LB(&mesh_data, load_balancer);
        PAR_END_TIMER(lb_time_spent, APP_COMM);

        if(!rank) std::cout << "Branch and Bound: Computation is starting." << std::endl;
        auto solution  = simulate_using_shortest_path<DIMENSION>(&mesh_data, load_balancer, &params, APP_COMM);
        if(!rank)
        {
            std::cout << std::fixed << std::setprecision(6) << solution.back()->cost() <<  " BaB LB" << std::endl;
            std::vector<Time> cum;
            std::transform(solution.begin(), solution.end(), std::back_inserter(cum), [](auto v){return v->stats.get_cumulative_load_imbalance_slowdown();});
            std::cout << cum << std::endl;
            std::vector<int> dec;
            std::transform(solution.begin(), solution.end(), std::back_inserter(dec), [](auto v){return v->decision == DoLB;});
            std::cout << dec << std::endl;
        }

    }

    {   /* Experiment 1 */

        mesh_data = original_data;
        auto load_balancer = Zoltan_Copy(zz);

        IterationStatistics it_stats(nproc);

        PAR_START_TIMER(lb_time_spent, APP_COMM);
        Zoltan_Do_LB(&mesh_data, load_balancer);
        PAR_END_TIMER(lb_time_spent, APP_COMM);

        MPI_Allreduce(&lb_time_spent, it_stats.get_lb_time_ptr(), 1, MPI_TIME, MPI_MAX, APP_COMM);

        if(!rank) std::cout << "SIM: Computation is starting." << std::endl;
        PolicyRunner<ThresholdPolicy> lb_policy(&it_stats,
                [](IterationStatistics* stats){ return stats->get_cumulative_load_imbalance_slowdown(); },// get data func
                [](IterationStatistics* stats){ return stats->compute_avg_lb_time(); });                  // get threshold func

        auto [t, cum, dec] = simulate<DIMENSION>(&mesh_data, load_balancer, std::move(lb_policy), &params, &it_stats, APP_COMM);

        if(!rank){
            std::cout << std::fixed << std::setprecision(6) << t << " Threshold LB" << std::endl;
            std::cout << cum << std::endl;
            std::cout << dec << std::endl;
        }

        Zoltan_Destroy(&load_balancer);
    }




    Zoltan_Destroy(&zz);

    MPI_Finalize();

    return 0;

}
