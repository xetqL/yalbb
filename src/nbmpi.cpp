#include <string>
#include <mpi.h>
#include <random>

#include "../includes/runners/simulator.hpp"
#include "../includes/initial_conditions.hpp"
#include "../includes/runners/shortest_path.hpp"

int main(int argc, char** argv) {

    constexpr int N = 3;
    sim_param_t params;
    int rank, nproc;
    float ver;
    MESH_DATA<elements::Element<N>> mesh_data;

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
            elements::import_from_file<N, Real>(IMPORT_FILENAME, mesh_data.els);
            std::cout << "Done !" << std::endl;
        } else {
            std::cout << "Generating data ..." << std::endl;
            std::shared_ptr<initial_condition::lj::RejectionCondition<N>> condition;
            const int MAX_TRIAL = 100000;
            int NB_CLUSTERS;
            std::vector<int> clusters;
            using ElementGeneratorCfg = std::pair<std::shared_ptr<initial_condition::RandomElementsGenerator<N>>, int>;
            std::queue<ElementGeneratorCfg> elements_generators;
            switch (params.particle_init_conf) {
                case 1: //uniformly distributed
                    condition = std::make_shared<initial_condition::lj::RejectionCondition<N>>(
                            &(mesh_data.els), params.sig_lj, params.sig_lj * params.sig_lj, params.T0, 0, 0, 0,
                            params.simsize, params.simsize, params.simsize, &params
                    );
                    elements_generators.push(std::make_pair(
                            std::make_shared<initial_condition::lj::UniformRandomElementsGenerator<N>>(
                                    params.seed, MAX_TRIAL), params.npart));
                    break;
                case 2: //Half full half empty
                    condition = std::make_shared<initial_condition::lj::RejectionCondition<N>>(
                            &(mesh_data.els), params.sig_lj, params.sig_lj * params.sig_lj, params.T0, 0, 0, 0,
                            params.simsize, params.simsize, params.simsize, &params
                    );
                    elements_generators.push(std::make_pair(
                            std::make_shared<initial_condition::lj::HalfLoadedRandomElementsGenerator<N>>(
                                    params.simsize / 2, false, params.seed, MAX_TRIAL), params.npart));
                    break;
                case 3: //Wall of particle
                    condition = std::make_shared<initial_condition::lj::RejectionCondition<N>>(
                            &(mesh_data.els), params.sig_lj, params.sig_lj * params.sig_lj, params.T0, 0, 0, 0,
                            params.simsize, params.simsize, params.simsize, &params
                    );
                    elements_generators.push(std::make_pair(
                            std::make_shared<initial_condition::lj::ParticleWallElementsGenerator<N>>(
                                    params.simsize * 0.99, false, params.seed, MAX_TRIAL), params.npart));
                    break;
                case 4: //cluster(s)
                    condition = std::make_shared<initial_condition::lj::RejectionCondition<N>>(
                            &(mesh_data.els), params.sig_lj, 6.25 * params.sig_lj * params.sig_lj, params.T0, 0, 0, 0,
                            params.simsize, params.simsize, params.simsize, &params
                    );
                    NB_CLUSTERS = 1;
                    clusters.resize(NB_CLUSTERS);
                    std::fill(clusters.begin(), clusters.end(), params.npart);
                    elements_generators.push(std::make_pair(
                            std::make_shared<initial_condition::lj::RandomElementsInNClustersGenerator<N>>(
                                    clusters, params.seed, MAX_TRIAL), params.npart));
                    break;
                case 5: //custom various density
                    condition = std::make_shared<initial_condition::lj::RejectionCondition<N>>(
                            &(mesh_data.els), params.sig_lj, params.sig_lj * params.sig_lj, params.T0, 0, 0, 0,
                            params.simsize, params.simsize, params.simsize, &params
                    );
                    NB_CLUSTERS = 2;
                    clusters.resize(NB_CLUSTERS);
                    std::fill(clusters.begin(), clusters.end(), params.npart / 4);
                    elements_generators.push(std::make_pair(
                            std::make_shared<initial_condition::lj::RandomElementsInNClustersGenerator<N>>(
                                    clusters, params.seed, MAX_TRIAL), params.npart / 4));
                    elements_generators.push(std::make_pair(
                            std::make_shared<initial_condition::lj::HalfLoadedRandomElementsGenerator<N>>(
                                    params.simsize / 10, false, params.seed, MAX_TRIAL), 3 * params.npart / 4));
                    break;
                case 6: //custom various density
                    condition = std::make_shared<initial_condition::lj::RejectionCondition<N>>(
                            &(mesh_data.els), params.sig_lj, params.sig_lj * params.sig_lj, params.T0, 0, 0, 0,
                            params.simsize, params.simsize, params.simsize, &params
                    );
                    NB_CLUSTERS = 1;
                    clusters.resize(NB_CLUSTERS);
                    std::fill(clusters.begin(), clusters.end(), params.npart);
                    elements_generators.push(std::make_pair(
                            std::make_shared<initial_condition::lj::RandomElementsInNClustersGenerator<N>>(
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

    auto zlb = Zoltan_Copy(zz);

    auto boxIntersectFunc   = [zlb](double x1, double y1, double z1, double x2, double y2, double z2, int* PEs, int* num_found){
        Zoltan_LB_Box_Assign(zlb, x1, y1, z1, x2, y2, z2, PEs, num_found);
    };
    auto pointAssignFunc    = [zlb](const elements::Element<N>& e, int* PE){
        auto pos_in_double = get_as_double_array<N>(e.position);
        Zoltan_LB_Point_Assign(zlb, &pos_in_double.front(), PE);
    };
    auto doLoadBalancingFunc= [zlb](MESH_DATA<elements::Element<N>>* mesh_data){ Zoltan_Do_LB(mesh_data, zlb); };
    auto getPositionPtrFunc = [](elements::Element<N>& e) {
        return &e.position;
    };
    auto getVelocityPtrFunc = [](elements::Element<N>& e) { return &e.velocity; };
    auto getForceFunc       = [eps_lj=params.eps_lj, sig=params.sig_lj](const auto& receiver, const auto& source){
        Real delta = 0.0;
        const Real sig2 = sig*sig;
        std::array<Real, N> delta_dim;
        std::array<Real, N> force;
        for (int dim = 0; dim < 3; ++dim)
            delta_dim[dim] = receiver.position.at(dim) - source.position.at(dim);
        for (int dim = 0; dim < 3; ++dim)
            delta += (delta_dim[dim] * delta_dim[dim]);
        Real C_LJ = compute_LJ_scalar(delta, eps_lj, sig2);
        for (int dim = 0; dim < 3; ++dim) {
            force.at(dim) = (C_LJ * delta_dim[dim]);
        }
        return force;
    };

    CustomBehaviorFunctionWrapper fWrapper(getPositionPtrFunc, getVelocityPtrFunc, getForceFunc, boxIntersectFunc, pointAssignFunc, doLoadBalancingFunc);

    auto datatype = elements::register_datatype<N>();

    /* Experiment 2 */
    {
        mesh_data = original_data;

        PAR_START_TIMER(lb_time_spent, APP_COMM);
        Zoltan_Do_LB(&mesh_data, zlb);
        PAR_END_TIMER(lb_time_spent, APP_COMM);

        if(!rank) std::cout << "Branch and Bound: Computation is starting." << std::endl;
        auto [solution, li, dec, thist] = simulate_using_shortest_path<N>(&mesh_data, zlb, fWrapper, &params, datatype, APP_COMM);

        if(!rank && params.nb_best_path > 0)
        {
            std::ofstream ofbab;

            ofbab.open(std::to_string(params.seed)+"_branch_and_bound.txt");
            ofbab << std::fixed << std::setprecision(6) << solution.back()->cost() << std::endl;
            ofbab << li << std::endl;
            ofbab << dec << std::endl;
            ofbab << thist << std::endl;

            ofbab.close();
        }

    }

    // Do not use Zoltan_Copy(...) as it invalidates pointer, zlb must be valid throughout the entire program
    Zoltan_Copy_To(zlb, zz);

    {   /* Experiment 1 */

        mesh_data = original_data;

        IterationStatistics it_stats(nproc);

        PAR_START_TIMER(lb_time_spent, APP_COMM);
        Zoltan_Do_LB(&mesh_data, zlb);
        PAR_END_TIMER(lb_time_spent, APP_COMM);
        MPI_Allreduce(&lb_time_spent, it_stats.get_lb_time_ptr(),  1, MPI_TIME, MPI_MAX, APP_COMM);

        if(!rank) {
            std::cout << "SIM: Computation is starting." << std::endl;
            std::cout << "Average C = " << it_stats.compute_avg_lb_time() << std::endl;
        }

        PolicyRunner<ThresholdPolicy> lb_policy(&it_stats,
                [](IterationStatistics* stats){ return stats->get_cumulative_load_imbalance_slowdown(); },// get data func
                [](IterationStatistics* stats){ return stats->compute_avg_lb_time(); });                  // get threshold func

        auto [t, cum, dec, thist] = simulate<N>(&mesh_data, std::move(lb_policy), fWrapper, &params, &it_stats, datatype, APP_COMM);

        if(!rank){

            std::ofstream ofcri;

            ofcri.open(std::to_string(params.seed)+"_criterion.txt");

            ofcri << std::fixed << std::setprecision(6) << t << std::endl;
            ofcri << cum << std::endl;
            ofcri << dec << std::endl;
            ofcri << thist << std::endl;

            ofcri.close();
        }

    }

    MPI_Finalize();

    return 0;

}
