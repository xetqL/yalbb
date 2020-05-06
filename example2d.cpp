//
// Created by xetql on 5/6/20.
//

#include <string>
#include <mpi.h>
#include <random>

#include "example/initial_conditions.hpp"
#include "runners/simulator.hpp"
#include "probe.hpp"
#include "strategy.hpp"
#include "params.hpp"
#include "example/zoltan_fn.hpp"

int main(int argc, char** argv) {

    constexpr int N = 2;

    int rank, nproc;
    float ver;
    MESH_DATA<elements::Element<N>> mesh_data;

    std::cout << std::fixed << std::setprecision(6);

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    MPI_Comm APP_COMM;
    MPI_Comm_dup(MPI_COMM_WORLD, &APP_COMM);

    auto option = get_params(argc, argv);
    if (!option.has_value()) {
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }
    auto params = option.value();

    params.world_size = nproc;
    params.rc = 2.5f * params.sig_lj;
    params.simsize = std::ceil(params.simsize / params.rc) * params.rc;

    MPI_Bcast(&params.seed, 1, MPI_INT, 0, APP_COMM);

    if (rank == 0) {
        print_params(params);
    }

    if(Zoltan_Initialize(argc, argv, &ver) != ZOLTAN_OK) {
        MPI_Finalize(); exit(EXIT_FAILURE);
    }

    auto zz = zoltan_create_wrapper(APP_COMM);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////START PARITCLE INITIALIZATION///////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if (rank == 0) {
        std::cout << "Generating data ..." << std::endl;
        std::shared_ptr<initial_condition::lj::RejectionCondition<N>> condition;
        const int MAX_TRIAL = 1000000;
        using ElementGeneratorCfg = std::pair<std::shared_ptr<initial_condition::RandomElementsGenerator<N>>, int>;
        condition = std::make_shared<initial_condition::lj::RejectionCondition<N>>(
                        &(mesh_data.els), params.sig_lj, params.sig_lj * params.sig_lj, params.T0, 0, 0, 0,
                        params.simsize, params.simsize, params.simsize, &params);
        auto el_generator = std::make_shared<initial_condition::lj::UniformRandomElementsGenerator<N>>(params.seed, MAX_TRIAL);
        el_generator->generate_elements(mesh_data.els, params.npart, condition);
        std::cout << mesh_data.els.size() << " Done !" << std::endl;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////FINISHED PARITCLE INITIALIZATION///////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    auto zlb = Zoltan_Copy(zz);

    auto boxIntersectFunc   = [](Zoltan_Struct* zlb, double x1, double y1, double z1, double x2, double y2, double z2, int* PEs, int* num_found){
        Zoltan_LB_Box_Assign(zlb, x1, y1, z1, x2, y2, z2, PEs, num_found);
    };

    auto pointAssignFunc    = [](Zoltan_Struct* zlb, const auto* e, int* PE) {
        auto pos_in_double = get_as_double_array<N>(e->position);
        Zoltan_LB_Point_Assign(zlb, &pos_in_double.front(), PE);
    };

    auto doLoadBalancingFunc= [](Zoltan_Struct* zlb, MESH_DATA<elements::Element<N>>* mesh_data){ Zoltan_Do_LB<N>(mesh_data, zlb); };

    auto getPositionPtrFunc = [](auto* e) -> std::array<Real, N>* { return &e->position; };
    auto getVelocityPtrFunc = [](auto* e) -> std::array<Real, N>* { return &e->velocity; };

    auto getForceFunc = [eps=params.eps_lj, sig=params.sig_lj, rc=params.rc, getPositionPtrFunc](const auto* receiver, const auto* source){
        return lj_compute_force<N>(receiver, source, eps, sig*sig, rc, getPositionPtrFunc);
    };

    FunctionWrapper fWrapper(getPositionPtrFunc, getVelocityPtrFunc, getForceFunc, boxIntersectFunc, pointAssignFunc, doLoadBalancingFunc);

    auto datatype = elements::register_datatype<N>();


    /* Experiment 1 */
    double load_balancing_cost = 0;
    double load_balancing_parallel_efficiency = 0;

    if(params.nb_best_path) {
        mesh_data = original_data;
        Zoltan_Do_LB(&mesh_data, zlb);
        if(!rank) std::cout << "Branch and Bound: Computation is starting." << std::endl;
        auto [solution, li, dec, thist] = simulate_using_shortest_path<N>(&mesh_data, zlb, fWrapper, &params, datatype, APP_COMM);
        if(!rank) {
            std::ofstream ofbab;
            ofbab.open(prefix+"_branch_and_bound.txt");
            ofbab << std::fixed << std::setprecision(6) << solution.back()->cost() << std::endl;
            ofbab << li << std::endl;
            ofbab << dec << std::endl;
            ofbab << thist << std::endl;
            ofbab << solution.back()->stats.lb_cost_to_string() << std::endl;
            ofbab.close();
        }
        load_balancing_cost = solution.back()->stats.compute_avg_lb_time();
        load_balancing_parallel_efficiency = solution.back()->stats.compute_avg_lb_parallel_efficiency();
    }

    // Do not use Zoltan_Copy(...) as it invalidates pointer, zlb must be valid throughout the entire program
    Zoltan_Copy_To(zlb, zz);

    {   /* Experiment 1 */

        mesh_data = original_data;

        Probe probe(nproc);
        probe.push_load_balancing_time(load_balancing_cost);

        Zoltan_Do_LB(&mesh_data, zlb);

        if(!rank) {
            std::cout << "SIM (Menon Criterion): Computation is starting." << std::endl;
            std::cout << "Average C = " << probe.compute_avg_lb_time() << std::endl;
        }

        PolicyExecutor menon_criterion_policy(&probe,
                                              [rank, npframe = params.npframe](Probe probe) {
                                                  bool is_new_batch = (probe.get_current_iteration() % npframe == 0);
                                                  bool is_cum_imb_higher_than_C = (probe.get_cumulative_imbalance_time() >= probe.compute_avg_lb_time());
                                                  return is_new_batch && is_cum_imb_higher_than_C;
                                              });

        auto [t, cum, dec, thist] = simulate<N>(zlb, &mesh_data, std::move(menon_criterion_policy), fWrapper, &params, &probe, datatype, APP_COMM, "menon_");

        if(!rank) {
            std::ofstream ofcri;
            ofcri.open(prefix+"_criterion_menon.txt");
            ofcri << std::fixed << std::setprecision(6) << t << std::endl;
            ofcri << cum << std::endl;
            ofcri << dec << std::endl;
            ofcri << thist << std::endl;
            ofcri << probe.lb_cost_to_string() << std::endl;
            ofcri.close();
        }

    }

    MPI_Finalize();
    return 0;

}
