//
// Created by xetql on 3/30/20.
//

#ifndef NBMPI_SHORTEST_PATH_HPP
#define NBMPI_SHORTEST_PATH_HPP

#include <sstream>
#include <fstream>
#include <iomanip>
#include <mpi.h>
#include <map>
#include <unordered_map>
#include <cstdlib>
#include <filesystem>

#include "probe.hpp"
#include "policy.hpp"
#include "output_formatter.hpp"
#include "utils.hpp"
#include "parallel_utils.hpp"
#include "physics.hpp"
#include "params.hpp"
#include "node.hpp"
#include "io.hpp"
#include "simulator.hpp"


template<int N, class T, class LoadBalancer, class LBCopyF, class LBDeleteF, class Wrapper>
std::tuple<Probe, std::vector<int>> simulate_shortest_path(
        LoadBalancer* LB,
        MESH_DATA<T> *_mesh_data,
        Boundary<N> boundary,
        Wrapper fWrapper,
        sim_param_t *params,
        MPI_Datatype datatype,
        LBCopyF&& lb_copy_f,
        LBDeleteF&& lb_delete_f,
        const MPI_Comm comm = MPI_COMM_WORLD,
        const std::string simulation_name = "") {

    using Node = Node<LoadBalancer, LBCopyF, LBDeleteF>;
    using LBSolutionPath = std::vector<std::shared_ptr<Node> >;
    using LBLiHist       = std::vector<Time>;
    using LBDecHist      = std::vector<int>;
    using NodeQueue      = std::multiset<std::shared_ptr<Node>, Compare>;

    auto rc = params->rc;
    auto dt = params->dt;
    auto simsize = params->simsize;

    int nproc, rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);

    auto boxIntersectFunc   = fWrapper.getBoxIntersectionFunc();
    auto doLoadBalancingFunc= fWrapper.getLoadBalancingFunc();
    auto pointAssignFunc    = fWrapper.getPointAssignationFunc();
    auto getPosPtrFunc      = fWrapper.getPosPtrFunc();
    auto getVelPtrFunc      = fWrapper.getVelPtrFunc();
    auto getForceFunc       = fWrapper.getForceFunc();
    auto unaryForceFunc       = fWrapper.getUnaryForceFunc();
    std::vector<Time> average_time;

    { // find \mu(i)
        MPI_Comm NEW_COMM;
        MPI_Comm_dup(comm, &NEW_COMM);
        auto data = *_mesh_data;
        LoadBalancer* new_LB = lb_copy_f(LB);
        auto new_params = *params;

        new_params.monitor = true;
        new_params.verbosity = 3;
        new_params.id += 10000;
        Probe probe(nproc);
        average_time = simulate<N>(new_LB, &data, lb::ImprovedMenon{}, boundary, fWrapper, &new_params, &probe, datatype, NEW_COMM, fmt("%s_mu_finder/", simulation_name));
        for(auto it = std::begin(average_time); it != std::end(average_time); it++){
            *it = std::accumulate(it, std::end(average_time), 0.0);
        }
        lb_delete_f(new_LB);
        Node::optimistic_remaining_time = average_time;
    }

    doLoadBalancingFunc(LB, _mesh_data);

    auto nb_solution_wanted = 1;
    const int nframes = params->nframes;
    const int npframe = params->npframe;
    const int nb_iterations = nframes * npframe;

    std::vector<T> recv_buf(params->npart);

    std::vector<Time> times(nproc), my_frame_times(nframes);
    std::vector<Complexity> my_frame_cmplx(nframes);

    const int nb_data = _mesh_data->els.size();
    for (int i = 0; i < nb_data; ++i) _mesh_data->els[i].lid = i;

    std::vector<std::shared_ptr<Node>> container;
    container.reserve((unsigned long) params->nframes * params->nframes);
    using PriorityQueue = std::multiset<std::shared_ptr<Node>, Compare>;
    PriorityQueue pQueue;

    pQueue.insert(std::make_shared<Node>(LB, -npframe, npframe, DoLB, lb_copy_f, lb_delete_f));

    std::vector<std::shared_ptr<Node>> solutions;
    std::vector<bool> foundYes(nframes + 1, false);
    std::vector<MESH_DATA<T>> rollback_data(nframes + 1);
    std::for_each(rollback_data.begin(), rollback_data.end(), [size = _mesh_data->els.size()](auto &vec) { vec.els.reserve(size); });

    rollback_data[0] = *_mesh_data;

    std::vector<Index> lscl, head;
    std::vector<Real> flocal;

    auto nb_cell_estimation = std::pow(simsize / rc, static_cast<Real>(N)) / nproc;
    apply_resize_strategy(&lscl, _mesh_data->els.size());
    apply_resize_strategy(&flocal, N * _mesh_data->els.size());
    apply_resize_strategy(&head, nb_cell_estimation);

    while (solutions.size() < 1) {
        std::shared_ptr<Node> currentNode = *pQueue.begin();
        pQueue.erase(pQueue.begin());
        if (!rank) std::cout << currentNode << " " << currentNode->cost() << std::endl;
        //Ok, I found a Yes Node for a given depth of the binary tree, no other Yes node at this depth can be better
        if (currentNode->decision == DoLB && currentNode->start_it > 0) {
            prune_similar_nodes(currentNode, pQueue);
            foundYes.at(currentNode->start_it / npframe) = true;
        }

        if (currentNode->end_it >= nb_iterations) {
            solutions.push_back(currentNode);
            break;
        } else {
            auto children = currentNode->get_children();
            for (std::shared_ptr<Node> node : children) {
                const auto frame = currentNode->end_it / npframe;
                const auto next_frame = frame + 1;
                if (node && ((node->decision == DontLB) || (node->decision == DoLB && !foundYes.at(frame)))) {
                    /* compute node cost */
                    Time batch_time = 0.0;
                    Time starting_time = currentNode->cost();
                    auto mesh_data = rollback_data.at(frame);
                    auto LB = node->lb;

                    auto &cum_li_hist = node->li_slowdown_hist;
                    auto &interactions_hist = node->interactions_hist;
                    auto &vanilla_cum_li_hist = node->van_li_slowdown_hist;
                    auto &time_hist   = node->time_hist;
                    auto &dec_hist    = node->dec_hist;
                    auto &eff_hist    = node->efficiency_hist;

                    auto probe        = &node->stats;

                    // Move data according to my parent's state
                    migrate_data(LB, mesh_data.els, pointAssignFunc, datatype, comm);

                    for (int i = 0; i < node->batch_size; ++i) {
                        Time it_time = 0.0;
                        Time lb_time = 0.0;
                        bool lb_decision = node->get_decision() == DoLB && i == 0;

                        if (lb_decision) {
                            PAR_START_TIMER(lb_time_spent, comm);
                            doLoadBalancingFunc(LB, &mesh_data);
                            PAR_END_TIMER(lb_time_spent, comm);
                            MPI_Allreduce(&lb_time_spent, &lb_time, 1, MPI_TIME, MPI_MAX, comm);
                            probe->push_load_balancing_time(lb_time_spent);
                            probe->reset_cumulative_imbalance_time();
                        }

                        probe->set_balanced(lb_decision || probe->get_current_iteration() == 0);

                        migrate_data(LB, mesh_data.els, pointAssignFunc, datatype, comm);

                        const auto nlocal  = mesh_data.els.size();
                        apply_resize_strategy(&lscl,   nlocal);

                        auto bbox          = get_bounding_box<N>(params->rc, getPosPtrFunc, mesh_data.els);
                        CLL_init<N, T>({{mesh_data.els.data(), nlocal}}, getPosPtrFunc, bbox, rc, &head, &lscl);

                        int n_neighbors;

                        auto remote_el     = retrieve_ghosts<N>(LB, mesh_data.els, bbox, boxIntersectFunc, params->rc,
                                                                head, lscl, datatype, comm, &n_neighbors);

                        const auto nremote = remote_el.size();

                        apply_resize_strategy(&lscl,   nlocal + nremote);
                        apply_resize_strategy(&flocal, N*nlocal);
                        bbox = update_bounding_box<N>(bbox, params->rc, getPosPtrFunc, remote_el);

                        CLL_init<N, T>({{mesh_data.els.data(), nlocal}, {remote_el.data(), nremote}}, getPosPtrFunc, bbox, rc, &head, &lscl);

                        PAR_START_TIMER(it_compute_time, comm);
                        auto nb_interactions = nbody_compute_step<N>(flocal,
                                                                     mesh_data.els,
                                                                     remote_el,
                                                                     getPosPtrFunc,
                                                                     getVelPtrFunc,
                                                                     &head, &lscl,
                                                                     bbox, unaryForceFunc, getForceFunc, boundary, rc, dt);
                        END_TIMER(it_compute_time);

                        it_compute_time += lb_time;

                        // Measure load imbalance
                        probe->sync_it_time_across_processors(&it_compute_time, comm);
                        probe->update_cumulative_imbalance_time();
                        probe->update_lb_parallel_efficiencies();

                        MPI_Allreduce(MPI_IN_PLACE,     &nb_interactions,     1, MPI_INT,  MPI_SUM, comm);

                        cum_li_hist[i] = probe->get_cumulative_imbalance_time();
                        vanilla_cum_li_hist[i] = probe->get_vanilla_cumulative_imbalance_time();
                        dec_hist[i]    = lb_decision;
                        time_hist[i]   = i == 0 ? starting_time + it_compute_time : time_hist[i-1] + it_compute_time;
                        eff_hist[i]    = probe->get_efficiency();
                        interactions_hist[i]    = nb_interactions;
                        batch_time    += it_compute_time;

                        probe->next_iteration();
                    }

                    node->set_cost(batch_time);

                    pQueue.insert(node);
                    if (node->end_it < nb_iterations)
                        rollback_data.at(next_frame) = mesh_data;
                }
                MPI_Barrier(comm);
            }
        }
    }
    std::vector<int> scenario;

    LBSolutionPath solution_path;
    LBLiHist cumulative_load_imbalance;
    LBLiHist van_cumulative_load_imbalance;
    LBDecHist decisions;
    std::vector<Time> time_hist;
    std::vector<Time>  eff_hist;
    std::vector<unsigned>  inter_hist;
    int sol_id = 0;

    std::string folder_prefix = fmt("%s/%s", "logs", simulation_name);

    for (auto solution : solutions) {
        Time total_time = solution->cost();
        simulation::MonitoringSession report_session{!rank, false, folder_prefix, std::to_string(sol_id)+"_"};
        auto it_li = cumulative_load_imbalance.begin();
        auto it_van_li  = van_cumulative_load_imbalance.begin();
        auto it_dec = decisions.begin();
        auto it_time = time_hist.begin();
        auto it_eff  = eff_hist.begin();
        auto it_inter  = inter_hist.begin();
        /* Reconstruct data from A* search */
        while (solution->start_it >= 0) { //reconstruct path
            solution_path.push_back(solution);
            it_li   = cumulative_load_imbalance.insert(it_li, solution->li_slowdown_hist.begin(),
                                                              solution->li_slowdown_hist.end());
            it_van_li = van_cumulative_load_imbalance.insert(it_van_li, solution->van_li_slowdown_hist.begin(),
                                                                      solution->van_li_slowdown_hist.end());
            it_dec  = decisions.insert(it_dec,  solution->dec_hist.begin(),  solution->dec_hist.end());
            it_time = time_hist.insert(it_time, solution->time_hist.begin(), solution->time_hist.end());
            it_eff  =  eff_hist.insert(it_eff,  solution->efficiency_hist.begin(),  solution->efficiency_hist.end());
            it_inter=  inter_hist.insert(it_inter,  solution->interactions_hist.begin(),  solution->interactions_hist.end());
            solution = solution->parent;
        }

        std::reverse(solution_path.begin(), solution_path.end());
        if(sol_id == 0) scenario = decisions;

        report_session.report(simulation::CumulativeImbalance,           cumulative_load_imbalance);
        report_session.report(simulation::CumulativeTime,                time_hist);
        report_session.report(simulation::Efficiency,                    eff_hist);
        report_session.report(simulation::LoadBalancingIteration,        decisions);
        report_session.report(simulation::Interactions,                  inter_hist);

        sol_id++;
    }
    return {solutions[0]->stats, scenario};
}
#endif //NBMPI_SHORTEST_PATH_HPP
