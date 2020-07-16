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
#include "strategy.hpp"
#include "output_formatter.hpp"
#include "utils.hpp"
#include "parallel_utils.hpp"
#include "physics.hpp"
#include "params.hpp"
#include "node.hpp"


template<int N, class T, class LoadBalancer, class LBCopyF, class LBDeleteF, class Wrapper>
std::tuple<Probe, std::vector<int>> simulate_shortest_path(
        LoadBalancer* LB,
        MESH_DATA<T> *_mesh_data,
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

    doLoadBalancingFunc(LB, _mesh_data);
    //probe->set_balanced(true);

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
    container.reserve((unsigned long) std::pow(2, 20));
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

    auto nb_cell_estimation = std::pow(simsize / rc, 3.0) / nproc;
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
                    auto &time_hist = node->time_hist;
                    auto &dec_hist = node->dec_hist;
                    auto probe = &node->stats;

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

                        auto bbox      = get_bounding_box<N>(params->rc, getPosPtrFunc, mesh_data.els);
                        auto remote_el = retrieve_ghosts<N>(LB, mesh_data.els, bbox, boxIntersectFunc, params->rc, datatype, comm);
                        const auto nlocal  = mesh_data.els.size(), nremote = remote_el.size();
                        apply_resize_strategy(&lscl,   nlocal + nremote);
                        apply_resize_strategy(&flocal, N*nlocal);
                        CLL_init<N, T>({{mesh_data.els.data(), nlocal}, {remote_el.data(), nremote}}, getPosPtrFunc, bbox, rc, &head, &lscl);

                        PAR_START_TIMER(it_compute_time, comm);
                        int nb_interactions = nbody_compute_step<N>(flocal, mesh_data.els, remote_el, getPosPtrFunc, getVelPtrFunc, &head, &lscl, bbox,  getForceFunc,  rc, dt, simsize, params->G);
                        END_TIMER(it_compute_time);

                        it_compute_time += lb_time;

                        // Measure load imbalance
                        probe->sync_it_time_across_processors(&it_compute_time, comm);
                        probe->update_cumulative_imbalance_time();
                        probe->update_lb_parallel_efficiencies();

                        MPI_Allreduce(MPI_IN_PLACE,     &nb_interactions,     1, MPI_INT,  MPI_SUM, comm);

                        cum_li_hist[i] = probe->get_cumulative_imbalance_time();
                        dec_hist[i]    = lb_decision;
                        time_hist[i]   = i == 0 ? starting_time + it_compute_time : time_hist[i-1] + it_compute_time;

                        batch_time += it_compute_time;
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
        LBDecHist decisions;
        std::vector<Time> time_hist;
        int sol_id = 0;
        std::string monitoring_files_folder = "logs/"+std::to_string(params->seed)+"/"+simulation_name+"/monitoring";

        std::filesystem::create_directories(monitoring_files_folder);

        std::ofstream fimbalance, fcumtime, ftime, fefficiency, flbit, flbcost;

        for (auto solution : solutions) {
            Time total_time = solution->cost();
            auto it_li = cumulative_load_imbalance.begin();
            auto it_dec = decisions.begin();
            auto it_time = time_hist.begin();
            if (!rank) {
                fimbalance.open(monitoring_files_folder + "/" + std::to_string(sol_id) + "_cum_imbalance.txt");
                fcumtime.open(monitoring_files_folder + "/" + std::to_string(sol_id) + "_cum_time.txt");
                ftime.open(monitoring_files_folder + "/" + std::to_string(sol_id) + "_time.txt");
                fefficiency.open(monitoring_files_folder + "/" + std::to_string(sol_id) + "_efficiency.txt");
                flbit.open(monitoring_files_folder + "/" + std::to_string(sol_id) + "_lb_it.txt");
                flbcost.open(monitoring_files_folder + "/" + std::to_string(sol_id) + "_lb_cost.txt");
            }
            /* Reconstruct data from A* search */
            while (solution->start_it >= 0) { //reconstruct path
                solution_path.push_back(solution);
                it_li = cumulative_load_imbalance.insert(it_li, solution->li_slowdown_hist.begin(),
                                                         solution->li_slowdown_hist.end());
                it_dec = decisions.insert(it_dec, solution->dec_hist.begin(), solution->dec_hist.end());
                it_time = time_hist.insert(it_time, solution->time_hist.begin(), solution->time_hist.end());
                solution = solution->parent;
            }
            std::reverse(solution_path.begin(), solution_path.end());
            if(sol_id == 0) scenario = decisions;
            if (!rank) {
                /* write */
                fimbalance << cumulative_load_imbalance << std::endl;
                ftime << time_hist << std::endl;
                flbcost << solution->stats.compute_avg_lb_time() << std::endl;
                flbit << decisions << std::endl;
                /* close files */
                fimbalance.close();
                ftime.close();
                fefficiency.close();
                flbit.close();
                flbcost.close();
            }
            sol_id++;
        }


    return {solutions[0]->stats, scenario};
}
#endif //NBMPI_SHORTEST_PATH_HPP
