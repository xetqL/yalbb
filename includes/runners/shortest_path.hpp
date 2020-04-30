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
#include <zoltan.h>
#include <cstdlib>

#include "../strategy.hpp"
#include "custom/ljpotential.hpp"
#include "../physics.hpp"
#include "../output_formatter.hpp"
#include "../utils.hpp"

#include "../params.hpp"
#include "../custom/zoltan_fn.hpp"
#include "../node.hpp"
#include "../parallel_utils.hpp"

#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"

using LBSolutionPath = std::vector<std::shared_ptr<Node> >;
using LBLiHist       = std::vector<Time>;
using LBDecHist      = std::vector<int>;
using NodeQueue      = std::multiset<std::shared_ptr<Node>, Compare>;

template<int N, class T, class Wrapper>
std::tuple<LBSolutionPath, LBLiHist, LBDecHist, TimeHistory> simulate_using_shortest_path(
            MESH_DATA<T> *mesh_data,
            Zoltan_Struct* load_balancer,
            Wrapper fWrapper,
            sim_param_t *params,
            MPI_Datatype datatype,
            MPI_Comm comm = MPI_COMM_WORLD) {
    auto rc = params->rc;
    auto dt = params->dt;
    auto simsize = params->simsize;

    int nproc, rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);

    auto boxIntersectFunc   = fWrapper.getBoxIntersectionFunc();
    auto pointAssignFunc    = fWrapper.getPointAssignationFunc();
    auto getPosPtrFunc      = fWrapper.getPosPtrFunc();
    auto getVelPtrFunc      = fWrapper.getVelPtrFunc();
    auto getForceFunc       = fWrapper.getForceFunc();

    auto nb_solution_wanted = params->nb_best_path;
    const int nframes = params->nframes;
    const int npframe = params->npframe;
    const int nb_iterations = nframes*npframe;

    std::vector<T> recv_buf(params->npart);

    std::vector<Time> times(nproc), my_frame_times(nframes);
    std::vector<Index> lscl(mesh_data->els.size()), head;
    std::vector<Complexity> my_frame_cmplx(nframes);

    const int nb_data = mesh_data->els.size();
    for(int i = 0; i < nb_data; ++i) mesh_data->els[i].lid = i;

    using TNode = Node;
    std::vector<std::shared_ptr<TNode>> container;
    container.reserve((unsigned long) std::pow(2, 20));
    using PriorityQueue = std::multiset<std::shared_ptr<TNode>, Compare>;
    PriorityQueue pQueue;
    pQueue.insert(std::make_shared<Node>(load_balancer, -npframe, npframe, DoLB));

    std::vector<std::shared_ptr<Node>> solutions;
    std::vector<bool> foundYes(nframes+1, false);
    std::vector<MESH_DATA<T>> rollback_data(nframes+1);
    std::for_each(rollback_data.begin(), rollback_data.end(), [mesh_data](auto& vec){vec.els.reserve(mesh_data->els.size());});

    rollback_data[0] = *mesh_data;

    while(solutions.size() < nb_solution_wanted) {
        std::shared_ptr<Node> currentNode = *pQueue.begin();
        pQueue.erase(pQueue.begin());
        if(!rank ) std::cout << currentNode << std::endl;
        //Ok, I found a Yes Node for a given depth of the binary tree, no other Yes node at this depth can be better
        if(currentNode->decision == DoLB && currentNode->start_it > 0) {
            prune_similar_nodes(currentNode, pQueue);
            foundYes.at(currentNode->start_it / npframe) = true;
        }

        if(currentNode->end_it >= nb_iterations) {
            solutions.push_back(currentNode);
            break;
        } else {
            auto children = currentNode->get_children();
            for(std::shared_ptr<Node> node : children) {
                const auto frame      = currentNode->end_it / npframe;
                const auto next_frame = frame + 1;
                if(node && ((node->decision == DontLB) || (node->decision == DoLB && !foundYes.at(frame)))) {
                    /* compute node cost */
                    Time comp_time = 0.0;

                    Time starting_time = currentNode->cost();

                    auto mesh_data     = rollback_data.at(frame);
                    auto load_balancer = node->lb;

                    auto& cum_li_hist = node->li_slowdown_hist;
                    auto& time_hist   = node->time_hist;
                    auto& dec_hist    = node->dec_hist;
                    auto& probe    = node->stats;

                    // Move data according to my parent's state
                    migrate_data(load_balancer, mesh_data.els, pointAssignFunc, datatype, comm);
                    // Compute my bounding box as function of my local data
                    auto bbox      = get_bounding_box<N>(params->rc, getPosPtrFunc, mesh_data.els);
                    // Compute which cells are on my borders
                    auto borders   = get_border_cells_index<N>(load_balancer, bbox, params->rc, boxIntersectFunc, comm);
                    // Get the ghost data from neighboring processors
                    auto remote_el = get_ghost_data<N>(mesh_data.els, getPosPtrFunc, &head, &lscl, bbox, borders, params->rc, datatype, comm);

                    for (int i = 0; i < node->batch_size; ++i) {
                        START_TIMER(it_compute_time);
                        nbody_compute_step<N>(mesh_data.els, remote_el, getPosPtrFunc, getVelPtrFunc, &head, &lscl, bbox, getForceFunc, borders, rc, dt, simsize);
                        END_TIMER(it_compute_time);

                        // Measure load imbalance
                        MPI_Allreduce(&it_compute_time, probe.max_it_time(), 1, MPI_TIME, MPI_MAX, comm);
                        MPI_Allreduce(&it_compute_time, probe.sum_it_time(), 1, MPI_TIME, MPI_SUM, comm);
                        probe.update_cumulative_imbalance_time();
                        it_compute_time = *probe.max_it_time();

                        if(currentNode->decision == DoLB) {
                            probe.update_lb_parallel_efficiencies();
                        }

                        cum_li_hist[i] = probe.get_cumulative_imbalance_time();
                        dec_hist[i]    = node->decision == DoLB && i == 0;

                        if (node->decision == DoLB && i == 0) {
                            PAR_START_TIMER(lb_time_spent, MPI_COMM_WORLD);
                            Zoltan_Do_LB<N>(&mesh_data, load_balancer);
                            PAR_END_TIMER(lb_time_spent, MPI_COMM_WORLD);
                            MPI_Allreduce(MPI_IN_PLACE, &lb_time_spent,  1, MPI_TIME, MPI_MAX, comm);
                            probe.push_load_balancing_time(lb_time_spent);
                            probe.reset_cumulative_imbalance_time();
                            it_compute_time += lb_time_spent;
                        } else {
                            migrate_data(load_balancer, mesh_data.els, pointAssignFunc, datatype, comm);
                        }

                        time_hist[i]   = i == 0 ? starting_time + it_compute_time : time_hist[i-1] + it_compute_time;

                        bbox      = get_bounding_box<N>(params->rc, getPosPtrFunc, mesh_data.els);
                        borders   = get_border_cells_index<N>(load_balancer, bbox, params->rc, boxIntersectFunc, comm);
                        remote_el = get_ghost_data<N>(mesh_data.els, getPosPtrFunc, &head, &lscl, bbox, borders, params->rc, datatype, comm);
                        comp_time += it_compute_time;
                    }
                    node->set_cost(comp_time);
                    pQueue.insert(node);
                    if(node->end_it < nb_iterations)
                        rollback_data.at(next_frame) = mesh_data;
                }
                MPI_Barrier(comm);
            }
        }
    }

    LBSolutionPath solution_path;
    LBLiHist cumulative_load_imbalance;
    LBDecHist decisions;
    TimeHistory time_hist;
    for(auto solution : solutions ){
        Time total_time = solution->cost();
        auto it_li = cumulative_load_imbalance.begin();
        auto it_dec= decisions.begin();
        auto it_time= time_hist.begin();
        while (solution->start_it >= 0) { //reconstruct path
            solution_path.push_back(solution);
            it_li = cumulative_load_imbalance.insert(it_li, solution->li_slowdown_hist.begin(), solution->li_slowdown_hist.end());
            it_dec= decisions.insert(it_dec, solution->dec_hist.begin(), solution->dec_hist.end());
            it_time= time_hist.insert(it_time, solution->time_hist.begin(), solution->time_hist.end());
            solution = solution->parent;
        }

        std::reverse(solution_path.begin(), solution_path.end());

        spdlog::drop("particle_logger");
        spdlog::drop("lb_times_logger");
        spdlog::drop("lb_cmplx_logger");
        spdlog::drop("frame_time_logger");
        spdlog::drop("frame_cmplx_logger");
    }

    if(params->record){
        SimpleCSVFormatter frame_formater(',');
        for(int frame = 0; frame < params->nframes+1; ++frame){
            gather_elements_on(nproc, rank, params->npart, rollback_data[frame].els, 0, recv_buf, datatype, comm);

            if (!rank) {
                auto particle_logger = spdlog::basic_logger_mt("particle_logger", "logs/"+std::to_string(params->seed)+"/frames_bab/particles.csv."+std::to_string(frame));
                particle_logger->set_pattern("%v");
                std::stringstream str;
                frame_formater.write_header(str, params->npframe, params->simsize);
                write_frame_data<N>(str, recv_buf, [](auto& e){return e.position;}, frame_formater);
                particle_logger->info(str.str());
                spdlog::drop("particle_logger");
            }
        }
    }

    return {solution_path, cumulative_load_imbalance, decisions, time_hist};
}
#endif //NBMPI_SHORTEST_PATH_HPP
