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

#include "../decision_makers/strategy.hpp"

#include "../ljpotential.hpp"
#include "../report.hpp"
#include "../physics.hpp"
#include "../nbody_io.hpp"
#include "../utils.hpp"

#include "../params.hpp"
#include "../spatial_elements.hpp"
#include "../zoltan_fn.hpp"
#include "../astar.hpp"

#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"

using LBSolutionPath = std::list<std::shared_ptr<Node> >;
using NodeQueue      = std::multiset<std::shared_ptr<Node>, Compare>;

template<int N>
LBSolutionPath simulate_using_shortest_path(MESH_DATA<N> *mesh_data,
              Zoltan_Struct *load_balancer,
              sim_param_t *params,
              MPI_Comm comm = MPI_COMM_WORLD) {
    constexpr bool automatic_migration = true;
    int nproc, rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);

    auto nb_solution_wanted = 1;
    const int nframes = params->nframes;
    const int npframe = params->npframe;
    const int nb_iterations = nframes*npframe;

    auto datatype = elements::register_datatype<N>();

    std::vector<elements::Element<N>> recv_buf(params->npart);

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
    std::vector<MESH_DATA<N>> rollback_data(nframes+1);
    rollback_data[0] = *mesh_data;

    do {
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
                    auto mesh_data = rollback_data.at(frame);
                    auto load_balancer = node->lb;
                    IterationStatistics* it_stats = &(node->stats);

                    // Move data according to my parent's state
                    Zoltan_Migrate_Particles<N>(mesh_data.els, load_balancer, datatype, comm);
                    // Compute my bounding box as function of my local data
                    auto bbox      = get_bounding_box<N>(params->rc, mesh_data.els);
                    // Compute which cells are on my borders
                    auto borders   = get_border_cells_index<N>(load_balancer, bbox, params->rc);
                    // Get the ghost data from neighboring processors
                    auto remote_el = get_ghost_data<N>(load_balancer, mesh_data.els, &head, &lscl, bbox, borders, params->rc, datatype, comm);

                    for (int i = node->start_it; i < node->end_it; ++i) {
                        START_TIMER(it_compute_time);
                        lj::compute_one_step<N>(mesh_data.els, remote_el, &head, &lscl, bbox, borders, params);
                        END_TIMER(it_compute_time);
                        // Measure load imbalance
                        MPI_Allreduce(&it_compute_time, it_stats->max_it_time(), 1, MPI_TIME, MPI_MAX, comm);
                        MPI_Allreduce(&it_compute_time, it_stats->sum_it_time(), 1, MPI_TIME, MPI_SUM, comm);
                        it_stats->update_cumulative_load_imbalance_slowdown();
                        it_compute_time = *it_stats->max_it_time();

                        if (node->decision == DoLB) {
                            PAR_START_TIMER(lb_time_spent, MPI_COMM_WORLD);
                            Zoltan_Do_LB<N>(&mesh_data, load_balancer);
                            PAR_END_TIMER(lb_time_spent, MPI_COMM_WORLD);
                            MPI_Allreduce(MPI_IN_PLACE, &lb_time_spent,  1, MPI_TIME, MPI_MAX, comm);
                            *it_stats->get_lb_time_ptr() = lb_time_spent;
                            it_stats->reset_load_imbalance_slowdown();
                            it_compute_time += lb_time_spent;
                        } else {
                            Zoltan_Migrate_Particles<N>(mesh_data.els, load_balancer, datatype, comm);
                        }
                        bbox      = get_bounding_box<N>(params->rc, mesh_data.els);
                        borders   = get_border_cells_index<N>(load_balancer, bbox, params->rc);
                        remote_el = get_ghost_data<N>(load_balancer, mesh_data.els, &head, &lscl, bbox, borders, params->rc, datatype, comm);
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
    } while(solutions.size() < 1);

    LBSolutionPath solution_path;

    auto solution = solutions[0];
    Time total_time = solution->cost();
    std::list<Time> cumulative_load_imbalance;
    std::list<int>  decisions;

    while (solution->start_it >= 0) { //reconstruct path
        solution_path.push_front(solution);
        solution = solution->parent;
    }

    spdlog::drop("particle_logger");
    spdlog::drop("lb_times_logger");
    spdlog::drop("lb_cmplx_logger");
    spdlog::drop("frame_time_logger");
    spdlog::drop("frame_cmplx_logger");
    return solution_path;
}
#endif //NBMPI_SHORTEST_PATH_HPP
