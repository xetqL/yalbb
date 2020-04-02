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
void simulate_using_shortest_path(MESH_DATA<N> *mesh_data,
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
    SimpleCSVFormatter frame_formater(',');

    auto time_logger = spdlog::basic_logger_mt("frame_time_logger", "logs/"+std::to_string(params->seed)+"/time/frame-p"+std::to_string(rank)+".txt");
    auto cmplx_logger = spdlog::basic_logger_mt("frame_cmplx_logger", "logs/"+std::to_string(params->seed)+"/complexity/frame-p"+std::to_string(rank)+".txt");

    time_logger->set_pattern("%v");
    cmplx_logger->set_pattern("%v");

    auto datatype = elements::register_datatype<N>();

    std::vector<elements::Element<N>> recv_buf(params->npart);

    if (params->record)
        gather_elements_on<N, elements::Element<N>>(nproc, rank, params->npart, mesh_data->els, 0, recv_buf,
                                           datatype, comm);
    if (params->record && !rank) {
        auto particle_logger = spdlog::basic_logger_mt("particle_logger", "logs/"+std::to_string(params->seed)+"/frames/particles.csv.0");
        particle_logger->set_pattern("%v");
        std::stringstream str;
        frame_formater.write_header(str, params->npframe, params->simsize);
        write_frame_data<N>(str, recv_buf, frame_formater, params);
        particle_logger->info(str.str());
    }

    std::vector<Time> times(nproc), my_frame_times(nframes);
    std::vector<Index> lscl(mesh_data->els.size()), head;
    std::vector<Complexity> my_frame_cmplx(nframes);

    BoundingBox<N> bbox = get_bounding_box<N>(params->rc, mesh_data->els);
    Borders borders     = get_border_cells_index<N>(load_balancer, bbox, params->rc);
    auto remote_el      = get_ghost_data(load_balancer, mesh_data->els, &head, &lscl, bbox, borders, params->rc, datatype, comm);

    const int nb_data = mesh_data->els.size();
    for(int i = 0; i < nb_data; ++i) mesh_data->els[i].lid = i;

    using TNode = Node;
    std::vector<std::shared_ptr<TNode>> container;
    container.reserve((unsigned long) std::pow(2, 20));
    using PriorityQueue = std::multiset<std::shared_ptr<TNode>, Compare>;
    PriorityQueue pQueue;
    pQueue.insert(std::make_shared<Node>(Zoltan_Copy(load_balancer), -npframe, npframe, DoLB));

    std::vector<std::shared_ptr<Node>> solutions;
    std::vector<bool> foundYes(nframes+1, false);
    std::vector<MESH_DATA<N>> rollback_data(nframes+1);
    rollback_data[0] = *mesh_data;

    do {
        std::shared_ptr<Node> currentNode = *pQueue.begin();
        pQueue.erase(pQueue.begin());

        //Ok, I found a Yes Node for a given depth of the binary tree, no other Yes node at this depth can be better
        if(currentNode->decision == DoLB && currentNode->start_it > 0) {
            prune_similar_nodes(currentNode, pQueue);
            foundYes.at(currentNode->start_it / npframe) = true;
        }
        if(currentNode->end_it >= nb_iterations-1) {
            solutions.push_back(currentNode);
            break;
        } else {

            std::cout<< rank << " b4" << std::endl;
            std::array<std::shared_ptr<Node>, 2> children = currentNode->get_children();
            std::cout<< "after" << std::endl;
            if(currentNode->end_it == 0) children[1].reset();

            for(const std::shared_ptr<Node>& node : children) {
                /* compute node cost */
                if(node && ((node->decision == DontLB && node->start_it > 0) || (node->decision == DoLB && !foundYes.at(node->start_it / npframe)))) {
                    Time comp_time = 0.0;
                    auto mesh_data = rollback_data.at(node->start_it / npframe);
                    auto load_balancer = node->lb;
                    auto it_stats = &node->stats;

                    Zoltan_Migrate_Particles<N>(mesh_data.els, load_balancer, datatype, comm);

                    std::cout << node->start_it << " " << node->decision << std::endl;
                    bbox      = get_bounding_box<N>(params->rc, mesh_data.els);
                    borders   = get_border_cells_index<N>(load_balancer, bbox, params->rc);
                    remote_el = get_ghost_data<N>(load_balancer, mesh_data.els, &head, &lscl, bbox, borders, params->rc, datatype, comm);

                    for (int i = node->start_it; i < node->end_it; ++i) {
                        std::cout << i << std::endl;
                        START_TIMER(compute_time);
                        lj::compute_one_step<N>(mesh_data.els, remote_el, &head, &lscl, bbox, borders, params);
                        END_TIMER(compute_time);

                        // Measure load imbalance
                        //MPI_Allreduce(&compute_time, it_stats->max_it_time(), 1, MPI_TIME, MPI_MAX, MPI_COMM_WORLD);
                        //MPI_Allreduce(&compute_time, it_stats->sum_it_time(), 1, MPI_TIME, MPI_SUM, MPI_COMM_WORLD);
                        //it_stats->update_cumulative_load_imbalance_slowdown();

                        START_TIMER(migration_time);

                        if (false) {
                            PAR_START_TIMER(lb_time_spent, MPI_COMM_WORLD);
                            Zoltan_Do_LB<N>(&mesh_data, load_balancer);
                            PAR_END_TIMER(lb_time_spent, MPI_COMM_WORLD);
                            MPI_Allreduce(&lb_time_spent, it_stats->get_lb_time_ptr(), 1, MPI_TIME, MPI_MAX, MPI_COMM_WORLD);
                            it_stats->reset_load_imbalance_slowdown();
                            compute_time += lb_time_spent;
                        } else {
                            Zoltan_Migrate_Particles<N>(mesh_data.els, load_balancer, datatype, comm);
                        }

                        bbox      = get_bounding_box<N>(params->rc, mesh_data.els);
                        borders   = get_border_cells_index<N>(load_balancer, bbox, params->rc);
                        remote_el = get_ghost_data<N>(load_balancer, mesh_data.els, &head, &lscl, bbox, borders, params->rc, datatype, comm);

                        END_TIMER(migration_time);
                        comp_time += compute_time;
                    }

                    node->set_cost(comp_time);
                    pQueue.insert(node);
                    if(node->end_it < nb_iterations)
                        rollback_data.at(node->end_it / npframe) = mesh_data;
                } else {
                    std::cout << rank << " nope" << std::endl;
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
        }
    } while(solutions.size() < 1);
    std::cout << "finished" <<std::endl;
    LBSolutionPath solution_path;
    auto solution = solutions[0];
    std::list<Time> costs;
    while (solution->parent != nullptr) { //reconstruct path
        costs.push_front(solution->stats.get_cumulative_load_imbalance_slowdown());
        solution_path.push_front(solution);
        solution = solution->parent;
    }
}
#endif //NBMPI_SHORTEST_PATH_HPP
