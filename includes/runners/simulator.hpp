//
// Created by xetql on 27.06.18.
//

#ifndef NBMPI_SIMULATE_HPP
#define NBMPI_SIMULATE_HPP

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

#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"

template<int N>
std::vector<elements::Element<N>> get_ghost_data(Zoltan_Struct* load_balancer,
                    std::vector<elements::Element<N>>& elements,
                    std::vector<Integer>* head, std::vector<Integer>* lscl,
                    BoundingBox<N>& bbox, Borders borders, Real rc,
                    MPI_Datatype datatype, MPI_Comm comm){
    int r,s;
    const size_t nb_elements = elements.size();
    if(const auto n_cells = get_total_cell_number<N>(bbox, rc); head->size() < n_cells){ head->resize(n_cells); }
    if(nb_elements > lscl->size()) { lscl->resize(nb_elements); }
    algorithm::CLL_init<N>({{elements.data(), nb_elements}}, bbox, rc, head->data(), lscl->data());
    return zoltan_exchange_data<N>(elements, load_balancer, head->data(), lscl->data(), borders, datatype, comm, r, s);
}

template<int N, class T>
void simulate(MESH_DATA<N> *mesh_data,
              Zoltan_Struct *load_balancer,
              decision_making::PolicyRunner<T> lb_policy,
              sim_param_t *params,
              decision_making::IterationStatistics* it_stats = nullptr,
              const MPI_Comm comm = MPI_COMM_WORLD) {
    constexpr bool automatic_migration = true;
    int nproc, rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);

    const int nframes = params->nframes;
    const int npframe = params->npframe;

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
    for (int frame = 0; frame < nframes; ++frame) {
        Time comm_time = 0.0, comp_time = 0.0;
        Complexity cmplx = 0;
        START_TIMER(frame_time);
        for (int i = 0; i < npframe; ++i) {
            START_TIMER(compute_time)
            cmplx += lj::compute_one_step<N>(mesh_data->els, remote_el, &head, &lscl, bbox, borders, params, frame);
            END_TIMER(compute_time)

            START_TIMER(migration_time);
            bool lb_decision = lb_policy.should_load_balance(i + frame * npframe);
            if (lb_decision) {
                PAR_START_TIMER(lb_time_spent, MPI_COMM_WORLD);
                Zoltan_LB<N>(mesh_data, load_balancer);
                PAR_END_TIMER(lb_time_spent, MPI_COMM_WORLD);
                MPI_Allreduce(&lb_time_spent, it_stats->get_lb_time_ptr(), 1, MPI_TIME, MPI_MAX, MPI_COMM_WORLD);
                it_stats->reset_load_imbalance_slowdown();
            } else {
                Zoltan_Migrate_Particles<N>(mesh_data->els, load_balancer, datatype, comm);
            }
            bbox      = get_bounding_box<N>(params->rc, mesh_data->els);
            borders   = get_border_cells_index<N>(load_balancer, bbox, params->rc);
            remote_el = get_ghost_data<N>(load_balancer, mesh_data->els, &head, &lscl, bbox, borders, params->rc, datatype, comm);
            END_TIMER(migration_time);

            comp_time += compute_time;
            comm_time += migration_time;
        }
        END_TIMER(frame_time);
        MPI_Allreduce(&comp_time, it_stats->max_it_time(), 1, MPI_TIME, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&comp_time, it_stats->sum_it_time(), 1, MPI_TIME, MPI_SUM, MPI_COMM_WORLD);
        it_stats->update_cumulative_load_imbalance_slowdown();

        time_logger->info("{:0.6f} {:0.6f} {:0.6f}", frame_time, comp_time, comm_time);
        cmplx_logger->info("{}", cmplx);

        if(frame % 5 == 0) {
            time_logger->flush();
            cmplx_logger->flush();
        }

        my_frame_times[frame] = comp_time;
        my_frame_cmplx[frame] = cmplx;

        // Write metrics to report file
        if (params->record) {
            gather_elements_on<N, elements::Element<N>>(nproc, rank, params->npart,
                                                        mesh_data->els, 0, recv_buf, datatype, comm);
            if (rank == 0) {
                spdlog::drop("particle_logger");
                auto particle_logger = spdlog::basic_logger_mt("particle_logger", "logs/"+std::to_string(params->seed)+"/frames/particles.csv."+ std::to_string(frame + 1));
                particle_logger->set_pattern("%v");
                std::stringstream str;
                frame_formater.write_header(str, params->npframe, params->simsize);
                write_frame_data<N>(str, recv_buf, frame_formater, params);
                particle_logger->info(str.str());
            }
        }
    }

    MPI_Barrier(comm);
    std::vector<Time> max_times(nframes), min_times(nframes), avg_times(nframes);
    Time sum_times;
    std::vector<Complexity > max_cmplx(nframes), min_cmplx(nframes), avg_cmplx(nframes);
    Complexity sum_cmplx;

    std::shared_ptr<spdlog::logger> lb_time_logger;
    std::shared_ptr<spdlog::logger> lb_cmplx_logger;
    if(!rank){
        lb_time_logger = spdlog::basic_logger_mt("lb_times_logger", "logs/"+std::to_string(params->seed)+"/time/frame_statistics.txt");
        lb_time_logger->set_pattern("%v");
        lb_cmplx_logger = spdlog::basic_logger_mt("lb_cmplx_logger", "logs/"+std::to_string(params->seed)+"/complexity/frame_statistics.txt");
        lb_cmplx_logger->set_pattern("%v");
    }

    for (int frame = 0; frame < nframes; ++frame) {
        MPI_Reduce(&my_frame_times[frame], &max_times[frame], 1, MPI_TIME, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&my_frame_times[frame], &min_times[frame], 1, MPI_TIME, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&my_frame_times[frame], &sum_times,        1, MPI_TIME, MPI_SUM, 0, MPI_COMM_WORLD);

        MPI_Reduce(&my_frame_cmplx[frame], &max_cmplx[frame], 1, MPI_COMPLEXITY, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&my_frame_cmplx[frame], &min_cmplx[frame], 1, MPI_COMPLEXITY, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&my_frame_cmplx[frame], &sum_cmplx,        1, MPI_COMPLEXITY, MPI_SUM, 0, MPI_COMM_WORLD);

        if(!rank) {
            avg_times[frame] = sum_times / nproc;
            avg_cmplx[frame] = sum_cmplx / nproc;
            lb_time_logger->info("{}\t{}\t{}\t{}", max_times[frame], min_times[frame], avg_times[frame], (max_times[frame]/avg_times[frame]-1.0));
            lb_cmplx_logger->info("{}\t{}\t{}\t{}", max_cmplx[frame], min_cmplx[frame], avg_cmplx[frame], (max_cmplx[frame]/avg_cmplx[frame]-1.0));
        }
    }
}

#endif //NBMPI_SIMULATE_HPP
