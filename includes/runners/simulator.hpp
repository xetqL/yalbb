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


template<int N, class T>
void simulate(FILE *fp,          // Output file (at 0)
                MESH_DATA<N> *mesh_data,
                Zoltan_Struct *load_balancer,
                decision_making::PolicyRunner<T> lb_policy,
                sim_param_t *params,
                const MPI_Comm comm = MPI_COMM_WORLD,
                bool automatic_migration = false) {
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

    CommunicationDatatype datatype = elements::register_datatype<N>();

    std::vector<elements::Element<N>> recv_buf(params->npart);

    if (params->record)
        gather_elements_on<N, elements::Element<N>>(nproc, rank, params->npart, mesh_data->els, 0, recv_buf,
                                           datatype.elements_datatype, comm);
    if (params->record && !rank) {
        auto particle_logger = spdlog::basic_logger_mt("particle_logger", "logs/"+std::to_string(params->seed)+"/frames/particles.csv.0");
        particle_logger->set_pattern("%v");
        std::stringstream str;
        frame_formater.write_header(str, params->npframe, params->simsize);
        write_frame_data<N>(str, recv_buf, frame_formater, params);
        particle_logger->info(str.str());
    }

    std::vector<Time> times(nproc);
    MESH_DATA<N> tmp_data;

    std::vector<Integer> lscl(mesh_data->els.size()), head;

    std::vector<Time>  my_frame_times(nframes);
    std::vector<Complexity> my_frame_cmplx(nframes);

    for (int frame = 0; frame < nframes; ++frame) {
        Time it_comm_time = 0.0, it_comp_time = 0.0, comm_time = 0.0,    comp_time = 0.0;
        Complexity it_cmplx = 0, cmplx = 0;
        START_TIMER(frame_time);
        for (int i = 0; i < npframe; ++i) {
            bool lb_decision = lb_policy.should_load_balance(i + frame * npframe);
            START_TIMER(migration_time);
            if (lb_decision) {
                zoltan_load_balance<N>(mesh_data, load_balancer, datatype, comm, automatic_migration);
            } else {
                zoltan_migrate_particles<N>(mesh_data->els, load_balancer, datatype, comm);
            }
            END_TIMER(migration_time);
            std::tie(it_cmplx, it_comp_time, it_comm_time) = lj::compute_one_step<N>(mesh_data, &lscl, &head, load_balancer, datatype, params, comm, frame);
            cmplx     += it_cmplx;
            comm_time += it_comm_time + migration_time;
            comp_time += it_comp_time;
        }
        END_TIMER(frame_time);

        time_logger->info("{:0.6f} {:0.6f} {:0.6f}", frame_time, comp_time, comm_time);
        cmplx_logger->info("{}", cmplx);

        if(frame % 5 == 0) {
            time_logger->flush();
            cmplx_logger->flush();
        }

        my_frame_times[frame] = frame_time;
        my_frame_cmplx[frame] = cmplx;

        // Write metrics to report file
        if (params->record) {
            gather_elements_on<N, elements::Element<N>>(nproc, rank, params->npart,
                                                        mesh_data->els, 0, recv_buf, datatype.elements_datatype, comm);
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
    std::vector<decltype(my_frame_times)::value_type> max_times(nframes);
    std::vector<decltype(my_frame_times)::value_type> min_times(nframes);
    decltype(my_frame_times)::value_type sum_times;
    std::vector<decltype(my_frame_cmplx)::value_type> max_cmplx(nframes);
    std::vector<decltype(my_frame_cmplx)::value_type> min_cmplx(nframes);
    decltype(my_frame_cmplx)::value_type sum_cmplx;

    std::vector<Time>  avg_times(nframes);
    std::vector<Time>  avg_cmplx(nframes);

    std::shared_ptr<spdlog::logger> lb_time_logger;
    std::shared_ptr<spdlog::logger> lb_cmplx_logger;
    if(!rank){
        lb_time_logger = spdlog::basic_logger_mt("lb_times_logger", "logs/"+std::to_string(params->seed)+"/time/frame_statistics.txt");
        lb_time_logger->set_pattern("%v");
        lb_cmplx_logger = spdlog::basic_logger_mt("lb_cmplx_logger", "logs/"+std::to_string(params->seed)+"/complexity/frame_statistics.txt");
        lb_cmplx_logger->set_pattern("%v");
    }

    for (int frame = 0; frame < nframes; ++frame) {
        MPI_Reduce(&my_frame_times[frame], &max_times[frame], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&my_frame_times[frame], &min_times[frame], 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&my_frame_times[frame], &sum_times,        1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        MPI_Reduce(&my_frame_cmplx[frame], &max_cmplx[frame], 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&my_frame_cmplx[frame], &min_cmplx[frame], 1, MPI_LONG_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&my_frame_cmplx[frame], &sum_cmplx,        1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        if(!rank) {
            avg_times[frame] = sum_times / nproc;
            avg_cmplx[frame] = sum_cmplx / nproc;
            lb_time_logger->info("{}\t{}\t{}\t{}", max_times[frame], min_times[frame], avg_times[frame], (max_times[frame]/avg_times[frame]-1.0));
            lb_cmplx_logger->info("{}\t{}\t{}\t{}", max_cmplx[frame], min_cmplx[frame], avg_cmplx[frame], (max_cmplx[frame]/avg_cmplx[frame]-1.0));
        }
    }
}

#endif //NBMPI_SIMULATE_HPP
