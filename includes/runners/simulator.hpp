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
double simulate(FILE *fp,          // Output file (at 0)
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

    auto time_logger = spdlog::basic_logger_mt("frame_time_logger", "logs/"+std::to_string(params->seed)+"/frame_times-p"+std::to_string(rank)+".txt");
    time_logger->set_pattern("%v");

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

    std::vector<double> times(nproc);
    MESH_DATA<N> tmp_data;
    double total_time = 0.0;

    std::vector<Integer> lscl(mesh_data->els.size()), head;

    std::vector<double>  my_frame_times(nframes);

    for (int frame = 0; frame < nframes; ++frame) {
        auto frame_time = MPI_Wtime();
        for (int i = 0; i < npframe; ++i) {
            bool lb_decision = lb_policy.should_load_balance(i + frame * npframe);
            if (lb_decision) {
                zoltan_load_balance<N>(mesh_data, load_balancer, datatype, comm, automatic_migration);
            } else {
                zoltan_migrate_particles<N>(mesh_data->els, load_balancer, datatype, comm);
            }
            lennard_jones::compute_one_step<N>(mesh_data, &lscl, &head, load_balancer, datatype, params, comm, frame);
        }
        frame_time = MPI_Wtime() - frame_time;

        time_logger->info("{:0.12f}", frame_time);
        my_frame_times[frame] = frame_time;

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
    std::vector<double> max_times(nframes), avg_times(nframes);
    double sum;
    std::shared_ptr<spdlog::logger> lb_time_logger;
    if(!rank){
        lb_time_logger = spdlog::basic_logger_mt("lb_logger", "logs/"+std::to_string(params->seed)+"/lb_times.txt");
        lb_time_logger->set_pattern("%v");
    }

    for (int frame = 0; frame < nframes; ++frame){
        MPI_Reduce(&my_frame_times[frame], &max_times[frame], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&my_frame_times[frame], &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if(!rank){
            avg_times[frame] = sum / nproc;
            lb_time_logger->info("{:0.8f}\t{:0.8f}\t{:0.8f}", max_times[frame], avg_times[frame], (max_times[frame]/avg_times[frame]-1.0));
        }
    }

    return total_time;
}


#endif //NBMPI_SIMULATE_HPP
