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
#include <cstdlib>
#include <filesystem>

#include "probe.hpp"
#include "strategy.hpp"
#include "output_formatter.hpp"
#include "utils.hpp"
#include "parallel_utils.hpp"
#include "physics.hpp"
#include "params.hpp"

using ApplicationTime = Time;
using CumulativeLoadImbalanceHistory = std::vector<Time>;
using TimeHistory = std::vector<Time>;
using Decisions = std::vector<int>;

template<int N> using Position  = std::array<Real, N>;
template<int N> using Velocity  = std::array<Real, N>;

template<int N, class T, class D, class LoadBalancer, class Wrapper>
std::tuple<ApplicationTime, CumulativeLoadImbalanceHistory, Decisions, TimeHistory>
        simulate(
              LoadBalancer* LB,
              MESH_DATA<T> *mesh_data,
              LBPolicy<D> *lb_policy,
              Wrapper fWrapper,
              sim_param_t *params,
              Probe* probe,
              MPI_Datatype datatype,
              const MPI_Comm comm = MPI_COMM_WORLD,
              const std::string output_names_prefix = "") {
    auto rc = params->rc;
    auto dt = params->dt;
    auto simsize = params->simsize;
    auto boxIntersectFunc   = fWrapper.getBoxIntersectionFunc();
    auto doLoadBalancingFunc= fWrapper.getLoadBalancingFunc();
    auto pointAssignFunc    = fWrapper.getPointAssignationFunc();
    auto getPosPtrFunc      = fWrapper.getPosPtrFunc();
    auto getVelPtrFunc      = fWrapper.getVelPtrFunc();
    auto getForceFunc       = fWrapper.getForceFunc();

    doLoadBalancingFunc(LB, mesh_data);
    probe->set_balanced(true);

    int nproc, rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);

    const int nframes = params->nframes;
    const int npframe = params->npframe;

    SimpleCSVFormatter frame_formater(',');


    std::vector<T> recv_buf;
    std::ofstream fparticle;
    if (params->record) {
        recv_buf.reserve(params->npart);
        gather_elements_on(nproc, rank, params->npart, mesh_data->els, 0, recv_buf, datatype, comm);
        if (!rank) {
            std::filesystem::create_directories("logs/"+output_names_prefix+std::to_string(params->seed)+"/frames");
            fparticle.open("logs/"+output_names_prefix+std::to_string(params->seed)+"/frames/particle.csv.0");
            std::stringstream str;
            frame_formater.write_header(str, params->npframe, params->simsize);
            write_frame_data<N>(str, recv_buf, [](auto& e){return e.position;}, frame_formater);
            fparticle << str.str();
            fparticle.close();
        }
    }

    std::vector<Time> times(nproc), my_frame_times(nframes);
    std::vector<Index> lscl(mesh_data->els.size()), head;
    std::vector<Complexity> my_frame_cmplx(nframes);

    // Compute my bounding box as function of my local data
    auto bbox      = get_bounding_box<N>(params->simsize, params->rc, getPosPtrFunc, mesh_data->els);
    // Init Cell linked-list
    CLL_init<N, T>({{mesh_data->els.data(), mesh_data->els.size()}}, getPosPtrFunc, bbox, rc, &head, &lscl);
    // Compute which cells are on my borders
    auto borders   = get_border_cells_index<N>(LB,  &head, bbox, params->rc, boxIntersectFunc, comm);
    // Get the ghost data from neighboring processors
    auto remote_el = get_ghost_data<N>(mesh_data->els, getPosPtrFunc, &head, &lscl, bbox, borders, params->rc, datatype, comm);

    apply_resize_strategy(&lscl, mesh_data->els.size() + remote_el.size() );
    CLL_update<N, T>(mesh_data->els.size(), {{remote_el.data(), remote_el.size()}}, getPosPtrFunc, bbox, rc, &head, &lscl);

    ApplicationTime app_time = 0.0;
    CumulativeLoadImbalanceHistory cum_li_hist; cum_li_hist.reserve(nframes*npframe);
    TimeHistory time_hist; time_hist.reserve(nframes*npframe);
    Decisions dec; dec.reserve(nframes*npframe);
    Time total_time = 0.0;
    for (int frame = 0; frame < nframes; ++frame) {
        Time comp_time = 0.0, other=0.0;
        Complexity complexity = 0;
        for (int i = 0; i < npframe; ++i) {
            START_TIMER(it_compute_time);
            complexity += nbody_compute_step<N>(mesh_data->els, remote_el, getPosPtrFunc, getVelPtrFunc, &head, &lscl, bbox,  getForceFunc, borders, rc, dt, simsize);
            END_TIMER(it_compute_time);

            // Measure load imbalance
            MPI_Allreduce(&it_compute_time, probe->max_it_time(), 1, MPI_TIME, MPI_MAX, comm);
            MPI_Allreduce(&it_compute_time, probe->min_it_time(), 1, MPI_TIME, MPI_MIN, comm);
            MPI_Allreduce(&it_compute_time, probe->sum_it_time(), 1, MPI_TIME, MPI_SUM, comm);

            probe->update_cumulative_imbalance_time();
            it_compute_time = *probe->max_it_time();

            if(probe->is_balanced()) { probe->update_lb_parallel_efficiencies(); }

            bool lb_decision = lb_policy->should_load_balance();

            cum_li_hist.push_back(probe->get_cumulative_imbalance_time());
            dec.push_back(lb_decision);

            if (lb_decision) {
                PAR_START_TIMER(lb_time_spent, comm);
                doLoadBalancingFunc(LB, mesh_data);
                PAR_END_TIMER(lb_time_spent, comm);
                MPI_Allreduce(MPI_IN_PLACE, &lb_time_spent,  1, MPI_TIME, MPI_MAX, comm);
                probe->push_load_balancing_time(lb_time_spent);
                probe->reset_cumulative_imbalance_time();
                it_compute_time += lb_time_spent;
            } else {
                migrate_data(LB, mesh_data->els, pointAssignFunc, datatype, comm);
            }

            probe->set_balanced(lb_decision);

            total_time += it_compute_time;
            time_hist.push_back(total_time);

            START_TIMER(other_it);
            bbox      = get_bounding_box<N>(params->simsize, params->rc, getPosPtrFunc, mesh_data->els);
            apply_resize_strategy(&lscl, mesh_data->els.size());
            // Init Cell linked-list
            CLL_init<N, T>({{mesh_data->els.data(), mesh_data->els.size()}}, getPosPtrFunc, bbox, rc, &head, &lscl);
            borders   = get_border_cells_index<N>(LB, &head, bbox, params->rc, boxIntersectFunc, comm);
            remote_el = get_ghost_data<N>(mesh_data->els, getPosPtrFunc, &head, &lscl, bbox, borders, params->rc, datatype, comm);
            apply_resize_strategy(&lscl, mesh_data->els.size() + remote_el.size() );
            CLL_update<N, T>(mesh_data->els.size(), {{remote_el.data(), remote_el.size()}}, getPosPtrFunc, bbox, rc, &head, &lscl);
            END_TIMER(other_it);

            other     += other_it;
            comp_time += it_compute_time;
            probe->next_iteration();
        }

        MPI_Allreduce(MPI_IN_PLACE, &comp_time, 1, MPI_TIME, MPI_MAX, comm);
        MPI_Allreduce(MPI_IN_PLACE, &other,     1, MPI_TIME, MPI_MAX, comm);
        if(!rank) std::cout << comp_time << " and " << other << std::endl;
        app_time += comp_time;

        // Write metrics to report file
        if (params->record) {
            gather_elements_on(nproc, rank, params->npart, mesh_data->els, 0, recv_buf, datatype, comm);
            if (rank == 0) {
                fparticle.open("logs/"+output_names_prefix+std::to_string(params->seed)+"/frames/particle.csv."+ std::to_string(frame + 1));
                std::stringstream str;
                frame_formater.write_header(str, params->npframe, params->simsize);
                write_frame_data<N>(str, recv_buf, [](auto& e){return e.position;}, frame_formater);
                fparticle << str.str();
                fparticle.close();
            }
        }

        my_frame_times[frame] = comp_time;
        my_frame_cmplx[frame] = complexity;
    }

    MPI_Barrier(comm);
    std::vector<Time> max_times(nframes), min_times(nframes), avg_times(nframes);
    Time sum_times;
    std::vector<Complexity > max_cmplx(nframes), min_cmplx(nframes), avg_cmplx(nframes);
    Complexity sum_cmplx;

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
        }
    }

    return { app_time, cum_li_hist, dec, time_hist};
}

#endif //NBMPI_SIMULATE_HPP
