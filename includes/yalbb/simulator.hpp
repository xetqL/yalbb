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
//std::tuple<ApplicationTime, CumulativeLoadImbalanceHistory, Decisions, TimeHistory>
void simulate(
        LoadBalancer* LB,
        MESH_DATA<T> *mesh_data,
        LBPolicy<D> *lb_policy,
        Wrapper fWrapper,
        sim_param_t *params,
        Probe* probe,
        MPI_Datatype datatype,
        const MPI_Comm comm = MPI_COMM_WORLD,
        const std::string simulation_name = "") {
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

    const auto nframes = params->nframes;
    const auto npframe = params->npframe;
    const auto niters  = nframes * npframe;

    std::vector<T> recv_buf;
    std::ofstream fparticle, fimbalance, fcumimbalance, ftime, fcumtime, fefficiency, flbit, flbcost;
    std::string monitoring_files_folder = "logs/"+std::to_string(params->seed)+"/"+simulation_name+"/monitoring";
    std::string frame_files_folder = "logs/"+std::to_string(params->seed)+"/"+simulation_name+"/frames";

    if(!rank) {
        std::filesystem::create_directories(monitoring_files_folder);
        fcumimbalance.open(monitoring_files_folder+"/cum_imbalance.txt");
        fimbalance.open(monitoring_files_folder+"/imbalance.txt");
        ftime.open(monitoring_files_folder+"/time.txt");
        fcumtime.open(monitoring_files_folder+"/cum_time.txt");
        fefficiency.open(monitoring_files_folder+"/efficiency.txt");
        flbit.open(monitoring_files_folder+"/lb_it.txt");
        flbcost.open(monitoring_files_folder+"/lb_cost.txt");
    }

    if (params->record) {
        recv_buf.reserve(params->npart);
        auto ranks = gather_elements_on(nproc, rank, params->npart, mesh_data->els, 0, recv_buf, datatype, comm);
        if (!rank) {
            std::filesystem::create_directories(frame_files_folder);
            fparticle.open(frame_files_folder+"/particle.csv.0");
            std::stringstream str;
            str << "x coord,y coord";
            if constexpr (N==3) str << ",z coord";
            str << ",rank" << std::endl;
            //write_frame_data<N>(str, recv_buf, [](auto& e){return e.position;}, frame_formater);
            for(int i = 0; i < params->npart; i++){
                str << recv_buf[i].position <<","<< ((Real) ranks[i]) <<std::endl;
            }
            fparticle << str.str();
            fparticle.close();
        }
    }
    auto nb_cell_estimation = std::pow(simsize / rc, 3.0)  / nproc;
    std::vector<Index> lscl, head;
    std::vector<Real> flocal;

    apply_resize_strategy(&lscl,     mesh_data->els.size());
    apply_resize_strategy(&flocal, N*mesh_data->els.size());
    apply_resize_strategy(&head,     nb_cell_estimation);

    Time it_time = 0.0, cum_time = 0.0;

    /* Vector holding data for output */
    std::vector<Time> cumulative_time(niters),
                      cumulative_imbalance(niters),
                      time_per_it(niters),
                      imbalance_per_it(niters),
                      efficiency_per_it(niters);
    std::vector<int>    lb_status_per_it(niters);
    std::vector<double> lb_costs;

    for (int frame = 0; frame < nframes; ++frame) {
        if(!rank) std::cout << "Computing frame "<< frame << std::endl;
        Time batch_time = 0.0;
        probe->start_batch(frame);
        for (int i = 0; i < npframe; ++i) {
            const auto iter = i + frame * npframe;
            it_time = 0.0;
            bool lb_decision = lb_policy->should_load_balance();
            if (lb_decision) {
                PAR_START_TIMER(lb_time_spent, comm);
                doLoadBalancingFunc(LB, mesh_data);
                PAR_END_TIMER(lb_time_spent, comm);
                MPI_Allreduce(MPI_IN_PLACE, &lb_time_spent,  1, MPI_TIME, MPI_MAX, comm);
                probe->push_load_balancing_time(lb_time_spent);
                probe->reset_cumulative_imbalance_time();
                it_time += lb_time_spent;
            }

            probe->set_balanced(lb_decision || probe->get_current_iteration() == 0);


            auto remote_el = get_ghost_data<N>(LB, mesh_data->els, getPosPtrFunc, boxIntersectFunc, params->rc, datatype, comm);
            auto bbox      = get_bounding_box<N>(params->rc, getPosPtrFunc, mesh_data->els, remote_el);
            const auto nlocal  = mesh_data->els.size(), nremote = remote_el.size();
            apply_resize_strategy(&lscl,   nlocal + nremote);
            apply_resize_strategy(&flocal, N*nlocal);
            CLL_init<N, T>({{mesh_data->els.data(), nlocal}, {remote_el.data(), nremote}}, getPosPtrFunc, bbox, rc, &head, &lscl);

            PAR_START_TIMER(it_compute_time, comm);
            nbody_compute_step<N>(flocal, mesh_data->els, remote_el, getPosPtrFunc, getVelPtrFunc, &head, &lscl, bbox,  getForceFunc,  rc, dt, simsize);
            PAR_END_TIMER(it_compute_time, comm);

            migrate_data(LB, mesh_data->els, pointAssignFunc, datatype, comm);

            // Measure load imbalance
            MPI_Allreduce(&it_compute_time, probe->max_it_time(), 1, MPI_TIME, MPI_MAX, comm);
            MPI_Allreduce(&it_compute_time, probe->min_it_time(), 1, MPI_TIME, MPI_MIN, comm);
            MPI_Allreduce(&it_compute_time, probe->sum_it_time(), 1, MPI_TIME, MPI_SUM, comm);

            probe->update_cumulative_imbalance_time();
            probe->update_lb_parallel_efficiencies();

            it_time    += *probe->max_it_time();
            cum_time   += it_time;
            batch_time += it_time;

            if(!rank) {
                cumulative_imbalance[iter]  = probe->get_cumulative_imbalance_time();
                imbalance_per_it[iter]      = probe->compute_load_imbalance();
                time_per_it[iter]           = it_time;
                cumulative_time[iter]       = cum_time;
                efficiency_per_it[iter]     = probe->get_current_parallel_efficiency();
                lb_status_per_it[iter]      = (int) lb_decision;
                if(lb_decision) flbcost << probe->compute_avg_lb_time()   << std::endl;
            }
            probe->next_iteration();
        }
        probe->end_batch(batch_time);

        if (params->record) {
            auto ranks = gather_elements_on(nproc, rank, params->npart, mesh_data->els, 0, recv_buf, datatype, comm);
            if (rank == 0) {
                fparticle.open(frame_files_folder+"/particle.csv."+ std::to_string(frame + 1));
                std::stringstream str;
                str << "x coord,y coord";
                if constexpr(N==3) str << ",z coord";
                str << ",rank" << std::endl;
                for(int i = 0; i < params->npart; i++){
                    str << recv_buf[i].position <<","<<((Real) ranks[i])<<std::endl;
                }
                fparticle << str.str();
                fparticle.close();
            }
        }
    }

    if(!rank) {
        fcumimbalance   << cumulative_imbalance << std::endl;
        fimbalance      << imbalance_per_it     << std::endl;
        ftime           << time_per_it          << std::endl;
        fcumtime        << cumulative_time      << std::endl;
        fefficiency     << efficiency_per_it    << std::endl;
        flbit           << lb_status_per_it     << std::endl;

        fimbalance.close();
        ftime.close();
        fefficiency.close();
        flbit.close();
        flbcost.close();
    }
}

#endif //NBMPI_SIMULATE_HPP
