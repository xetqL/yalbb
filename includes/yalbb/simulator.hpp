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
        D criterion,
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

    PolicyRunner lb_policy(probe, criterion);

    int nproc, rank;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);

    const auto nframes = params->nframes;
    const auto npframe = params->npframe;
    const auto niters  = nframes * npframe;

    std::vector<T> recv_buf;
    std::ofstream fparticle, fimbalance, fcumimbalance, ftime, fcumtime, fefficiency, flbit, flbcost, finteractions;
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
        finteractions.open(monitoring_files_folder+"/interactions.txt");
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

    Time lb_time = 0.0, it_time = 0.0, cum_time = 0.0, batch_time = 0.0;

    /* Vector holding data for output */
    std::vector<Time> interactions(niters),
                      cumulative_time(niters),
                      cumulative_imbalance(niters),
                      time_per_it(niters),
                      imbalance_per_it(niters),
                      efficiency_per_it(niters);

    std::vector<int>    lb_status_per_it(niters);
    std::vector<double> lb_costs;

    for (int frame = 0; frame < nframes; ++frame) {
        if(!rank) std::cout << "Computing frame " << frame << " ";
        batch_time = 0.0;
        probe->start_batch(frame);
        for (int i = 0; i < npframe; ++i) {
            lb_time = 0.0;
            it_time = 0.0;
            bool lb_decision = lb_policy.should_load_balance();

            MPI_Bcast(&lb_decision, 1, MPI_INT, 0, comm);

            if (lb_decision) {
                PAR_START_TIMER(lb_time_spent, comm);
                doLoadBalancingFunc(LB, mesh_data);
                PAR_END_TIMER(lb_time_spent, comm);
                MPI_Allreduce(&lb_time_spent, &lb_time, 1, MPI_TIME, MPI_MAX, comm);
                probe->push_load_balancing_time(lb_time_spent);
                probe->reset_cumulative_imbalance_time();
            }

            probe->set_balanced(lb_decision || probe->get_current_iteration() == 0);

            migrate_data(LB, mesh_data->els, pointAssignFunc, datatype, comm);

            auto bbox      = get_bounding_box<N>(params->rc, getPosPtrFunc, mesh_data->els);
            auto remote_el = retrieve_ghosts<N>(LB, mesh_data->els, bbox, boxIntersectFunc, params->rc, datatype, comm);
            const auto nlocal  = mesh_data->els.size(), nremote = remote_el.size();
            apply_resize_strategy(&lscl,   nlocal + nremote);
            apply_resize_strategy(&flocal, N*nlocal);
            CLL_init<N, T>({{mesh_data->els.data(), nlocal}, {remote_el.data(), nremote}}, getPosPtrFunc, bbox, rc, &head, &lscl);

            PAR_START_TIMER(it_compute_time, comm);
            int nb_interactions = nbody_compute_step<N>(flocal, mesh_data->els, remote_el, getPosPtrFunc, getVelPtrFunc, &head, &lscl, bbox,  getForceFunc,  rc, dt, simsize, params->G, params->bounce);
            END_TIMER(it_compute_time);

            it_compute_time += lb_time;

            //------ end ------ //

            probe->sync_it_time_across_processors(&it_compute_time, comm);

            probe->update_cumulative_imbalance_time();
            probe->update_lb_parallel_efficiencies();

            MPI_Allreduce(MPI_IN_PLACE,     &nb_interactions,     1, MPI_INT,  MPI_SUM, comm);

            it_time     = it_compute_time;
            cum_time   += it_time;
            batch_time += it_time;

            if(!rank) {
                fcumimbalance << probe->get_cumulative_imbalance_time() << " ";
                fimbalance    << probe->compute_load_imbalance() << " ";
                ftime         << it_time << " ";
                fcumtime      << cum_time << " ";
                fefficiency   << probe->get_current_parallel_efficiency() << " ";
                if(lb_decision) flbcost << probe->compute_avg_lb_time() << " ";
                finteractions << nb_interactions << " ";
            }

            flbit << ((int) lb_decision) << " ";
            probe->next_iteration();
        }
        if(!rank) std::cout << batch_time << std::endl;
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
        fimbalance.close();
        ftime.close();
        fefficiency.close();
        flbit.close();
        flbcost.close();
    }
}

#endif //NBMPI_SIMULATE_HPP
