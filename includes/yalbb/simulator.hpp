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
#include "policy.hpp"
#include "output_formatter.hpp"
#include "utils.hpp"
#include "parallel_utils.hpp"
#include "physics.hpp"
#include "params.hpp"
#include "boundary.hpp"
#include "io.hpp"

using ApplicationTime = Time;
using CumulativeLoadImbalanceHistory = std::vector<Time>;
using TimeHistory = std::vector<Time>;
using Decisions = std::vector<int>;

template<int N> using Position  = std::array<Real, N>;
template<int N> using Velocity  = std::array<Real, N>;

template<int N, class T, class D, class LoadBalancer, class Wrapper>
//std::vector<std::vector<Time>> proc_time_per_iteration
// t11,t12,t13,...,t1p: sum_x=1 to p t1x/p = mu(1)
// t21,t22,t23,...,t2p: sum_x=1 to p t2x/p = mu(2)
// ...............................................
// ti1,ti2,ti3,...,tip: sum_x=1 to p tix/p = mu(i)
std::vector<Time> simulate(
        LoadBalancer* LB,
        MESH_DATA<T> *mesh_data,
        D criterion,
        Boundary<N> boundary,
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

    std::vector<Time> average_it_time(niters);

    std::vector<T> recv_buf;
    std::string folder_prefix = "logs/"+std::to_string(params->seed)+"/"+std::to_string(params->id)+"/"+simulation_name;

    simulation::MonitoringSession report_session {!rank, params->record, folder_prefix, "", params->monitor};

    if (params->record) {
        recv_buf.reserve(params->npart);
        auto ranks = gather_elements_on(nproc, rank, params->npart, mesh_data->els, 0, recv_buf, datatype, comm);
        report_session.report_particle<N>(recv_buf, ranks, getPosPtrFunc, 0);
    }

    auto nb_cell_estimation = std::pow(simsize / rc, 3.0)  / nproc;
    std::vector<Index> lscl, head;
    std::vector<Real> flocal;

    apply_resize_strategy(&lscl,     mesh_data->els.size());
    apply_resize_strategy(&flocal, N*mesh_data->els.size());
    apply_resize_strategy(&head,     nb_cell_estimation);

    Time lb_time = 0.0, it_time, cum_time = 0.0, batch_time;

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

            auto bbox          = get_bounding_box<N>(params->rc, getPosPtrFunc, mesh_data->els);
            auto remote_el     = retrieve_ghosts<N>(LB, mesh_data->els, bbox, boxIntersectFunc, params->rc, datatype, comm);
            const auto nlocal  = mesh_data->els.size(), nremote = remote_el.size();
            apply_resize_strategy(&lscl,   nlocal + nremote);
            apply_resize_strategy(&flocal, N*nlocal);
            CLL_init<N, T>({{mesh_data->els.data(), nlocal}, {remote_el.data(), nremote}}, getPosPtrFunc, bbox, rc, &head, &lscl);

            PAR_START_TIMER(it_compute_time, comm);
            int nb_interactions = nbody_compute_step<N>(flocal, mesh_data->els, remote_el, getPosPtrFunc, getVelPtrFunc, &head, &lscl, bbox,  getForceFunc, boundary, rc, dt, simsize, params->G, params->bounce);
            END_TIMER(it_compute_time);
            auto TIME = it_compute_time;
            MPI_Allreduce(MPI_IN_PLACE, &TIME, 1, get_mpi_type<decltype(TIME)>(), MPI_SUM, comm);
            it_compute_time += lb_time;

            //------ end ------ //

            probe->sync_it_time_across_processors(&it_compute_time, comm);
            probe->update_cumulative_imbalance_time();
            probe->update_lb_parallel_efficiencies();

            average_it_time.at(i + frame * npframe) = probe->get_avg_it();

            MPI_Allreduce(MPI_IN_PLACE,     &nb_interactions,     1, MPI_INT,  MPI_SUM, comm);

            it_time     = it_compute_time;
            cum_time   += it_time;
            batch_time += it_time;

            report_session.report(simulation::CumulativeImbalance,    probe->get_cumulative_imbalance_time(), " ");
            report_session.report(simulation::Imbalance,              probe->compute_load_imbalance(), " ");
            report_session.report(simulation::Time,                   it_time, " ");
            report_session.report(simulation::CumulativeTime,         cum_time, " ");
            report_session.report(simulation::Efficiency,             probe->get_current_parallel_efficiency(), " ");
            if(lb_decision)
                report_session.report(simulation::LoadBalancingCost,  probe->compute_avg_lb_time(), " ");
            report_session.report(simulation::Interactions,           nb_interactions, " ");
            report_session.report(simulation::LoadBalancingIteration, static_cast<int>(lb_decision), " ");

            probe->next_iteration();
        }
        if(!rank) std::cout << batch_time << std::endl;
        probe->end_batch(batch_time);

        if (params->record) {
            auto ranks = gather_elements_on(nproc, rank, params->npart, mesh_data->els, 0, recv_buf, datatype, comm);
            report_session.report_particle<N>(recv_buf, ranks, getPosPtrFunc, frame+1);
        }
    }

    return average_it_time;
}

#endif //NBMPI_SIMULATE_HPP
