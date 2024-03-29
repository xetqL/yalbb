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

using Decisions = std::vector<int>;

template<int N, class T, class D, class LoadBalancer, class Wrapper>
std::vector<Time> simulate(
        LoadBalancer* LB,
        MESH_DATA<T> *mesh_data,
        D criterion,
        Boundary<N> boundary,
        Wrapper fWrapper,
        sim_param_t *params,
        Probe* probe,
        MPI_Datatype datatype,
        simulation::MonitoringSession& report_session,
        const MPI_Comm comm = MPI_COMM_WORLD,
        const std::string& simulation_name = "") {

    ProbeProcessor probeProcessor;
    io::ParallelOutput pcout(std::cout);

    auto rc = params->rc;
    auto dt = params->dt;

    auto boxIntersectFunc   = fWrapper.getBoxIntersectionFunc();
    auto doLoadBalancingFunc= fWrapper.getLoadBalancingFunc();
    auto pointAssignFunc    = fWrapper.getPointAssignationFunc();
    auto getPosPtrFunc      = fWrapper.getPosPtrFunc();
    auto getVelPtrFunc      = fWrapper.getVelPtrFunc();
    auto getForceFunc       = fWrapper.getForceFunc();
    auto unaryForceFunc     = fWrapper.getUnaryForceFunc();

    PolicyRunner lb_policy(probe, criterion);

    int nproc, rank;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);

    const auto nframes = params->nframes;
    const auto npframe = params->npframe;
    const auto niters  = nframes * npframe;

    std::vector<Time> average_it_time(niters);

    std::vector<T> recv_buf;

    report_session << "[MPI]" << std::endl;
    report_session << show(nproc) << std::endl;

    print_params(report_session, *params);

    if (params->record) {
        recv_buf.reserve(params->npart);
        auto ranks = gather_elements_on(nproc, rank, params->npart, mesh_data->els, 0, recv_buf, datatype, comm);
        report_session.report_particle<N>(recv_buf, ranks, getPosPtrFunc, 0);
    }

    std::vector<Index> lscl, head;
    std::vector<Real> flocal;

    apply_resize_strategy(&lscl,     mesh_data->els.size());
    apply_resize_strategy(&flocal, N*mesh_data->els.size());

    Time lb_time = 0.0, it_time, cum_time = 0.0, batch_time;
    std::vector<Integer> non_empty_boxes{};
    Time lb_perf_metric = 0.0;
    for (int frame = 0; frame < nframes; ++frame) {
        pcout << "Computing frame " << frame << std::endl;
        batch_time = 0.0;
        probe->start_batch(frame);
        for (int i = 0; i < npframe; ++i) {
            START_TIMER(complete_iteration_time);

            lb_time = 0.0;
            it_time = 0.0;

            bool lb_decision = lb_policy.should_load_balance();

            MPI_Bcast(&lb_decision, 1, MPI_INT, 0, comm);

            if (lb_decision)
            {
                PAR_START_TIMER(lb_time_spent, comm);
                doLoadBalancingFunc(LB, mesh_data);
                PAR_END_TIMER(lb_time_spent, comm);
                MPI_Allreduce(&lb_time_spent, &lb_time, 1, MPI_TIME, MPI_MAX, comm);
                report_session.report(simulation::LoadBalancingCost,  lb_time, " ");
                lb_perf_metric = probeProcessor.compute_load_balancing_effort(probe);
                probe->push_load_balancing_time(lb_time);
                probe->reset_cumulative_imbalance_time();
                probe->lb_interval_time = 0;
                if(params->verbosity >= 2) pcout << fmt("Load Balancing at %d; cost = %f", frame * npframe + i, lb_time) << std::endl;
            }

            probe->set_balanced(lb_decision || probe->get_current_iteration() == 0);

            PAR_START_TIMER(migrate_data_time, comm);
            migrate_data(LB, mesh_data->els, pointAssignFunc, datatype, comm);
            END_TIMER(migrate_data_time);

            const auto nlocal  = mesh_data->els.size();
            apply_resize_strategy(&lscl,   nlocal);

            auto bbox          = get_bounding_box<N>(params->rc, getPosPtrFunc, mesh_data->els);
            CLL_init<N, T>({{mesh_data->els.data(), nlocal}}, getPosPtrFunc, bbox, rc, &head, &lscl);

            int n_neighbors;

            PAR_START_TIMER(retrieve_ghosts_time, comm);
            auto remote_el     = retrieve_ghosts<N>(LB, mesh_data->els, bbox, boxIntersectFunc, params->rc,
                                                    head, lscl, datatype, comm, &n_neighbors);
            END_TIMER(retrieve_ghosts_time);

            const auto nremote = remote_el.size();

            apply_resize_strategy(&lscl,   nlocal + nremote);
            apply_resize_strategy(&flocal, N*nlocal);
            bbox = update_bounding_box<N>(bbox, params->rc, getPosPtrFunc, remote_el);

            CLL_init<N, T>({{mesh_data->els.data(), nlocal}, {remote_el.data(), nremote}}, getPosPtrFunc, bbox, rc, &head, &lscl);

            PAR_START_TIMER(compute_time, comm);
            auto nb_interactions = nbody_compute_step<N>(flocal,
                                                         mesh_data->els,
                                                         remote_el,
                                                         getPosPtrFunc,
                                                         getVelPtrFunc,
                                                         &head, &lscl,
                                                         bbox, unaryForceFunc, getForceFunc, boundary, rc, dt);
            END_TIMER(compute_time);

            auto it_compute_time = compute_time + lb_time + migrate_data_time + retrieve_ghosts_time;

            //------ end ------ //

            probe->sync_it_time_across_processors(&it_compute_time, comm);
            probe->update_cumulative_imbalance_time(probeProcessor.compute_imbalance_time(probe));
            probe->update_lb_parallel_efficiencies();

            average_it_time.at(i + frame * npframe) = probe->get_avg_it();

            MPI_Allreduce(MPI_IN_PLACE,     &nb_interactions,     1, get_mpi_type<decltype(nb_interactions)>(),  MPI_SUM, comm);

            it_time     = it_compute_time;
            probe->lb_interval_time += it_time;
            cum_time   += it_time;
            batch_time += it_time;

            report_session.report(simulation::CumulativeImbalance,    probe->get_cumulative_imbalance_time(), " ");
            report_session.report(simulation::CumulativeVanillaImbalance,    probe->get_vanilla_cumulative_imbalance_time(), " ");
            report_session.report(simulation::Imbalance,              probeProcessor.compute_load_imbalance(probe), " ");
            report_session.report(simulation::Time,                   it_time, " ");
            report_session.report(simulation::SequentialTime,         probe->sum_it, " ");
            report_session.report(simulation::CumulativeTime,         cum_time, " ");
            report_session.report(simulation::Efficiency,             probe->get_current_parallel_efficiency(), " ");

            if(lb_decision || probe->current_iteration == (params->npframe * params->nframes - 1)){
                report_session.report(simulation::LbPerf, lb_perf_metric, " ");
            }

            report_session.report(simulation::Interactions,           nb_interactions, " ");
            report_session.report(simulation::LoadBalancingIteration, static_cast<int>(lb_decision), " ");
            report_session.report(simulation::NumOfNeighbors, n_neighbors, " ");

            probe->next_iteration();
            END_TIMER(complete_iteration_time);
            MPI_Allreduce(MPI_IN_PLACE, &retrieve_ghosts_time, 1, MPI_DOUBLE, MPI_MAX, comm);
            MPI_Allreduce(MPI_IN_PLACE, &migrate_data_time, 1, MPI_DOUBLE, MPI_MAX, comm);
            MPI_Allreduce(MPI_IN_PLACE, &compute_time, 1, MPI_DOUBLE, MPI_MAX, comm);

            if(params->verbosity >= 2)
                pcout << fmt("\n%f\t|\t%f\t|\t%f\t|\t%f\t|\t%f", complete_iteration_time, compute_time, retrieve_ghosts_time, migrate_data_time, complete_iteration_time-it_time) << std::endl;
        }

        if(params->verbosity >= 1) pcout << batch_time << std::endl;

        probe->end_batch(batch_time);

        if (params->record) {
            auto ranks = gather_elements_on(nproc, rank, params->npart, mesh_data->els, 0, recv_buf, datatype, comm);
            report_session.report_particle<N>(recv_buf, ranks, getPosPtrFunc, frame+1);
        }
    }

    return average_it_time;
}

#endif //NBMPI_SIMULATE_HPP
