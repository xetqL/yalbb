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
    std::ofstream fparticle, fimbalance, ftime, fefficiency, flbit, flbcost;
    std::string monitoring_files_folder = "logs/"+output_names_prefix+std::to_string(params->seed)+"/monitoring";
    if(!rank) {
        std::filesystem::create_directories(monitoring_files_folder);
        fimbalance.open(monitoring_files_folder+"/imbalance.txt");
        ftime.open(monitoring_files_folder+"/time.txt");
        fefficiency.open(monitoring_files_folder+"/efficiency.txt");
        flbit.open(monitoring_files_folder+"/lb_it.txt");
        flbcost.open(monitoring_files_folder+"/lb_cost.txt");
    }

    if (params->record) {
        recv_buf.reserve(params->npart);
        auto ranks = gather_elements_on(nproc, rank, params->npart, mesh_data->els, 0, recv_buf, datatype, comm);
        if (!rank) {
            std::filesystem::create_directories("logs/"+output_names_prefix+std::to_string(params->seed)+"/frames");
            fparticle.open("logs/"+output_names_prefix+std::to_string(params->seed)+"/frames/particle.csv.0");
            std::stringstream str;
            str << "x coord,y coord";
            if constexpr(N==3) str << ",z coord";
            str << ",rank" << std::endl;
            //write_frame_data<N>(str, recv_buf, [](auto& e){return e.position;}, frame_formater);
            for(int i = 0; i < params->npart; i++){
                str << recv_buf[i].position <<","<< ((Real) ranks[i]) <<std::endl;
            }
            fparticle << str.str();
            fparticle.close();
        }
    }
    Real nb_cell_estimation = ((simsize / rc) *  (simsize / rc) *  (simsize / rc)) / nproc;
    std::vector<Index> lscl(mesh_data->els.size()), head(nb_cell_estimation);
    Time it_time = 0.0;
    MPI_Barrier(comm);
    for (int frame = 0; frame < nframes; ++frame) {
        if(!rank) std::cout << "Computing frame "<< frame << std::endl;
        for (int i = 0; i < npframe; ++i) {
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

            START_TIMER(it_compute_time);
            auto bbox      = get_bounding_box<N>(params->simsize, params->rc, getPosPtrFunc, mesh_data->els);
            CLL_init<N, T>({{mesh_data->els.data(), mesh_data->els.size()}}, getPosPtrFunc, bbox, rc, &head, &lscl);
            auto remote_el = get_ghost_data<N>(LB, mesh_data->els, getPosPtrFunc, boxIntersectFunc, params->rc, datatype, comm);
            apply_resize_strategy(&lscl, mesh_data->els.size() + remote_el.size() );
            CLL_update<N, T>(mesh_data->els.size(), {{remote_el.data(), remote_el.size()}}, getPosPtrFunc, bbox, rc, &head, &lscl);
            nbody_compute_step<N>(mesh_data->els, remote_el, getPosPtrFunc, getVelPtrFunc, &head, &lscl, bbox,  getForceFunc,  rc, dt, simsize);
            END_TIMER(it_compute_time);

            migrate_data(LB, mesh_data->els, pointAssignFunc, datatype, comm);

            // Measure load imbalance
            MPI_Allreduce(&it_compute_time, probe->max_it_time(), 1, MPI_TIME, MPI_MAX, comm);
            MPI_Allreduce(&it_compute_time, probe->min_it_time(), 1, MPI_TIME, MPI_MIN, comm);
            MPI_Allreduce(&it_compute_time, probe->sum_it_time(), 1, MPI_TIME, MPI_SUM, comm);

            probe->update_cumulative_imbalance_time();
            it_time += *probe->max_it_time();
            probe->update_lb_parallel_efficiencies();

            if(!rank) {
                fimbalance << probe->compute_load_imbalance() << " " << std::endl;
                ftime << it_time << " " << std::endl;
                fefficiency << probe->get_current_parallel_efficiency() << " " << std::endl;
                if(lb_decision) flbit << ((i+1)+frame*npframe) << " " << std::endl;
                if(lb_decision) flbcost << probe->compute_avg_lb_time() << " " << std::endl;
            }

            probe->set_balanced(lb_decision);
            probe->next_iteration();
        }
        if (params->record) {
            auto ranks = gather_elements_on(nproc, rank, params->npart, mesh_data->els, 0, recv_buf, datatype, comm);
            if (rank == 0) {
                fparticle.open("logs/"+output_names_prefix+std::to_string(params->seed)+"/frames/particle.csv."+ std::to_string(frame + 1));
                std::stringstream str;
                //frame_formater.write_header(str, params->npframe, params->simsize);
                str << "x coord,y coord";
                if constexpr(N==3) str << ",z coord";
                str << ",rank" << std::endl;
                //write_frame_data<N>(str, recv_buf, [rank](auto& e){return e.position << "," << rank;}, frame_formater);
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
