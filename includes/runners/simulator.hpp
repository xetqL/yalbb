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

template<int N>
double simulate(FILE *fp,          // Output file (at 0)
            MESH_DATA<N> *mesh_data,
            Zoltan_Struct *load_balancer,
            std::shared_ptr<decision_making::Policy> lb_policy,
            sim_param_t *params,
            const MPI_Comm comm = MPI_COMM_WORLD,
            bool automatic_migration = false) {
    int nproc, rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);
    std::ofstream lb_file, metric_file, frame_file;

    const int nframes = params->nframes;
    const int npframe = params->npframe;

    SimpleCSVFormatter frame_formater(',');

    CommunicationDatatype datatype = elements::register_datatype<N>();

    std::vector<elements::Element<N>> recv_buf(params->npart);
    auto date = get_date_as_string();

    if (params->record)
        gather_elements_on<N, elements::Element<N>>(nproc, rank, params->npart, mesh_data->els, 0, recv_buf,
                                           datatype.elements_datatype, comm);
    if (params->record && !rank) {
        std::string mkdir_cmd = "mkdir -p data/time-series/"+std::to_string(params->seed);
        int err = system(mkdir_cmd.c_str());
        frame_file.open("data/time-series/"+std::to_string(params->seed)+"/run_cpp.csv.0", std::ofstream::out | std::ofstream::trunc);
        frame_formater.write_header(frame_file, params->npframe, params->simsize);
        write_frame_data<N>(frame_file, recv_buf, frame_formater, params);
        frame_file.close();
    }

    int nb_lb = 0;
    //std::vector<elements::Element<N>> remote_el;
    std::vector<double> times(nproc);
    MESH_DATA<N> tmp_data;
    double total_time = 0.0;
    std::vector<double> max, avg;
    Real cut_off_radius = params->rc; // cut_off
    auto cell_per_row = (Integer) std::ceil(params->simsize / cut_off_radius); // number of cell in a row
    auto n_cells = cell_per_row * cell_per_row * cell_per_row;
    Integer lc[N];
    lc[0] = cell_per_row;
    lc[1] = cell_per_row;
    if constexpr (N==3) {
        lc[2] = cell_per_row;
    }

    std::vector<Integer> lscl(mesh_data->els.size()), head(n_cells);
    int prev_size = mesh_data->els.size();
    for (int frame = 0; frame < nframes; ++frame) {
        double frame_time = 0.0;
        double wtime = MPI_Wtime();
        for (int i = 0; i < npframe; ++i) {
            //bool should_load_balance_now = lb_policy->should_load_balance(i + frame * npframe, nullptr);
            if (false) {
                zoltan_load_balance<N>(mesh_data, load_balancer, datatype, comm, automatic_migration);
                nb_lb ++;
            } else {

                zoltan_migrate_particles<N>(mesh_data->els, load_balancer, datatype, comm);

            }

            // resize linked_list
            if(mesh_data->els.size() > prev_size) {
                lscl.resize(mesh_data->els.size());
                prev_size = mesh_data->els.size();
            }

            lennard_jones::compute_one_step<N>(mesh_data, lscl.data(), head.data(), load_balancer, datatype, params, comm, frame);
        }

        wtime = MPI_Wtime() - wtime;
        double maxv;
        MPI_Allreduce(&wtime,&maxv, 1, MPI_DOUBLE, MPI_MAX, comm);
        max.push_back(maxv);
        MPI_Allgather(&wtime, 1, MPI_DOUBLE, &times.front(), 1, MPI_DOUBLE, comm);
        avg.push_back(std::accumulate(times.begin(), times.end(), 0.0) / nproc);
        double true_iteration_time = *std::max_element(times.begin(), times.end());
        frame_time += true_iteration_time;

        if(!rank) {
            printf("Frame [%d] completed in %f seconds\n", frame, true_iteration_time);
        }

        total_time += frame_time;

        // Write metrics to report file
        if (params->record)
            gather_elements_on<N, elements::Element<N>>(nproc, rank, params->npart,
                                               mesh_data->els, 0, recv_buf, datatype.elements_datatype, comm);
        if (rank == 0) {
	        if (params->record) {
                frame_file.open("data/time-series/"+std::to_string(params->seed)+"/run_cpp.csv."+std::to_string(frame+1), std::ofstream::out | std::ofstream::trunc);
                frame_formater.write_header(frame_file, params->npframe, params->simsize);
                write_frame_data(frame_file, recv_buf, frame_formater, params);
                frame_file.close();
            }
        }
    }

    MPI_Barrier(comm);
    if (!rank) std::cout << "nb lb = " << nb_lb << std::endl;
    if (rank == 0 && frame_file.is_open()) frame_file.close();

    return total_time;
}


#endif //NBMPI_SIMULATE_HPP
