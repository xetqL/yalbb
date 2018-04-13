//
// Created by xetql on 05.03.18.
//

#ifndef NBMPI_BOXRUNNER_HPP
#define NBMPI_BOXRUNNER_HPP

#include <sstream>
#include <fstream>
#include <iomanip>
#include <mpi.h>
#include <map>
#include <unordered_map>
#include <zoltan.h>

#include "../includes/ljpotential.hpp"
#include "../includes/report.hpp"
#include "../includes/physics.hpp"
#include "../includes/nbody_io.hpp"
#include "../includes/utils.hpp"
#include "../includes/geometric_load_balancer.hpp"
#include "../includes/params.hpp"
#include "../includes/spatial_elements.hpp"
#include "../includes/graph.hpp"
#include "../includes/metrics.hpp"

#include "zoltan_fn.hpp"
#include <gsl/gsl_statistics.h>
template<int N>
void run_box(FILE* fp, // Output file (at 0)
             const int npframe, // Steps per frame
             const int nframes, // Frames
             const double dt, // Time step
             std::vector<elements::Element<2>> local_elements,
             std::vector<partitioning::geometric::Domain<N>> domain_boundaries,
             load_balancing::geometric::GeometricLoadBalancer<N> load_balancer,
             const sim_param_t* params,
             MPI_Comm comm = MPI_COMM_WORLD) // Simulation params
{
    std::ofstream lb_file;
    partitioning::CommunicationDatatype datatype = elements::register_datatype<2>();
    int nproc, rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);
    double start_sim = MPI_Wtime();

    std::unordered_map<int, std::unique_ptr<std::vector<elements::Element<N> > > > plklist;

    double rm = 3.2 * std::sqrt(params->sig_lj); // r_m = 3.2 * sig
    int M = (int) (params->simsize / rm); //number of cell in a row
    float lsub = params->simsize / ((float) M); //cell size
    std::vector<elements::Element<2>> recv_buf(params->npart);

    if(params->record) load_balancing::gather_elements_on(nproc, rank, params->npart, local_elements, 0, recv_buf, load_balancer.get_element_datatype(), comm);
    if (rank == 0) {
        auto date = get_date_as_string();
        lb_file.open("load_imbalance_report-"+date+".data", std::ofstream::out | std::ofstream::trunc );
        write_report_header(lb_file, params, rank);
        if(params->record) {
            write_header(fp, params->npart, params->simsize);
            write_frame_data(fp, params->npart, &recv_buf[0]);
        }
    }

    auto local_el = local_elements;
    double begin = MPI_Wtime();

    for (int frame = 1; frame < nframes; ++frame) {
        for (int i = 0; i < npframe; ++i) {
            MPI_Barrier(comm);
            double start = MPI_Wtime();
            // Rebalance if asked
            if (params->one_shot_lb_call == (i+(frame-1)*npframe) || params->lb_interval > 0 && ((i+(frame-1)*npframe) % params->lb_interval) == 0) {
                load_balancing::gather_elements_on(nproc, rank, params->npart, local_el, 0, local_el, load_balancer.get_element_datatype(), comm);
                partitioning::geometric::Domain<N> _domain_boundary = {
                        std::make_pair(0.0, params->simsize), std::make_pair(0.0, params->simsize)};
                domain_boundaries = { _domain_boundary };
                load_balancer.load_balance(local_el, domain_boundaries);
            }

            // get particles that can potentially interact with mine
            std::vector<elements::Element<2>> remote_el = load_balancing::geometric::exchange_data<2>(local_el, domain_boundaries,datatype , comm);
            //select computation method
            switch (params->computation_method) {
                case 1:
                    lennard_jones::compute_forces(local_el, remote_el, params);
                    break;
                case 2:
                case 3:
                    lennard_jones::create_cell_linkedlist(M, lsub, local_el, remote_el, plklist);
                    lennard_jones::compute_forces(M, lsub, local_el, remote_el, plklist, params);
                    break;
            }
            leapfrog2(dt, local_el);
            leapfrog1(dt, local_el);
            apply_reflect(local_el, params->simsize);
            //finish this time step by sending particles that does not belong to me anymore...
            load_balancing::geometric::migrate_particles<2>(local_el, domain_boundaries,datatype , comm);

            double diff = (MPI_Wtime() - start) / 1e-3; //divide time by tick resolution
            std::vector<double> times(nproc);
            MPI_Gather(&diff, 1, MPI_DOUBLE, &times.front(), 1, MPI_DOUBLE, 0, comm);
            write_report_data(lb_file, i+(frame-1)*npframe, times, rank);
        }

        if(params->record) load_balancing::gather_elements_on(nproc, rank, params->npart,
                                           local_el, 0, recv_buf, load_balancer.get_element_datatype(), comm);
        if (rank == 0) {
            double end = MPI_Wtime();
            double time_spent = (end - begin);
            if(params->record) write_frame_data(fp, params->npart, &recv_buf[0]);
            printf("Frame [%d] completed in %f seconds\n", frame, time_spent);
            begin = MPI_Wtime();
        }
    }

    load_balancer.stop();
    if(rank == 0){
        double diff =(MPI_Wtime() - start_sim);
        lb_file << diff << std::endl;
        lb_file.close();
    }
}

template<int N>
void zoltan_run_box(FILE* fp,          // Output file (at 0)
                    MESH_DATA<N>* mesh_data,
                    Zoltan_Struct* load_balancer,
                    const sim_param_t* params,
                    const MPI_Comm comm = MPI_COMM_WORLD)
{
    int nproc,rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);
    std::ofstream lb_file, dataset, frame_file;
    const double dt = params->dt;
    const int nframes = params->nframes;
    const int npframe = params->npframe;
    int dim;

    SimpleXYZFormatter frame_formater;

    // ZOLTAN VARIABLES
    int changes, numGidEntries, numLidEntries, numImport, numExport;
    ZOLTAN_ID_PTR importGlobalGids, importLocalGids, exportGlobalGids, exportLocalGids;
    int *importProcs, *importToPart, *exportProcs, *exportToPart;
    double xmin, ymin, zmin, xmax, ymax, zmax;
    // END OF ZOLTAN VARIABLES

    partitioning::CommunicationDatatype datatype = elements::register_datatype<N>();
    std::vector<partitioning::geometric::Domain<N>> domain_boundaries(nproc);

    // get boundaries of all domains
    for(int part = 0; part < nproc; ++part) {
        Zoltan_RCB_Box(load_balancer, part, &dim, &xmin, &ymin, &zmin, &xmax, &ymax, &zmax);
        auto domain = partitioning::geometric::borders_to_domain<N>(xmin, ymin, zmin, xmax, ymax, zmax, params->simsize);
        domain_boundaries[part] = domain;
    }
    
    double start_sim = MPI_Wtime();
    std::unordered_map<int, std::unique_ptr<std::vector<elements::Element<N> > > > plklist;

    double rm = 3.2 * params->sig_lj; // r_m = 3.2 * sig

    int M = std::ceil(params->simsize / rm); // number of cell in a row
    float lsub = rm; //cell size

    std::vector<elements::Element<N>> recv_buf(params->npart);
    auto date = get_date_as_string();
    if(params->record) load_balancing::gather_elements_on(nproc, rank, params->npart, mesh_data->els, 0, recv_buf, datatype.elements_datatype, comm);

    std::unique_ptr<SlidingWindow<double>> window_load_imbalance, window_complexity, window_loads;

    if (rank == 0) { // Write report and outputs ...
        if(params->record) {
            frame_file.open("run_cpp.out", std::ofstream::out | std::ofstream::trunc);
            frame_formater.write_header(frame_file, params->npframe, params->simsize);
            write_frame_data(frame_file, recv_buf, frame_formater, params);
        }
        /*
        lb_file.open("LIr_"+std::to_string(params->world_size)+
                        "-"+std::to_string(params->npart)+
                        "-"+std::to_string((params->nframes*params->npframe))+
                        "-"+std::to_string((int)(params->T0))+
                        "-"+std::to_string((params->G))+
                        "-"+std::to_string((params->eps_lj))+
                        "-"+std::to_string((params->sig_lj))+
                        "-"+std::to_string((params->one_shot_lb_call))+
                        "_"+date+".data", std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
        write_report_header_bin(lb_file, params, rank);     // write header
        */
        dataset.open("dataset-rcb-"+std::to_string(params->world_size)+
                         "-"+std::to_string(params->npart)+
                         "-"+std::to_string((params->nframes*params->npframe))+
                         "-"+std::to_string((int)(params->T0))+
                         "-"+std::to_string((params->G))+
                         "-"+std::to_string((params->eps_lj))+
                         "-"+std::to_string((params->sig_lj)),
                         std::ofstream::out | std::ofstream::app | std::ofstream::binary);
        //write_report_header_bin(dataset, params, rank); // write the same header
        window_load_imbalance = std::make_unique<SlidingWindow<double>>(50); //sliding window of size 50
        window_loads          = std::make_unique<SlidingWindow<double>>(50); //sliding window of size 50
        window_complexity     = std::make_unique<SlidingWindow<double>>(50); //sliding window of size 50
    }

    std::vector<elements::Element<N>> remote_el;
    double begin = MPI_Wtime();
    std::vector<float> dataset_entry(13);
    for (int frame = 0; frame < nframes; ++frame) {
        for (int i = 0; i < npframe; ++i) {
            MPI_Barrier(comm);
            double start = MPI_Wtime();
            if ((params->one_shot_lb_call == (i+frame*npframe) || (params->lb_interval > 0 && ((i+frame*npframe) % params->lb_interval) == 0)) && (i+frame*npframe) > 0) {
                zoltan_fn_init(load_balancer, mesh_data);
                Zoltan_LB_Partition(load_balancer,           /* input (all remaining fields are output) */
                                         &changes,           /* 1 if partitioning was changed, 0 otherwise */
                                         &numGidEntries,     /* Number of integers used for a global ID */
                                         &numLidEntries,     /* Number of integers used for a local ID */
                                         &numImport,         /* Number of vertices to be sent to me */
                                         &importGlobalGids,  /* Global IDs of vertices to be sent to me */
                                         &importLocalGids,   /* Local IDs of vertices to be sent to me */
                                         &importProcs,       /* Process rank for source of each incoming vertex */
                                         &importToPart,      /* New partition for each incoming vertex */
                                         &numExport,         /* Number of vertices I must send to other processes*/
                                         &exportGlobalGids,  /* Global IDs of the vertices I must send */
                                         &exportLocalGids,   /* Local IDs of the vertices I must send */
                                         &exportProcs,       /* Process to which I send each of the vertices */
                                         &exportToPart);     /* Partition to which each vertex will belong */
                if(changes) for(int part = 0; part < nproc; ++part) { // algorithm specific ...
                    //TODO: Use bounding box and Zoltan_LB_Box_Assign etc. to be more generic..?
                    Zoltan_RCB_Box(load_balancer, part, &dim, &xmin, &ymin, &zmin, &xmax, &ymax, &zmax);
                    auto domain = partitioning::geometric::borders_to_domain<N>(xmin, ymin, zmin,
                                                                                xmax, ymax, zmax, params->simsize);
                    domain_boundaries[part] = domain;
                }
                load_balancing::geometric::migrate_zoltan<N>(mesh_data->els, numImport, numExport, exportProcs,
                                                             exportGlobalGids, datatype, MPI_COMM_WORLD);
                Zoltan_LB_Free_Part(&importGlobalGids, &importLocalGids, &importProcs, &importToPart);
                Zoltan_LB_Free_Part(&exportGlobalGids, &exportLocalGids, &exportProcs, &exportToPart);
            }
            remote_el = load_balancing::geometric::exchange_data<N>(mesh_data->els, domain_boundaries, datatype, comm, lsub);

            // update local ids
            for(size_t i = 0; i < mesh_data->els.size(); ++i) mesh_data->els[i].lid = i;

            lennard_jones::create_cell_linkedlist(M, lsub, mesh_data->els, remote_el, plklist);
            int complexity = lennard_jones::compute_forces(M, lsub, mesh_data->els, remote_el, plklist, params);

            leapfrog2(dt, mesh_data->els);
            leapfrog1(dt, mesh_data->els);
            apply_reflect(mesh_data->els, params->simsize);

            load_balancing::geometric::migrate_particles<N>(mesh_data->els, domain_boundaries, datatype, comm);

            if((params->one_shot_lb_call - 1) == (i+frame*npframe)) {
                // Particles are migrated, PEs are ready for the next time-step.
                // Now compute the metrics ...
                float iteration_time = (MPI_Wtime() - start) / 1e-3; // divide diff time by tick resolution
                start = MPI_Wtime(); // start overhead measurement
                auto cell_load = metric::topology::compute_cells_loads<double, N>(M, mesh_data->els.size(), plklist);
                // Retrieve local data to Master PE
                std::vector<float> times(nproc);
                MPI_Gather(&iteration_time, 1, MPI_FLOAT, &times.front(), 1, MPI_FLOAT, 0, comm);

                std::vector<float> loads(nproc);
                MPI_Gather(&cell_load, 1, MPI_FLOAT, &loads.front(), 1, MPI_FLOAT, 0, comm);

                std::vector<int> complexities(nproc);
                MPI_Gather(&complexity, 1, MPI_INT, &complexities.front(), 1, MPI_INT, 0, comm);
                // compute metrics and store
                if (rank == 0) {
                    float gini_times = metric::load_balancing::compute_gini_index(times);
                    float gini_loads = metric::load_balancing::compute_gini_index(loads);
                    float gini_complexities = metric::load_balancing::compute_gini_index(complexities);

                    float skewness_times = gsl_stats_float_skew(&times.front(), times.size(), 1);
                    float skewness_loads = gsl_stats_float_skew(&loads.front(), loads.size(), 1);
                    float skewness_complexities = gsl_stats_int_skew(&complexities.front(), complexities.size(), 1);

                    window_load_imbalance->add(gini_times);
                    window_loads->add(gini_loads);
                    window_complexity->add(gini_complexities);

                    // Generate y from 0 to 1 and store in a vector
                    std::vector<float> i(window_load_imbalance->data_container.size());
                    std::iota(i.begin(), i.end(), 0);

                    float slope_load_imbalance = statistic::linear_regression(window_load_imbalance->data_container,
                                                                               i).second;
                    float macd_load_imbalance = metric::load_dynamic::compute_macd(
                            window_load_imbalance->data_container);
                    float slope_loads = statistic::linear_regression(window_loads->data_container, i).second;
                    float macd_loads = metric::load_dynamic::compute_macd(window_loads->data_container);
                    float slope_complexity = statistic::linear_regression(window_complexity->data_container, i).second;
                    float macd_complexity = metric::load_dynamic::compute_macd(window_complexity->data_container);

                    dataset_entry = {
                            gini_times, gini_loads, gini_complexities,
                            skewness_times, skewness_loads, skewness_complexities,
                            slope_load_imbalance, slope_loads, slope_complexity,
                            macd_load_imbalance, macd_loads, macd_complexity,
                            0.0
                    };
                }
                float metric_overhead = (MPI_Wtime() - start) / 1e-3;
                //write_report_data_bin(lb_file, i + frame * npframe, times, rank, 0);
                ////////////////////////////////////////////////////////////////////////////////////////
            }
        }



        // Write metrics to report file
        if(params->record)
            load_balancing::gather_elements_on(nproc, rank, params->npart,
                                               mesh_data->els, 0, recv_buf, datatype.elements_datatype, comm);
        MPI_Barrier(comm);
        if (rank == 0) {
            double end = MPI_Wtime();
            double time_spent = (end - begin);

            if(params->record) {
                write_frame_data(frame_file, recv_buf, frame_formater, params);
                write_frame_data(fp, params->npart, &recv_buf[0]);
            }
            printf("Frame [%d] completed in %f seconds\n", frame, time_spent);
            begin = MPI_Wtime();
        }
    }

    MPI_Barrier(comm);
    if(rank == 0){
        double diff = (MPI_Wtime() - start_sim) / 1e-3;
        dataset_entry[dataset_entry.size() - 1] = diff;
        write_report_data_bin<float>(dataset, params->one_shot_lb_call, dataset_entry, rank);
        if(params->record){
            write_report_total_time_bin<float>(lb_file, diff, rank);
            lb_file.close();
            frame_file.close();
        }
    }
}

#endif //NBMPI_BOXRUNNER_HPP