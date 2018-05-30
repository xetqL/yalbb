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

#include "../includes/astar.hpp"
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

#ifndef WINDOW_SIZE
#define WINDOW_SIZE 30
#endif

#ifndef N_FEATURES
#define N_FEATURES 14
#endif

#ifndef N_LABEL
#define N_LABEL 1
#endif

#ifndef TICK_FREQ
#define TICK_FREQ 1 // MPI_Wtick()
#endif

template<int N>
void run_box(FILE *fp, // Output file (at 0)
             const int npframe, // Steps per frame
             const int nframes, // Frames
             const double dt, // Time step
             std::vector<elements::Element<2>> local_elements,
             std::vector<partitioning::geometric::Domain<N>> domain_boundaries,
             load_balancing::geometric::GeometricLoadBalancer<N> load_balancer,
             const sim_param_t *params,
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

    if (params->record)
        load_balancing::gather_elements_on(nproc, rank, params->npart, local_elements, 0, recv_buf,
                                           load_balancer.get_element_datatype(), comm);
    if (rank == 0) {
        auto date = get_date_as_string();
        lb_file.open("load_imbalance_report-" + date + ".data", std::ofstream::out | std::ofstream::trunc);
        write_report_header(lb_file, params, rank);
        if (params->record) {
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
            if (params->one_shot_lb_call == (i + (frame - 1) * npframe) ||
                params->lb_interval > 0 && ((i + (frame - 1) * npframe) % params->lb_interval) == 0) {
                load_balancing::gather_elements_on(nproc, rank, params->npart, local_el, 0, local_el,
                                                   load_balancer.get_element_datatype(), comm);
                partitioning::geometric::Domain<N> _domain_boundary = {
                        std::make_pair(0.0, params->simsize), std::make_pair(0.0, params->simsize)};
                domain_boundaries = {_domain_boundary};
                load_balancer.load_balance(local_el, domain_boundaries);
            }
            int r, s;
            // get particles that can potentially interact with mine
            std::vector<elements::Element<2>> remote_el = load_balancing::geometric::exchange_data<2>(local_el,
                                                                                                      domain_boundaries,
                                                                                                      datatype, comm, r,
                                                                                                      s);
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
            load_balancing::geometric::migrate_particles<2>(local_el, domain_boundaries, datatype, comm);

            double diff = (MPI_Wtime() - start) / 1e-3; //divide time by tick resolution
            std::vector<double> times(nproc);
            MPI_Gather(&diff, 1, MPI_DOUBLE, &times.front(), 1, MPI_DOUBLE, 0, comm);
            write_report_data(lb_file, i + (frame - 1) * npframe, times, rank);
        }

        if (params->record)
            load_balancing::gather_elements_on(nproc, rank, params->npart,
                                               local_el, 0, recv_buf, load_balancer.get_element_datatype(), comm);
        if (rank == 0) {
            double end = MPI_Wtime();
            double time_spent = (end - begin);
            if (params->record) write_frame_data(fp, params->npart, &recv_buf[0]);
            printf("Frame [%d] completed in %f seconds\n", frame, time_spent);
            begin = MPI_Wtime();
        }
    }

    load_balancer.stop();
    if (rank == 0) {
        double diff = (MPI_Wtime() - start_sim);
        lb_file << diff << std::endl;
        lb_file.close();
    }
}

template<int N>
void generate_dataset(MESH_DATA<N> *mesh_data,
                      Zoltan_Struct *load_balancer,
                      const sim_param_t *params,
                      const MPI_Comm comm = MPI_COMM_WORLD) {
    int nproc, rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);
    std::ofstream dataset, frame_file;
    SimpleXYZFormatter frame_formater;
    MESH_DATA<N> mem_data;
    const double dt = params->dt;
    const int nframes = params->nframes;
    const int npframe = params->npframe;

    const std::string DATASET_FILENAME = "full-dataset-" + std::to_string(params->seed) +
                                         "-" + std::to_string(params->world_size) +
                                         "-" + std::to_string(params->npart) +
                                         "-" + std::to_string((params->T0)) +
                                         "-" + std::to_string((params->G)) +
                                         "-" + std::to_string((params->simsize)) +
                                         "-" + std::to_string((params->eps_lj)) +
                                         "-" + std::to_string((params->sig_lj));
    int dim;
    double rm = 3.2 * params->sig_lj; // r_m = 3.2 * sig
    int M = std::ceil(params->simsize / rm); // number of cell in a row
    float lsub = rm; //cell size
    auto date = get_date_as_string();

    // ZOLTAN VARIABLES
    int changes, numGidEntries, numLidEntries, numImport, numExport;
    ZOLTAN_ID_PTR importGlobalGids, importLocalGids, exportGlobalGids, exportLocalGids;
    int *importProcs, *importToPart, *exportProcs, *exportToPart;
    double xmin, ymin, zmin, xmax, ymax, zmax;
    // END OF ZOLTAN VARIABLES

    partitioning::CommunicationDatatype datatype = elements::register_datatype<N>();
    std::vector<partitioning::geometric::Domain<N>> domain_boundaries(nproc);

    // get boundaries of all domains
    for (int part = 0; part < nproc; ++part) {
        Zoltan_RCB_Box(load_balancer, part, &dim, &xmin, &ymin, &zmin, &xmax, &ymax, &zmax);
        auto domain = partitioning::geometric::borders_to_domain<N>(xmin, ymin, zmin, xmax, ymax, zmax,
                                                                    params->simsize);
        domain_boundaries[part] = domain;
    }
    std::unordered_map<int, std::unique_ptr<std::vector<elements::Element<N> > > > plklist;

    std::vector<elements::Element<N>> remote_el;
    std::shared_ptr<SlidingWindow<double>> window_gini_times, window_gini_complexities, window_times, window_gini_communications;
    if (rank == 0) { // Write report and outputs ...
        window_gini_times = std::make_shared<SlidingWindow<double>>(WINDOW_SIZE);
        window_times = std::make_shared<SlidingWindow<double>>(WINDOW_SIZE);
        window_gini_complexities = std::make_shared<SlidingWindow<double>>(WINDOW_SIZE);
        window_gini_communications = std::make_shared<SlidingWindow<double>>(WINDOW_SIZE);
    }
    std::vector<float> dataset_entry(N_FEATURES + N_LABEL), features(N_FEATURES + N_LABEL);

    double compute_time_with_lb = 0.0;
    double compute_time_without_lb = 0.0;
    int time_step_index = 0;

    bool start_with_lb = params->start_with_lb;
    bool with_lb = start_with_lb;

    auto saved_domains = domain_boundaries;
    mem_data = *mesh_data;
    while (time_step_index < (nframes * npframe)) {
        if ((time_step_index % DELTA_LB_CALL) == 0 && time_step_index > 0)
            if (!with_lb) { // NO LB PART
                with_lb = true;
                if (start_with_lb) {
                    if (rank == 0)
                        std::cout << "LB= " << compute_time_with_lb << "; NoLB= " << compute_time_without_lb
                                  << std::endl;
                    metric::io::write_load_balancing_reports(dataset, DATASET_FILENAME, time_step_index,
                                                             compute_time_without_lb - compute_time_with_lb,
                                                             features, rank, params);
                    compute_time_with_lb = 0.0;
                    compute_time_without_lb = 0.0;
                    mem_data = *mesh_data;
                }
                if (!start_with_lb) {
                    time_step_index -= DELTA_LB_CALL;
                    domain_boundaries = saved_domains;
                    *mesh_data = mem_data;
                }

                if (rank == 0)
                    std::cout << " Compute time for: " << time_step_index << " to " << (time_step_index + DELTA_LB_CALL)
                              << " with load balancing" << std::endl;
            } else { // LB PART
                with_lb = false;
                if (!start_with_lb) {
                    if (rank == 0)
                        std::cout << "LB= " << compute_time_with_lb << "; NoLB= " << compute_time_without_lb
                                  << std::endl;
                    metric::io::write_load_balancing_reports(dataset, DATASET_FILENAME, time_step_index,
                                                             compute_time_with_lb - compute_time_without_lb,
                                                             features, rank, params);
                    compute_time_with_lb = 0.0;
                    compute_time_without_lb = 0.0;
                    mem_data = *mesh_data;
                    saved_domains = domain_boundaries;
                }
                if (start_with_lb) {
                    time_step_index -= DELTA_LB_CALL;
                    domain_boundaries = saved_domains;
                    *mesh_data = mem_data;
                }
                if (rank == 0)
                    std::cout << " Compute time for: " << time_step_index << " to " << (time_step_index + DELTA_LB_CALL)
                              << " without load balancing" << std::endl;
            }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        MPI_Barrier(comm);
        double start = MPI_Wtime();
        if ((time_step_index % DELTA_LB_CALL) == 0 && time_step_index > 0 && with_lb) {
            if (start_with_lb) features = dataset_entry;
            zoltan_fn_init(load_balancer, mesh_data);
            Zoltan_LB_Partition(load_balancer, &changes, &numGidEntries, &numLidEntries,
                                &numImport, &importGlobalGids, &importLocalGids, &importProcs, &importToPart,
                                &numExport, &exportGlobalGids, &exportLocalGids, &exportProcs, &exportToPart);
            if (changes) {
                for (int part = 0; part < nproc; ++part) { // algorithm specific ...
                    Zoltan_RCB_Box(load_balancer, part, &dim, &xmin, &ymin, &zmin, &xmax, &ymax, &zmax);
                    auto domain = partitioning::geometric::borders_to_domain<N>(xmin, ymin, zmin,
                                                                                xmax, ymax, zmax, params->simsize);
                    domain_boundaries[part] = domain;
                }
            }
            load_balancing::geometric::migrate_zoltan<N>(mesh_data->els, numImport, numExport, exportProcs,
                                                         exportGlobalGids, datatype, MPI_COMM_WORLD);
            Zoltan_LB_Free_Part(&importGlobalGids, &importLocalGids, &importProcs, &importToPart);
            Zoltan_LB_Free_Part(&exportGlobalGids, &exportLocalGids, &exportProcs, &exportToPart);
        } else if ((time_step_index % DELTA_LB_CALL) == 0 && time_step_index > 0 && !with_lb) {
            if (!start_with_lb) features = dataset_entry;
            load_balancing::geometric::migrate_particles<N>(mesh_data->els, domain_boundaries, datatype, comm);
        } else load_balancing::geometric::migrate_particles<N>(mesh_data->els, domain_boundaries, datatype, comm);

        MPI_Barrier(comm);
        int received = 0, sent = 0;
        remote_el = load_balancing::geometric::exchange_data<N>(mesh_data->els, domain_boundaries, datatype, comm,
                                                                received, sent, lsub);

        // update local ids
        for (size_t i = 0; i < mesh_data->els.size(); ++i) mesh_data->els[i].lid = i;
        lennard_jones::create_cell_linkedlist(M, lsub, mesh_data->els, remote_el, plklist);
        float complexity = (float) lennard_jones::compute_forces(M, lsub, mesh_data->els, remote_el, plklist, params);

        leapfrog2(dt, mesh_data->els);
        leapfrog1(dt, mesh_data->els);
        apply_reflect(mesh_data->els, params->simsize);

        double my_iteration_time = (MPI_Wtime() - start) / TICK_FREQ;
        MPI_Barrier(comm);
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        std::vector<double> times(nproc);
        MPI_Gather(&my_iteration_time, 1, MPI_DOUBLE, &times.front(), 1, MPI_DOUBLE, 0, comm);
        double true_iteration_time = *std::max_element(times.begin(), times.end());

        if (with_lb) compute_time_with_lb += true_iteration_time;
        else compute_time_without_lb += true_iteration_time;

        dataset_entry = metric::compute_metrics(window_times, window_gini_times,
                                                window_gini_complexities, window_gini_communications,
                                                true_iteration_time, times,
                                                sent, received, complexity, rank, comm);
        time_step_index++;
    }

    if (rank == 0) if (dataset.is_open()) dataset.close();
    if (rank == 0) if (frame_file.is_open()) frame_file.close();
}

template<int N>
void zoltan_run_box_dataset(FILE *fp,          // Output file (at 0)
                            MESH_DATA<N> *mesh_data,
                            Zoltan_Struct *load_balancer,
                            const sim_param_t *params,
                            const MPI_Comm comm = MPI_COMM_WORLD) {
    int nproc, rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);
    std::ofstream dataset;
    const double dt = params->dt;
    const int nframes = params->nframes;
    const int npframe = params->npframe;


    const std::string DATASET_FILENAME = "dataset-rcb-" + std::to_string(params->seed) +
                                         "-" + std::to_string(params->world_size) +
                                         "-" + std::to_string(params->npart) +
                                         "-" + std::to_string((params->T0)) +
                                         "-" + std::to_string((params->G)) +
                                         "-" + std::to_string((params->simsize)) +
                                         "-" + std::to_string((params->eps_lj)) +
                                         "-" + std::to_string((params->sig_lj));
    int dim;
    double rm = 3.2 * params->sig_lj; // r_m = 3.2 * sig
    int M = std::ceil(params->simsize / rm); // number of cell in a row
    float lsub = rm; //cell size
    auto date = get_date_as_string();

    // ZOLTAN VARIABLES
    int changes, numGidEntries, numLidEntries, numImport, numExport;
    ZOLTAN_ID_PTR importGlobalGids, importLocalGids, exportGlobalGids, exportLocalGids;
    int *importProcs, *importToPart, *exportProcs, *exportToPart;
    double xmin, ymin, zmin, xmax, ymax, zmax;
    // END OF ZOLTAN VARIABLES

    partitioning::CommunicationDatatype datatype = elements::register_datatype<N>();
    std::vector<partitioning::geometric::Domain<N>> domain_boundaries(nproc);

    // get boundaries of all domains
    for (int part = 0; part < nproc; ++part) {
        Zoltan_RCB_Box(load_balancer, part, &dim, &xmin, &ymin, &zmin, &xmax, &ymax, &zmax);
        auto domain = partitioning::geometric::borders_to_domain<N>(xmin, ymin, zmin, xmax, ymax, zmax,
                                                                    params->simsize);
        domain_boundaries[part] = domain;
    }
    std::shared_ptr<SlidingWindow<double>> window_gini_times, window_gini_complexities, window_times, window_gini_communications;
    std::unordered_map<int, std::unique_ptr<std::vector<elements::Element<N> > > > plklist;

    std::vector<elements::Element<N>> remote_el;

    if (rank == 0) { // Write report and outputs ...
        window_gini_times = std::make_shared<SlidingWindow<double>>(WINDOW_SIZE);
        window_times = std::make_shared<SlidingWindow<double>>(WINDOW_SIZE);
        window_gini_complexities = std::make_shared<SlidingWindow<double>>(WINDOW_SIZE);
        window_gini_communications = std::make_shared<SlidingWindow<double>>(WINDOW_SIZE);
    }

    std::vector<float> dataset_entry(N_FEATURES + N_LABEL);

    double total_metric_computation_time = 0.0;
    double compute_time_after_lb = 0.0;
    for (int frame = 0; frame < nframes; ++frame) {
        for (int i = 0; i < npframe; ++i) {
            if ((params->one_shot_lb_call + DELTA_LB_CALL) == (i + frame * npframe)) {
                if (rank == 0) {
                    std::cout << " Time within " << ((i + frame * npframe) - DELTA_LB_CALL) << " and "
                              << (i + frame * npframe) << ": " << compute_time_after_lb << " ms. "
                              << ", metrics: " << total_metric_computation_time << std::endl;
                    dataset_entry[dataset_entry.size() - 1] = compute_time_after_lb;
                    dataset.open(DATASET_FILENAME, std::ofstream::out | std::ofstream::app | std::ofstream::binary);
                    write_report_data_bin<float>(dataset, params->one_shot_lb_call, dataset_entry, rank);
                    dataset.close();
                    std::cout << " Go to the next experiment. " << std::endl;
                }
                return;
            }
            MPI_Barrier(comm);
            double start = MPI_Wtime();
            if ((params->one_shot_lb_call == (i + frame * npframe)) && (i + frame * npframe) > 0) {
                compute_time_after_lb = 0.0;
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
                if (changes)
                    for (int part = 0; part < nproc; ++part) { // algorithm specific ...
                        Zoltan_RCB_Box(load_balancer, part, &dim, &xmin, &ymin, &zmin, &xmax, &ymax, &zmax);
                        auto domain = partitioning::geometric::borders_to_domain<N>(xmin, ymin, zmin,
                                                                                    xmax, ymax, zmax, params->simsize);
                        domain_boundaries[part] = domain;
                    }
                load_balancing::geometric::migrate_zoltan<N>(mesh_data->els, numImport, numExport, exportProcs,
                                                             exportGlobalGids, datatype, MPI_COMM_WORLD);

                Zoltan_LB_Free_Part(&importGlobalGids, &importLocalGids, &importProcs, &importToPart);
                Zoltan_LB_Free_Part(&exportGlobalGids, &exportLocalGids, &exportProcs, &exportToPart);
            } else load_balancing::geometric::migrate_particles<N>(mesh_data->els, domain_boundaries, datatype, comm);
            MPI_Barrier(comm);
            int received = 0, sent = 0;
            remote_el = load_balancing::geometric::exchange_data<N>(mesh_data->els, domain_boundaries, datatype, comm,
                                                                    received, sent, lsub);

            // update local ids
            for (size_t i = 0; i < mesh_data->els.size(); ++i) mesh_data->els[i].lid = i;

            lennard_jones::create_cell_linkedlist(M, lsub, mesh_data->els, remote_el, plklist);
            float complexity = (float) lennard_jones::compute_forces(M, lsub, mesh_data->els, remote_el, plklist,
                                                                     params);

            leapfrog2(dt, mesh_data->els);
            leapfrog1(dt, mesh_data->els);
            apply_reflect(mesh_data->els, params->simsize);

            double my_iteration_time = (MPI_Wtime() - start) / TICK_FREQ;
            MPI_Barrier(comm);
            // gather local data on Master PE
            std::vector<double> times(nproc);
            MPI_Gather(&my_iteration_time, 1, MPI_DOUBLE, &times.front(), 1, MPI_DOUBLE, 0, comm);
            double true_iteration_time = *std::max_element(times.begin(), times.end());
            compute_time_after_lb += true_iteration_time;

            if ((i + frame * npframe) > params->one_shot_lb_call - (WINDOW_SIZE) &&
                (i + frame * npframe) < params->one_shot_lb_call) {
                dataset_entry = metric::compute_metrics(window_times, window_gini_times,
                                                        window_gini_complexities, window_gini_communications,
                                                        true_iteration_time, times,
                                                        sent, received, complexity, rank, comm);
            } // end of metric computation
        } // end of time-steps
    } // end of frames
    if (rank == 0) dataset.close();

    if (rank == 0) {
        std::cout << " Time within " << params->one_shot_lb_call << " and "
                  << params->npframe * params->nframes << ": " << compute_time_after_lb << " ms. "
                  << ", metrics: " << total_metric_computation_time << std::endl;
        dataset_entry[dataset_entry.size() - 1] = compute_time_after_lb;
        dataset.open(DATASET_FILENAME, std::ofstream::out | std::ofstream::app | std::ofstream::binary);
        write_report_data_bin<float>(dataset, params->one_shot_lb_call, dataset_entry, rank);
        dataset.close();
        std::cout << " Go to the next experiment. " << std::endl;
    }
}


template<int N>
void compute_dataset_base_gain(FILE *fp,          // Output file (at 0)
                               MESH_DATA<N> *mesh_data,
                               Zoltan_Struct *load_balancer,
                               const sim_param_t *params,
                               const MPI_Comm comm = MPI_COMM_WORLD) {
    int nproc, rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);
    std::ofstream dataset;
    const std::string DATASET_FILENAME = "dataset-base-times-rcb-" + std::to_string(params->seed) +
                                         "-" + std::to_string(params->world_size) +
                                         "-" + std::to_string(params->npart) +
                                         "-" + std::to_string((params->T0)) +
                                         "-" + std::to_string((params->G)) +
                                         "-" + std::to_string((params->simsize)) +
                                         "-" + std::to_string((params->eps_lj)) +
                                         "-" + std::to_string((params->sig_lj));
    if (rank == 0) dataset.open(DATASET_FILENAME, std::ofstream::out | std::ofstream::app | std::ofstream::binary);

    const double dt = params->dt;
    const int nframes = params->nframes;
    const int npframe = params->npframe;
    // ZOLTAN VARIABLES
    int dim;
    double xmin, ymin, zmin, xmax, ymax, zmax;
    // END OF ZOLTAN VARIABLES
    std::unique_ptr<SlidingWindow<double>> window_gini_times, window_gini_complexities, window_times, window_gini_communications;
    if (rank == 0) { // Write report and outputs ...
        window_gini_times = std::make_unique<SlidingWindow<double>>(WINDOW_SIZE);
        window_times = std::make_unique<SlidingWindow<double>>(WINDOW_SIZE);
        window_gini_complexities = std::make_unique<SlidingWindow<double>>(WINDOW_SIZE);
        window_gini_communications = std::make_unique<SlidingWindow<double>>(WINDOW_SIZE);
    }
    std::vector<float> dataset_entry(N_FEATURES + N_LABEL);

    partitioning::CommunicationDatatype datatype = elements::register_datatype<N>();
    std::vector<partitioning::geometric::Domain<N>> domain_boundaries(nproc);

    // get boundaries of all domains
    for (int part = 0; part < nproc; ++part) {
        Zoltan_RCB_Box(load_balancer, part, &dim, &xmin, &ymin, &zmin, &xmax, &ymax, &zmax);
        auto domain = partitioning::geometric::borders_to_domain<N>(xmin, ymin, zmin, xmax, ymax, zmax,
                                                                    params->simsize);
        domain_boundaries[part] = domain;
    }

    std::unordered_map<int, std::unique_ptr<std::vector<elements::Element<N> > > > plklist;
    double rm = 3.2 * params->sig_lj; // r_m = 3.2 * sig
    int M = std::ceil(params->simsize / rm); // number of cell in a row
    float lsub = rm; //cell size

    std::vector<elements::Element<N>> remote_el;
    double compute_time_after_lb = 0.0;
    for (int frame = 0; frame < nframes; ++frame) {
        for (int i = 0; i < npframe; ++i) {
            if ((i + frame * npframe) % DELTA_LB_CALL == 0 && (i + frame * npframe) > 0 && rank == 0) {
                std::cout << " Time within " << ((i + frame * npframe) - DELTA_LB_CALL) << " and "
                          << (i + frame * npframe) << " is " << compute_time_after_lb << " ms" << std::endl;
                dataset_entry[dataset_entry.size() - 1] = compute_time_after_lb;
                //write_metric_data_bin(dataset, (i+frame*npframe) - DELTA_LB_CALL, dataset_entry, rank);
                write_report_data_bin<float>(dataset, (i + frame * npframe) - DELTA_LB_CALL, dataset_entry, rank);
                compute_time_after_lb = 0.0;
            }
            MPI_Barrier(comm);
            double start = MPI_Wtime();
            load_balancing::geometric::migrate_particles<N>(mesh_data->els, domain_boundaries, datatype, comm);
            MPI_Barrier(comm);
            int received = 0, sent = 0;
            remote_el = load_balancing::geometric::exchange_data<N>(mesh_data->els, domain_boundaries, datatype, comm,
                                                                    received, sent, lsub);
            // update local ids
            for (size_t i = 0; i < mesh_data->els.size(); ++i) mesh_data->els[i].lid = i;
            lennard_jones::create_cell_linkedlist(M, lsub, mesh_data->els, remote_el, plklist);
            float complexity = (float) lennard_jones::compute_forces(M, lsub, mesh_data->els, remote_el, plklist,
                                                                     params);
            leapfrog2(dt, mesh_data->els);
            leapfrog1(dt, mesh_data->els);
            apply_reflect(mesh_data->els, params->simsize);
            double my_iteration_time = (MPI_Wtime() - start) / TICK_FREQ;
            MPI_Barrier(comm);
            std::vector<double> times(nproc);
            MPI_Gather(&my_iteration_time, 1, MPI_DOUBLE, &times.front(), 1, MPI_DOUBLE, 0, comm);
            double true_iteration_time = *std::max_element(times.begin(), times.end());
            compute_time_after_lb += true_iteration_time;

            std::vector<float> communications(nproc);
            float fsent = (float) (sent + received);
            MPI_Gather(&fsent, 1, MPI_FLOAT, &communications.front(), 1, MPI_FLOAT, 0, comm);

            std::vector<float> complexities(nproc);
            MPI_Gather(&complexity, 1, MPI_FLOAT, &complexities.front(), 1, MPI_FLOAT, 0, comm);

            if (rank == 0) {
                float gini_times = (float) metric::load_balancing::compute_gini_index<double>(times);
                float gini_complexities = metric::load_balancing::compute_gini_index(complexities);
                float gini_communications = metric::load_balancing::compute_gini_index(communications);

                float skewness_times = (float) gsl_stats_skew(&times.front(), 1, times.size());
                float skewness_complexities = gsl_stats_float_skew(&complexities.front(), 1, complexities.size());
                float skewness_communications = gsl_stats_float_skew(&communications.front(), 1, communications.size());

                window_times->add(true_iteration_time);
                window_gini_complexities->add(gini_complexities);
                window_gini_times->add(gini_times);
                window_gini_communications->add(gini_communications);

                // Generate y from 0 to 1 and store in a vector
                std::vector<float> it(window_gini_times->data_container.size());
                std::iota(it.begin(), it.end(), 0);

                float slope_gini_times = statistic::linear_regression<float>(it,
                                                                             window_gini_times->data_container).first;
                float macd_gini_times = metric::load_dynamic::compute_macd_ema<float>(window_gini_times->data_container,
                                                                                      12, 26, 2.0 /
                                                                                              (window_gini_times->data_container.size() +
                                                                                               1));

                float slope_gini_complexity = statistic::linear_regression<float>(it,
                                                                                  window_gini_complexities->data_container).first;
                float macd_gini_complexity = metric::load_dynamic::compute_macd_ema<float>(
                        window_gini_complexities->data_container, 12, 26,
                        1.0 / (window_gini_complexities->data_container.size() + 1));

                float slope_gini_communications = statistic::linear_regression<float>(it,
                                                                                      window_gini_communications->data_container).first;
                float macd_gini_communications = metric::load_dynamic::compute_macd_ema<float>(
                        window_gini_communications->data_container, 12, 26,
                        1.0 / (window_gini_complexities->data_container.size() + 1));

                float slope_times = statistic::linear_regression<float>(it, window_times->data_container).first;
                float macd_times = metric::load_dynamic::compute_macd_ema<float>(window_times->data_container, 12, 26,
                                                                                 1.0 /
                                                                                 (window_times->data_container.size() +
                                                                                  1));

                dataset_entry = {
                        gini_times, gini_complexities, gini_communications,
                        skewness_times, skewness_complexities, skewness_communications,
                        slope_gini_times, slope_gini_complexity, slope_times, slope_gini_communications,
                        macd_gini_times, macd_gini_complexity, macd_times, macd_gini_communications, 0.0
                };
            }
        } // end of time-steps
    } // end of frames
    if (rank == 0) dataset.close();
}

template<int N>
void zoltan_run_box(FILE *fp,          // Output file (at 0)
                    MESH_DATA<N> *mesh_data,
                    Zoltan_Struct *load_balancer,
                    const sim_param_t *params,
                    const MPI_Comm comm = MPI_COMM_WORLD) {
    int nproc, rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);
    std::ofstream lb_file, metric_file, frame_file;
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
    std::unordered_map<int, std::unique_ptr<std::vector<elements::Element<N> > > > plklist;

    // get boundaries of all domains
    for (int part = 0; part < nproc; ++part) {
        Zoltan_RCB_Box(load_balancer, part, &dim, &xmin, &ymin, &zmin, &xmax, &ymax, &zmax);
        auto domain = partitioning::geometric::borders_to_domain<N>(xmin, ymin, zmin, xmax, ymax, zmax,
                                                                    params->simsize);
        domain_boundaries[part] = domain;
    }

    double start_sim = MPI_Wtime();

    double rm = 3.2 * params->sig_lj; // r_m = 3.2 * sig

    int M = std::ceil(params->simsize / rm); // number of cell in a row
    float lsub = rm; //cell size

    std::vector<elements::Element<N>> recv_buf(params->npart);
    auto date = get_date_as_string();
    if (params->record)
        load_balancing::gather_elements_on(nproc, rank, params->npart, mesh_data->els, 0, recv_buf,
                                           datatype.elements_datatype, comm);
    if (rank == 0) { // Write report and outputs ...
        lb_file.open("LIr_" + std::to_string(params->world_size) +
                     "-" + std::to_string(params->npart) +
                     "-" + std::to_string((params->nframes * params->npframe)) +
                     "-" + std::to_string((int) (params->T0)) +
                     "-" + std::to_string((params->G)) +
                     "-" + std::to_string((params->eps_lj)) +
                     "-" + std::to_string((params->sig_lj)) +
                     "-" + std::to_string((params->one_shot_lb_call)) +
                     "_" + date + ".data", std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
        write_report_header_bin(lb_file, params, rank);     // write header

        metric_file.open(params->uuid + "-" + date + ".metrics",
                         std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
        write_report_header_bin(metric_file, params, rank, rank); // write the same header

        if (params->record) {
            //write_header(fp, params->npart, params->simsize);
            //write_frame_data(fp, params->npart, &recv_buf[0]);
            frame_file.open("run_cpp.out", std::ofstream::out | std::ofstream::trunc);
            frame_formater.write_header(frame_file, params->npframe, params->simsize);
            write_frame_data(frame_file, recv_buf, frame_formater, params);
        }

    }
    std::vector<elements::Element<N>> remote_el;
    double begin = MPI_Wtime();
    for (int frame = 0; frame < nframes; ++frame) {
        for (int i = 0; i < npframe; ++i) {
            MPI_Barrier(comm);
            // double start = MPI_Wtime();
            // Load balance criteria...
            if (params->one_shot_lb_call == (i + frame * npframe) ||
                (params->lb_interval > 0 && ((i + frame * npframe) % params->lb_interval) == 0)) {
                zoltan_fn_init(load_balancer, mesh_data);
                Zoltan_LB_Partition(load_balancer,      /* input (all remaining fields are output) */
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
                if (changes)
                    for (int part = 0; part < nproc; ++part) {
                        Zoltan_RCB_Box(load_balancer, part, &dim, &xmin, &ymin, &zmin, &xmax, &ymax, &zmax);
                        auto domain = partitioning::geometric::borders_to_domain<N>(xmin, ymin, zmin, xmax, ymax, zmax,
                                                                                    params->simsize);
                        domain_boundaries[part] = domain;
                    }
                load_balancing::geometric::migrate_zoltan<N>(mesh_data->els, numImport, numExport, exportProcs,
                                                             exportGlobalGids, datatype, MPI_COMM_WORLD);
                Zoltan_LB_Free_Part(&importGlobalGids, &importLocalGids, &importProcs, &importToPart);
                Zoltan_LB_Free_Part(&exportGlobalGids, &exportLocalGids, &exportProcs, &exportToPart);
            }

            int received = 0, sent = 0;
            remote_el = load_balancing::geometric::exchange_data<N>(mesh_data->els, domain_boundaries, datatype, comm,
                                                                    received, sent, lsub);

            // update local ids
            for (size_t i = 0; i < mesh_data->els.size(); ++i) mesh_data->els[i].lid = i;

            lennard_jones::create_cell_linkedlist(M, lsub, mesh_data->els, remote_el, plklist);

            lennard_jones::compute_forces(M, lsub, mesh_data->els, remote_el, plklist, params);

            leapfrog2(dt, mesh_data->els);
            leapfrog1(dt, mesh_data->els);
            apply_reflect(mesh_data->els, params->simsize);

            load_balancing::geometric::migrate_particles<N>(mesh_data->els, domain_boundaries, datatype, comm);

        }
        // Send metrics

        // Write metrics to report file
        if (params->record)
            load_balancing::gather_elements_on(nproc, rank, params->npart,
                                               mesh_data->els, 0, recv_buf, datatype.elements_datatype, comm);
        MPI_Barrier(comm);
        if (rank == 0) {
            double end = MPI_Wtime();
            double time_spent = (end - begin);
            if (params->record) {
                write_frame_data(frame_file, recv_buf, frame_formater, params);
                //write_frame_data(fp, params->npart, &recv_buf[0]);
            }
            printf("Frame [%d] completed in %f seconds\n", frame, time_spent);
            begin = MPI_Wtime();
        }
    }

    MPI_Barrier(comm);
    if (rank == 0) {
        double diff = (MPI_Wtime() - start_sim) / 1e-3;
        write_report_total_time_bin<float>(lb_file, diff, rank);
        lb_file.close();
        frame_file.close();
        metric_file.close();
    }
}

template<int N>
std::list<std::shared_ptr<Node<MESH_DATA<N>, std::vector<partitioning::geometric::Domain<N>>>>> astar_runner(
        MESH_DATA<N> *p_mesh_data,
        Zoltan_Struct *load_balancer,
        const sim_param_t *params,
        const MPI_Comm comm = MPI_COMM_WORLD) {

    int nproc, rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);
    const int nframes = params->nframes;
    const int npframe = params->npframe;
    double it_start, true_iteration_time, my_iteration_time;
    MESH_DATA<N> mesh_data = *p_mesh_data;
    using Domain = std::vector<partitioning::geometric::Domain<N>>;

    partitioning::CommunicationDatatype datatype = elements::register_datatype<N>();
    Domain domain_boundaries(nproc);
    std::unordered_map<int, std::unique_ptr<std::vector<elements::Element<N> > > > plklist;

    std::priority_queue<
            std::shared_ptr<Node<MESH_DATA<N>, Domain> >,
            std::vector<std::shared_ptr<Node<MESH_DATA<N>, Domain> > >,
            Compare<MESH_DATA<N>, Domain> > queue;

    std::shared_ptr<SlidingWindow<double>> window_gini_times, window_gini_complexities, window_times, window_gini_communications;
    window_gini_times = std::make_shared<SlidingWindow<double>>(params->npframe / 2);
    window_times = std::make_shared<SlidingWindow<double>>(params->npframe / 2);
    window_gini_complexities = std::make_shared<SlidingWindow<double>>(params->npframe / 2);
    window_gini_communications = std::make_shared<SlidingWindow<double>>(params->npframe / 2);
    std::vector<float> dataset_entry(N_FEATURES + N_LABEL), features(N_FEATURES + N_LABEL);
    std::vector<double> times(nproc);

    std::shared_ptr<Node<MESH_DATA<N>, Domain>> current_node = std::make_shared<Node<MESH_DATA<N>, Domain>>(mesh_data,
                                                                                                            domain_boundaries),
            solution;
    current_node->metrics_before_decision = dataset_entry;
    current_node->last_metric = dataset_entry;

    std::list<std::shared_ptr<Node<MESH_DATA<N>, Domain>>> solution_path;
    //solution.push_front(current_node);
    int it = 0;
    double time = 0, start, child_cost, true_child_cost;
    const int total_iteration = nframes * npframe;

    //Compute the optimal time per step////////////////////////////////////////////////////////////////////////////////// Get the group of processes in MPI_COMM_WORLD
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    int ranks[1] = {0};
// Construct a group containing all of the prime ranks in world_group
    MPI_Group foreman_group;
    MPI_Group_incl(world_group, 1, ranks, &foreman_group);
// Create a new communicator based on the group
    MPI_Comm foreman_comm;
    MPI_Comm_create_group(MPI_COMM_WORLD, foreman_group, 0, &foreman_comm);
    MESH_DATA<N> tmp_data;
    Domain tmp_domain_boundary = {{
            std::make_pair(0.0, params->simsize), std::make_pair(0.0, params->simsize)}};
    load_balancing::gather_elements_on(nproc, rank, params->npart, mesh_data.els, 0, tmp_data.els,
                                       datatype.elements_datatype, comm);
    MESH_DATA<N> *p_tmp_data = &tmp_data;
    double optimal_step_time;
    if (rank == 0) {
        it_start = MPI_Wtime();
        load_balancing::geometric::migrate_particles<N>(p_tmp_data->els, tmp_domain_boundary, datatype, foreman_comm);
        auto computation_info = lennard_jones::compute_one_step<N>(p_tmp_data, plklist, tmp_domain_boundary, datatype, params,
                                                               foreman_comm);
        optimal_step_time = (MPI_Wtime() - it_start) / nproc;
    }
    MPI_Bcast(&optimal_step_time, 1, MPI_DOUBLE, 0, comm);
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if(rank==0) std::cout << "Optimal time: " << (optimal_step_time) << std::endl;

    MPI_Barrier(comm);
    MPI_Group_free(&foreman_group);
    MPI_Group_free(&world_group);
    if(rank==0) MPI_Comm_free(&foreman_comm);

    while (it < nframes * npframe) {
        auto children = current_node->get_children();
        mesh_data = children.first->mesh_data; //TODO: It is a shallow copy not a deep copy!!! fix this!
        domain_boundaries = children.first->domain;

        child_cost = 0;
        MPI_Barrier(comm);
        for (int i = 0; i < npframe; i++) {
            it_start = MPI_Wtime();
            if (i == 0)
                zoltan_load_balance<N>(&mesh_data, domain_boundaries, load_balancer, nproc, params, datatype, comm);
            if (i > 0)
                load_balancing::geometric::migrate_particles<N>(mesh_data.els, domain_boundaries, datatype, comm);
            MPI_Barrier(comm);
            std::tuple<int, int, int> computation_info;
            try{
                computation_info = lennard_jones::compute_one_step<N>(&mesh_data, plklist, domain_boundaries, datatype,
                                                                           params, comm);
                my_iteration_time = MPI_Wtime() - it_start;
                MPI_Allgather(&my_iteration_time, 1, MPI_DOUBLE, &times.front(), 1, MPI_DOUBLE, comm);
                true_iteration_time = *std::max_element(times.begin(), times.end());
                int complexity = std::get<0>(computation_info), received = std::get<1>(
                        computation_info), sent = std::get<2>(computation_info);
                dataset_entry = metric::all_compute_metrics(window_times, window_gini_times,
                                                            window_gini_complexities, window_gini_communications,
                                                            true_iteration_time, times, sent, received, complexity, comm);
                child_cost += true_iteration_time;
            } catch (const std::runtime_error& error){
                std::cout << "Panic! ";
                std::cout << children.first << std::endl;
                throw new std::runtime_error("particle out domain");
            }

        }
        MPI_Barrier(comm);
        MPI_Allreduce(&child_cost, &true_child_cost, 1, MPI_DOUBLE, MPI_MAX, comm);

        children.first->mesh_data = mesh_data;
        children.first->end_it = it + npframe;
        children.first->node_cost = true_child_cost;
        children.first->heuristic_cost = (total_iteration - (children.first->end_it)) * optimal_step_time;
        children.first->domain = domain_boundaries;
        children.first->path_cost += true_child_cost;
        children.first->last_metric = dataset_entry;

        mesh_data = children.second->mesh_data;
        domain_boundaries = children.second->domain;
        child_cost = 0;
        MPI_Barrier(comm);
        for (int i = 0; i < npframe; i++) {
            it_start = MPI_Wtime();
            load_balancing::geometric::migrate_particles<N>(mesh_data.els, domain_boundaries, datatype, comm);
            MPI_Barrier(comm);
            std::tuple<int, int, int> computation_info;
            try{
                computation_info = lennard_jones::compute_one_step<N>(&mesh_data, plklist, domain_boundaries, datatype,
                                                                      params, comm);
                my_iteration_time = MPI_Wtime() - it_start;
                MPI_Allgather(&my_iteration_time, 1, MPI_DOUBLE, &times.front(), 1, MPI_DOUBLE, comm);
                true_iteration_time = *std::max_element(times.begin(), times.end());
                int complexity = std::get<0>(computation_info), received = std::get<1>(
                        computation_info), sent = std::get<2>(computation_info);
                dataset_entry = metric::all_compute_metrics(window_times, window_gini_times,
                                                            window_gini_complexities, window_gini_communications,
                                                            true_iteration_time, times, sent, received, complexity, comm);
                child_cost += true_iteration_time;
            } catch (const std::runtime_error& error){
                std::cout << "Panic! ";
                std::cout << children.second << std::endl;
                throw new std::runtime_error("particle out domain");
            }
        }
        MPI_Allreduce(&child_cost, &true_child_cost, 1, MPI_DOUBLE, MPI_MAX, comm);

        children.second->mesh_data = mesh_data;
        children.second->end_it = it + npframe;
        children.second->node_cost = true_child_cost;
        children.second->heuristic_cost =
                true_child_cost + (total_iteration - (children.second->end_it + npframe)) * optimal_step_time;
        children.second->domain = domain_boundaries;
        children.second->path_cost += true_child_cost;
        children.second->last_metric = dataset_entry;

        queue.push(children.first);
        queue.push(children.second);
        current_node = queue.top();
        queue.pop();

        if (current_node->end_it >= nframes * npframe) {
            solution = current_node;
        }

        it = current_node->end_it;
        MPI_Barrier(comm);
    }

    if (!rank) std::cout << solution->cost() << " seconds" << std::endl;
    //retrieve best path
    while (solution->idx > 0) {
        solution_path.push_front(solution);
        solution = solution->parent;
    }
    return solution_path;
}

#endif //NBMPI_BOXRUNNER_HPP
