//
// Created by xetql on 27.06.18.
//

#ifndef NBMPI_DATASET_GENERATOR_HPP
#define NBMPI_DATASET_GENERATOR_HPP

#include <sstream>
#include <fstream>
#include <iomanip>
#include <mpi.h>
#include <map>
#include <unordered_map>
#include <zoltan.h>
#include <cstdlib>
#include <gsl/gsl_statistics.h>

#include "../astar.hpp"
#include "../ljpotential.hpp"
#include "../report.hpp"
#include "../physics.hpp"
#include "../nbody_io.hpp"
#include "../utils.hpp"
#include "../geometric_load_balancer.hpp"
#include "../params.hpp"
#include "../spatial_elements.hpp"
#include "../graph.hpp"
#include "../metrics.hpp"
#include "../zoltan_fn.hpp"

#define TICK_FREQ 1
#define WINDOW_SIZE 50
#define N_FEATURES 10
#define N_LABEL 1
#define DELTA_LB_CALL 100

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

    CommunicationDatatype datatype = elements::register_datatype<N>();
    std::vector<partitioning::geometric::Domain<N>> domain_boundaries(nproc);

    // get boundaries of all domains
    for (int part = 0; part < nproc; ++part) {
        Zoltan_RCB_Box(load_balancer, part, &dim, &xmin, &ymin, &zmin, &xmax, &ymax, &zmax);
        auto domain = partitioning::geometric::borders_to_domain<N>(xmin, ymin, zmin, xmax, ymax, zmax,
                                                                    params->simsize);
        domain_boundaries[part] = domain;
    }
    std::unordered_map<long long, std::unique_ptr<std::vector<elements::Element<N> > > > plklist;

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
        lj::create_cell_linkedlist(M, lsub, mesh_data->els, remote_el, plklist);
        float complexity = (float) lj::compute_forces(M, lsub, mesh_data->els, remote_el, plklist, params);

        leapfrog2(dt, mesh_data->els);
        leapfrog1(dt, mesh_data->els, 3.2*params->sig_lj);
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

    CommunicationDatatype datatype = elements::register_datatype<N>();
    std::vector<partitioning::geometric::Domain<N>> domain_boundaries(nproc);

    // get boundaries of all domains
    for (int part = 0; part < nproc; ++part) {
        Zoltan_RCB_Box(load_balancer, part, &dim, &xmin, &ymin, &zmin, &xmax, &ymax, &zmax);
        auto domain = partitioning::geometric::borders_to_domain<N>(xmin, ymin, zmin, xmax, ymax, zmax,
                                                                    params->simsize);
        domain_boundaries[part] = domain;
    }

    std::unordered_map<long long, std::unique_ptr<std::vector<elements::Element<N> > > > plklist;
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
            lj::create_cell_linkedlist(M, lsub, mesh_data->els, remote_el, plklist);
            float complexity = (float) lj::compute_forces(M, lsub, mesh_data->els, remote_el, plklist, params);
            leapfrog2(dt, mesh_data->els);
            leapfrog1(dt, mesh_data->els, 3.2 * params->sig_lj);
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

    CommunicationDatatype datatype = elements::register_datatype<N>();
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

            lj::create_cell_linkedlist(M, lsub, mesh_data->els, remote_el, plklist);
            float complexity = (float) lj::compute_forces(M, lsub, mesh_data->els, remote_el, plklist,
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

#endif //NBMPI_DATASET_GENERATOR_HPP
