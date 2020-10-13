//
// Created by xetql on 4/29/20.
//

#ifndef NBMPI_PROBE_HPP
#define NBMPI_PROBE_HPP

#include "utils.hpp"
#include "parallel_utils.hpp"

struct Probe {
    int current_iteration = 0;
    Time max_it = 0, min_it = 0, sum_it = 0, cumulative_imbalance_time = 0, vanilla_cumulative_imbalance_time, lb_imbalance_baseline = 0, batch_time = 0;
    std::vector<Time> lb_times;
    std::vector<Real> lb_parallel_efficiencies;

    bool balanced = true;
    bool batch_started = false;
    int i = 0, nproc, current_batch = 0;

    unsigned int batch_id = 0;
    Probe(int nproc);
    Time get_cumulative_imbalance_time() const;
    Time get_vanilla_cumulative_imbalance_time() const;
    Time get_batch_time() const;
    Time get_avg_it() const;
    Time get_max_it() const;
    Time get_min_it() const;
    Time get_sum_it() const;
    int  get_current_iteration() const;
    Real get_current_parallel_efficiency() const;
    Real get_efficiency() const;
    bool is_balanced() const;
    bool is_batch_started();
    void set_balanced(bool lb_status);

    void  update_cumulative_imbalance_time();
    void  reset_cumulative_imbalance_time();
    Time  compute_avg_lb_time() const;
    Time  compute_load_imbalance() const { return (get_max_it()/get_avg_it() - 1.0); }
    Real  compute_avg_lb_parallel_efficiency();
    Time* max_it_time() ;
    Time* min_it_time() ;
    Time* sum_it_time();
    void  sync_it_time_across_processors(Time *t, MPI_Comm comm) {
        // Measure load imbalance
        MPI_Allreduce(t, this->max_it_time(), 1, MPI_TIME, MPI_MAX, comm);
        MPI_Allreduce(t, this->min_it_time(), 1, MPI_TIME, MPI_MIN, comm);
        MPI_Allreduce(t, this->sum_it_time(), 1, MPI_TIME, MPI_SUM, comm);
        *t = *this->max_it_time();
    }

    void  push_load_balancing_time(Time lb_time);
    void  push_load_balancing_parallel_efficiency(Real lb_parallel_efficiency);
    void  update_lb_parallel_efficiencies();
    void  next_iteration();
    void  start_batch(Index batch);
    void  end_batch(Time time);
    std::string lb_cost_to_string();
};


#endif //NBMPI_PROBE_HPP
