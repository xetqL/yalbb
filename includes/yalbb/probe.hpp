//
// Created by xetql on 4/29/20.
//

#ifndef NBMPI_PROBE_HPP
#define NBMPI_PROBE_HPP

#include "utils.hpp"

class Probe {
    int current_iteration = 0;
    Time max_it = 0, min_it = 0, sum_it = 0, cumulative_imbalance_time = 0, lb_imbalance_baseline = 0, batch_time = 0;
    std::vector<Time> lb_times;
    std::vector<Real> lb_parallel_efficiencies;

    bool balanced = true;
    bool batch_started = false;
    int i = 0, nproc, current_batch = 0;
public:
    void new_batch();
    unsigned int batch_id = 0;
    Probe(int nproc);
    void  update_cumulative_imbalance_time();
    void   reset_cumulative_imbalance_time();
    Time  compute_avg_lb_time();
    Time* max_it_time() ;
    Time* min_it_time() ;
    void set_balanced(bool lb_status);
    Real get_efficiency();
    bool is_balanced() const ;
    int get_current_iteration() const;
    Time  get_avg_it();
    Time get_max_it() const;
    Time get_min_it() const;
    Time get_sum_it() const;

    Time  get_cumulative_imbalance_time() const;
    Time compute_load_imbalance() { return (get_max_it()/get_avg_it() - 1.0); }
    Time* sum_it_time();
    void  push_load_balancing_time(Time lb_time);
    void  push_load_balancing_parallel_efficiency(Real lb_parallel_efficiency);
    void update_lb_parallel_efficiencies();

    Real compute_avg_lb_parallel_efficiency();
    Real get_current_parallel_efficiency();

    void next_iteration();

    void start_batch(Index batch);
    void end_batch(Time time);

    bool is_batch_started();

    std::string lb_cost_to_string();
};


#endif //NBMPI_PROBE_HPP
