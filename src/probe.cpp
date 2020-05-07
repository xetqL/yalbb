//
// Created by xetql on 4/29/20.
//

#include "probe.hpp"
#include <numeric>

Probe::Probe(int nproc) : nproc(nproc) {}

void  Probe::update_cumulative_imbalance_time() { cumulative_imbalance_time += max_it - sum_it/nproc; }
void  Probe::reset_cumulative_imbalance_time() { cumulative_imbalance_time = 0.0; }
Time  Probe::compute_avg_lb_time() { return lb_times.size() == 0 ? 0.0 : std::accumulate(lb_times.cbegin(), lb_times.cend(), 0.0) / lb_times.size(); }
Time* Probe::max_it_time() { return &max_it; }
Time* Probe::min_it_time() { return &min_it; }

void Probe::set_balanced(bool lb_status) {
    Probe::balanced = lb_status;
}
Real Probe::get_efficiency() {
    return get_avg_it() / get_max_it();
}
bool Probe::is_balanced() const {
    return balanced;
}

int Probe::get_current_iteration() const {
    return current_iteration;
}
Time  Probe::get_avg_it() {
    return sum_it/nproc;
}

Time Probe::get_max_it() const {
    return max_it;
}

Time Probe::get_min_it() const {
    return min_it;
}

Time Probe::get_sum_it() const {
    return sum_it;
}

Time Probe::get_cumulative_imbalance_time() const {
    return cumulative_imbalance_time;
}

Time* Probe::sum_it_time() { return &sum_it; }
void  Probe::push_load_balancing_time(Time lb_time){ lb_times.push_back(lb_time); }
void  Probe::push_load_balancing_parallel_efficiency(Real lb_parallel_efficiency){ lb_parallel_efficiencies.push_back(lb_parallel_efficiency); }
void  Probe::update_lb_parallel_efficiencies() { lb_parallel_efficiencies.push_back(get_avg_it() / get_max_it());}

Real Probe::compute_avg_lb_parallel_efficiency() {return std::accumulate(lb_parallel_efficiencies.cbegin(), lb_parallel_efficiencies.cend(), 0.0) / lb_parallel_efficiencies.size();}
void Probe::next_iteration() {current_iteration++;}
void Probe::start_batch(Index frame) { batch_started=true; }
void Probe::end_batch() { batch_started=false; batch_id++; }
bool Probe::is_batch_started() { return batch_started; }
std::string Probe::lb_cost_to_string(){
    std::stringstream str;
    str << lb_times;
    return str.str();
}