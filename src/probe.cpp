//
// Created by xetql on 4/29/20.
//


#include <numeric>
#include "probe.hpp"

Probe::Probe(int nproc) : nproc(nproc) {}

void Probe::update_cumulative_imbalance_time() {
    /**The cumulative load imbalance since a load balancing.
     * I don't think that we have to take into account the remaining imbalance after a load balancing because
     * re-using the LB algorithm won't remove it. For instance, let pretends that the load balancing cost is
     * C=0.1s and the remaining imbalance (max-avg) >= C, then the criterion is matched at every iteration, alas,
     * we know that the algorithm is not capable to do better. Hence, we can treat C as a baseline.*/
    if(this->balanced) lb_imbalance_baseline = (max_it - (sum_it/nproc)); // use the remaining imbalance as baseline
    cumulative_imbalance_time += std::max(0.0, ((max_it - (sum_it/nproc)) - lb_imbalance_baseline));
    vanilla_cumulative_imbalance_time += (max_it - (sum_it/nproc));
}

void  Probe::reset_cumulative_imbalance_time() {
    cumulative_imbalance_time = 0.0;
    vanilla_cumulative_imbalance_time = 0.0;
}
Time  Probe::compute_avg_lb_time() {
    if(lb_times.empty()) return 0.0;
    const auto N = lb_times.size();
    const auto window_size = std::min((decltype(N)) 5, N);
    return std::accumulate(lb_times.cbegin(), std::next(lb_times.cbegin(), window_size), 0.0) / N; }

Time* Probe::max_it_time() { return &max_it; }
Time* Probe::min_it_time() { return &min_it; }

void Probe::set_balanced(bool lb_status){
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
Time Probe::get_avg_it() {
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

Time Probe::get_batch_time() {
    return batch_time;
}

Time Probe::get_cumulative_imbalance_time() const {
    return cumulative_imbalance_time;
}
Time Probe::get_vanilla_cumulative_imbalance_time() const {
    return vanilla_cumulative_imbalance_time;
}


Time* Probe::sum_it_time() { return &sum_it; }
void  Probe::push_load_balancing_time(Time lb_time){ lb_times.push_back(lb_time); }
void  Probe::push_load_balancing_parallel_efficiency(Real lb_parallel_efficiency){ lb_parallel_efficiencies.push_back(lb_parallel_efficiency); }
void  Probe::update_lb_parallel_efficiencies() { lb_parallel_efficiencies.push_back(get_avg_it() / get_max_it()); }

Real Probe::compute_avg_lb_parallel_efficiency() {return std::accumulate(lb_parallel_efficiencies.cbegin(), lb_parallel_efficiencies.cend(), 0.0) / lb_parallel_efficiencies.size();}
Real Probe::get_current_parallel_efficiency(){ return lb_parallel_efficiencies.back();}
void Probe::next_iteration() {current_iteration++;}
void Probe::start_batch(Index batch) { current_batch = batch; }
void Probe::end_batch(Time time) { batch_time = time; }
bool Probe::is_batch_started() { return batch_started; }
std::string Probe::lb_cost_to_string(){
    std::stringstream str;
    str << lb_times;
    return str.str();
}