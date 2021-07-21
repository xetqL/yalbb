//
// Created by xetql on 10/22/20.
//

#include "criterion.hpp"

bool lb::Static::operator()(Probe &probe) const {
    return false;
}

bool lb::Periodic::operator()(Probe &probe) const {
    return ((probe.get_current_iteration() + 1) % N) == 0;
}

bool lb::VanillaMenon::operator()(Probe &probe) const  {
    cumulative_imbalance += (probe.get_max_it() - (probe.get_sum_it()/probe.nproc));
    const auto decision = (cumulative_imbalance >= probeProcessor.compute_average_load_balancing_time(&probe));
    if(decision) cumulative_imbalance = 0.0;
    return decision;
}

bool lb::OfflineMenon::operator()(Probe &probe) const {

    if(probe.get_current_iteration() > 0) {
        iteration_times_since_lb.push_back(probe.get_max_it());
        average_times_since_lb.push_back(probe.get_avg_it());
    }

    const auto N = iteration_times_since_lb.size();
    // Compute tau based on history
    diff.resize(N);
    // compute average dmax/dt
    std::adjacent_difference(iteration_times_since_lb.cbegin(), iteration_times_since_lb.cend(), diff.begin());
    const Time dmax = std::accumulate(diff.cbegin(), diff.cend(), 0.0) / N;
    // compute average davg/dt
    std::adjacent_difference(average_times_since_lb.cbegin(), average_times_since_lb.cend(), diff.begin());
    const Time davg = std::accumulate(diff.cbegin(), diff.cend(), 0.0) / N;
    // compute difference between max and avg
    const Time m = dmax - davg;
    // compute average load balancing cost
    const Time C = probeProcessor.compute_average_load_balancing_time(&probe);
    tau = m > 0 ? static_cast<unsigned>( std::sqrt(2*C / m) ) : -1;
    // check if we are after or before the load balancing iteration
    const auto decision = (prev_lb + tau) <= probe.get_current_iteration();
    // reset data
    if(decision) {
        prev_lb = probe.get_current_iteration();
        diff.clear();
        iteration_times_since_lb.clear();
        average_times_since_lb.clear();
    }

    return decision;
}

bool lb::ImprovedMenon::operator()(Probe &probe) const  {
    if(probe.balanced) baseline = (probe.get_max_it() - (probe.get_sum_it()/probe.nproc));
    cumulative_imbalance += std::max(0.0, (probe.get_max_it() - (probe.get_sum_it()/probe.nproc)) - baseline);
    const auto decision = (cumulative_imbalance >= probeProcessor.compute_average_load_balancing_time(&probe));
    if(decision) cumulative_imbalance = 0.0;
    return decision;
}

bool lb::ImprovedMenonNoMax::operator()(Probe &probe) const  {
    if(probe.balanced) baseline = (probe.get_max_it() - (probe.get_sum_it()/probe.nproc));
    cumulative_imbalance += (probe.get_max_it() - (probe.get_sum_it()/probe.nproc)) - baseline;
    const auto decision = (cumulative_imbalance >= probeProcessor.compute_average_load_balancing_time(&probe));
    if(decision) cumulative_imbalance = 0.0;
    return decision;
}

bool lb::ZhaiMenon::operator()(Probe &probe) const  {
    if(probe.get_current_iteration() > 0) iteration_times_since_lb.push_back(probe.get_max_it());
    const auto N = probe.iteration_times_since_lb.size();
    cumulative_imbalance += math::median(iteration_times_since_lb.end() - std::min(N, 3ul), iteration_times_since_lb.end()) - math::mean(iteration_times_since_lb.begin(), iteration_times_since_lb.end());
    const auto decision = (cumulative_imbalance >= probeProcessor.compute_average_load_balancing_time(&probe));
    if(decision) {
        cumulative_imbalance = 0.0;
        iteration_times_since_lb.clear();
    }
    return decision;
}

bool lb::BastienMenon::operator()(Probe &probe) const {
    if(probe.balanced) baseline = (probe.get_max_it() - (probe.get_sum_it()/probe.nproc));
    const double Ui = (probe.get_max_it() - (probe.get_sum_it()/probe.nproc));
    cumulative_imbalance += Ui;
    const auto decision = ((tau*Ui - cumulative_imbalance) >= probeProcessor.compute_average_load_balancing_time(&probe));
    if(decision) {
        cumulative_imbalance = 0.0;
        tau = 1;
    } else {
        tau++;
    }
    return decision;
}

bool lb::Procassini::operator()(Probe &probe) const {
    Real epsilon_c = probeProcessor.compute_efficiency(&probe);
    Real epsilon_lb= probeProcessor.compute_average_parallel_load_balancing_efficiency(&probe); //estimation based on previous lb call
    Real S         = epsilon_c / epsilon_lb;
    Real tau_prime = probe.get_batch_time() *  S + probeProcessor.compute_average_load_balancing_time(&probe); //estimation of next iteration time based on speed up + LB cost
    Real tau       = probe.get_batch_time();
    return (tau_prime < speedup_factor*tau);
}

bool lb::Marquez::operator()(Probe &probe) const {
    Real tolerance      = probe.get_avg_it() * threshold;
    Real tolerance_plus = probe.get_avg_it() + tolerance;
    Real tolerance_minus= probe.get_avg_it() - tolerance;
    return (probe.get_min_it() < tolerance_minus || tolerance_plus < probe.get_max_it());
}

bool lb::Reproduce::operator()(Probe &probe) const {
    return (bool) scenario.at(probe.get_current_iteration());
}
