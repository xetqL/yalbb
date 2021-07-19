//
// Created by xetql on 5/22/20.
//

#include "probe_processor.hpp"

Real ProbeProcessor::compute_efficiency(const Probe* probe) const {
    return probe->get_avg_it() / probe->get_max_it();
}

Time ProbeProcessor::compute_average_load_balancing_time(const Probe* probe) const {
    if (probe->lb_times.empty()) return 0.0;
    return probe->lb_times.back();
}

Real ProbeProcessor::compute_load_imbalance(const Probe* probe) const {
    return (probe->get_max_it()/probe->get_avg_it() - 1.0);
}
Real ProbeProcessor::compute_imbalance_time(const Probe* probe) const {
    return (probe->get_max_it() - probe->get_avg_it());
}
Real ProbeProcessor::compute_average_parallel_load_balancing_efficiency(const Probe* probe) const {
    return std::accumulate(probe->lb_parallel_efficiencies.cbegin(),
                           probe->lb_parallel_efficiencies.cend(), 0.0) /
                           static_cast<double>(probe->lb_parallel_efficiencies.size());

}

Real ProbeProcessor::compute_load_balancing_effort(const Probe* probe) const {
    auto tau = static_cast<double>(probe->current_iteration - probe->previous_lb_it);
    return probe->lb_interval_time / tau;
}
