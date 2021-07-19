//
// Created by xetql on 5/22/20.
//

#ifndef NBMPI_PROBE_PROCESSOR_HPP
#define NBMPI_PROBE_PROCESSOR_HPP

#include "probe.hpp"

/**
 * This class processes what's inside a probe to compute iterations metrics and run metrics*/
struct ProbeProcessor {

    [[nodiscard]] Real compute_efficiency(const Probe* p) const ;
    [[nodiscard]] Time compute_average_load_balancing_time(const Probe* p) const ;
    [[nodiscard]] Real compute_load_imbalance(const Probe* p) const ;
    [[nodiscard]] Real compute_imbalance_time(const Probe* p) const ;
    [[nodiscard]] Real compute_average_parallel_load_balancing_efficiency(const Probe* p) const ;
    [[nodiscard]] Real compute_load_balancing_effort(const Probe* p) const ;

};


#endif //NBMPI_PROBE_PROCESSOR_HPP
