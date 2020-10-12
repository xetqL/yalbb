//
// Created by xetql on 10/12/20.
//

#pragma once

#include <variant>
#include "probe.hpp"

namespace lb {
    struct Static{
        bool operator()(Probe& probe) const{
            return false;
        }
    };
    struct Periodic {
        const unsigned N;
        bool operator()(Probe& probe) const{
            return ((probe.get_current_iteration() + 1) % N) == 0;
        }
    };
    struct VanillaMenon {
        bool operator()(Probe& probe) const{
            return (probe.get_vanilla_cumulative_imbalance_time() >= probe.compute_avg_lb_time());
        }
    };
    using  WhenCumulativeImbalanceIsGtLbCost = VanillaMenon;
    struct ImprovedMenon {
        bool operator()(Probe& probe) const {
            return (probe.get_cumulative_imbalance_time() >= probe.compute_avg_lb_time());
        }
    };
    using  WhenCumulativeImbalanceAboveBaselineIsGtLbCost = ImprovedMenon;
    struct Procassini {
        const Real speedup_factor;
        bool operator()(Probe& probe) const{
            Real epsilon_c = probe.get_efficiency();
            Real epsilon_lb= probe.compute_avg_lb_parallel_efficiency(); //estimation based on previous lb call
            Real S         = epsilon_c / epsilon_lb;
            Real tau_prime = probe.get_batch_time() *  S + probe.compute_avg_lb_time(); //estimation of next iteration time based on speed up + LB cost
            Real tau       = probe.get_batch_time();
            return (tau_prime < speedup_factor*tau);
        }
    };
    using  WhenTimeDecreasedBy = Procassini;
    struct Marquez {
        const Real threshold;
        bool operator()(Probe& probe) const {
            Real tolerance      = probe.get_avg_it() * threshold;
            Real tolerance_plus = probe.get_avg_it() + tolerance;
            Real tolerance_minus= probe.get_avg_it() - tolerance;
            return (probe.get_min_it() < tolerance_minus || tolerance_plus < probe.get_max_it());
        }
    };
    using  WhenAtLeastOneProcIsOutsideToleranceRange = Marquez;
    using  Criterion = std::variant<Static, Periodic, VanillaMenon, ImprovedMenon, Procassini, Marquez>;
}
