//
// Created by xetql on 10/12/20.
//

#pragma once

#include <variant>
#include <numeric>

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
        mutable double cumulative_imbalance = 0.0;
        bool operator()(Probe& probe) const {
            const auto N = probe.iteration_times_since_lb.size();
            cumulative_imbalance += (probe.get_max_it() - (probe.get_sum_it()/probe.nproc));
            const auto decision = (cumulative_imbalance >= probe.compute_avg_lb_time());
            if(decision) cumulative_imbalance = 0.0;
            return decision;
        }
    };
/**
 * Compute \tau after each load balancing based on average C and average workload increase rate
 */
    struct OfflineMenon {
        mutable int prev_lb  = 0;
        mutable unsigned tau = -1;
        mutable std::vector<Time> iteration_times_since_lb {};
        mutable std::vector<Time> average_times_since_lb {};
        mutable std::vector<Time> diff {};

        bool operator()(Probe& probe) const {

            if(probe.get_current_iteration() > 0) {
                iteration_times_since_lb.push_back(probe.get_max_it());
                average_times_since_lb.push_back(probe.get_avg_it());
            }

            const auto N = iteration_times_since_lb.size();

            const auto decision = (prev_lb + tau) <= probe.get_current_iteration();

            // Compute tau
            diff.resize(N);
            std::adjacent_difference(iteration_times_since_lb.cbegin(), iteration_times_since_lb.cend(), diff.begin());
            const Time dmax = std::accumulate(diff.cbegin(), diff.cend(), 0.0) / N;
            std::adjacent_difference(average_times_since_lb.cbegin(), average_times_since_lb.cend(), diff.begin());
            const Time davg = std::accumulate(diff.cbegin(), diff.cend(), 0.0) / N;
            const Time m = dmax - davg;
            const Time C = probe.compute_avg_lb_time();
            tau = m > 0 ? static_cast<unsigned>( std::sqrt(2*C / m) ) : -1;

            if(decision) {
                prev_lb = probe.get_current_iteration();
                diff.clear();
                iteration_times_since_lb.clear();
                average_times_since_lb.clear();
            }

            return decision;
        }
    };
    using  WhenCumulativeImbalanceIsGtLbCost = VanillaMenon;

    struct ImprovedMenon {
        mutable double baseline = 0.0;
        mutable double cumulative_imbalance = 0.0;
        bool operator()(Probe& probe) const {
            if(probe.balanced) baseline = (probe.get_max_it() - (probe.get_sum_it()/probe.nproc));
            cumulative_imbalance += std::max(0.0, (probe.get_max_it() - (probe.get_sum_it()/probe.nproc)) - baseline);
            const auto decision = (cumulative_imbalance >= probe.compute_avg_lb_time());
            if(decision) cumulative_imbalance = 0.0;
            return decision;
        }
    };
    struct ImprovedMenonNoMax {
        mutable double baseline = 0.0;
        mutable double cumulative_imbalance = 0.0;
        bool operator()(Probe& probe) const {
            if(probe.balanced) baseline = (probe.get_max_it() - (probe.get_sum_it()/probe.nproc));
            cumulative_imbalance += (probe.get_max_it() - (probe.get_sum_it()/probe.nproc)) - baseline;
            const auto decision = (cumulative_imbalance >= probe.compute_avg_lb_time());
            if(decision) cumulative_imbalance = 0.0;
            return decision;
        }
    };

    using  WhenCumulativeImbalanceAboveBaselineIsGtLbCost = ImprovedMenon;

    struct ZhaiMenon {
        mutable double cumulative_imbalance = 0.0;
        mutable std::vector<Time> iteration_times_since_lb {};
        bool operator()(Probe& probe) const {
            if(probe.get_current_iteration() > 0) iteration_times_since_lb.push_back(probe.get_max_it());
            const auto N = probe.iteration_times_since_lb.size();
            cumulative_imbalance += math::median(iteration_times_since_lb.end() - std::min(N, 3ul), iteration_times_since_lb.end()) - math::mean(iteration_times_since_lb.begin(), iteration_times_since_lb.end());
            const auto decision = (cumulative_imbalance >= probe.compute_avg_lb_time());
            if(decision) {
                cumulative_imbalance = 0.0;
                iteration_times_since_lb.clear();
            }
            return decision;
        }
    };

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
    using  WhenTimeDecreasesBy = Procassini;
    struct Marquez {        const Real threshold;
        bool operator()(Probe& probe) const {
            Real tolerance      = probe.get_avg_it() * threshold;
            Real tolerance_plus = probe.get_avg_it() + tolerance;
            Real tolerance_minus= probe.get_avg_it() - tolerance;
            return (probe.get_min_it() < tolerance_minus || tolerance_plus < probe.get_max_it());
        }
    };
    using  WhenAtLeastOneProcIsOutsideToleranceRange = Marquez;

    struct Reproduce {
        const std::vector<int> scenario;
        bool operator()(Probe& probe) const {
            return (bool) scenario.at(probe.get_current_iteration());
        }
    };

    using  Criterion = std::variant<
            Static,
            Reproduce,
            Periodic,
            VanillaMenon,
            OfflineMenon,
            ImprovedMenon,
            ImprovedMenonNoMax,
            ZhaiMenon,
            Procassini,
            Marquez>;
}
