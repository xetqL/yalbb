//
// Created by xetql on 10/12/20.
//

#pragma once

#include <variant>
#include <numeric>

#include "probe.hpp"
#include "probe_processor.hpp"

namespace lb {
    struct Criterion {
        ProbeProcessor probeProcessor{};
        virtual bool operator()(Probe& probe) const = 0;
        virtual ~Criterion() = default;
    }; // To add a new criterion: extend this class;
    // Static Load Balancing
    struct Static : public Criterion {
        bool operator()(Probe& probe) const override;
    };

    // Dynamic Load Balancing - Automatic Balancing
    struct VanillaMenon :       public Criterion {
        mutable double cumulative_imbalance = 0.0;
        bool operator()(Probe& probe) const override;
    };
    struct OfflineMenon :       public Criterion {
        mutable int prev_lb  = 0;
        mutable unsigned tau = -1;
        mutable std::vector<Time> iteration_times_since_lb {};
        mutable std::vector<Time> average_times_since_lb   {};
        mutable std::vector<Time> diff {};
        bool operator()(Probe& probe) const override;
    };
    struct ImprovedMenon :      public Criterion {
        mutable double baseline = 0.0;
        mutable double cumulative_imbalance = 0.0;

        bool operator()(Probe& probe) const override;
    };
    struct ImprovedMenonNoMax : public Criterion {
        mutable double baseline = 0.0;
        mutable double cumulative_imbalance = 0.0;

        bool operator()(Probe& probe) const override;
    };
    struct ZhaiMenon :          public Criterion {
        mutable double cumulative_imbalance = 0.0;
        mutable std::vector<Time> iteration_times_since_lb {};
        bool operator()(Probe& probe) const override;
    };
    struct BastienMenon :       public Criterion {
        mutable double cumulative_imbalance = 0.0;
        mutable int tau = 1;
        mutable double baseline = 0.0;

        bool operator()(Probe &probe) const override;
    };

    // DLB - User-defined
    struct Periodic : public Criterion {
        const unsigned N{};
        explicit Periodic(unsigned N) : N(N) {}
        bool operator()(Probe& probe) const override;
    };

    struct Procassini :         public Criterion {
        const Real speedup_factor{};
        explicit Procassini(Real speedup_factor) : speedup_factor(speedup_factor) {}
        bool operator()(Probe& probe) const override;
    };
    struct Marquez :            public Criterion {
        const Real threshold;
        explicit Marquez(Real threshold) : threshold(threshold) {}
        bool operator()(Probe& probe) const override;
    };
    struct Reproduce :          public Criterion {
        const std::vector<int>& scenario;
        explicit Reproduce(const std::vector<int>& scenario) : scenario(scenario){}
        bool operator()(Probe& probe) const override;
    };

}
