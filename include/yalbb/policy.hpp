//
// Created by xetql on 02.07.18.
//

#ifndef NBMPI_POLICY_HPP
#define NBMPI_POLICY_HPP


#include <random>
#include <queue>
#include <variant>
#include "utils.hpp"
#include "probe.hpp"
#include "criterion.hpp"

template<class P>
class LBPolicy {
public:
    virtual bool should_load_balance() = 0;
};

class PolicyRunner {
    lb::Criterion criterion;
    Probe* probe;
public:
    PolicyRunner(Probe* probe, lb::Criterion criterion) : probe(probe), criterion(std::move(criterion)) {}
    bool should_load_balance() {
        return std::visit([probePtr=probe](const auto& v){return v(*probePtr); }, criterion);
    };
};

template<class Policy>
class PolicyExecutor : public LBPolicy<Policy>{
    Probe* probe;
    Policy p;
public:
    PolicyExecutor(Probe* probe, Policy p) : probe(probe), p(p) {}
    bool should_load_balance() {
        return p(*probe);
    };
};



template<class Policy>
class CustomPolicy {
    Probe* dataHolder;
    Policy p;
public:
    CustomPolicy(Probe* dataHolder, Policy p) : dataHolder(dataHolder), p(p) {};
    bool apply() {
        return p(*dataHolder);
    }
};
#endif //NBMPI_POLICY_HPP
