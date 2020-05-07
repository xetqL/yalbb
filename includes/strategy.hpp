//
// Created by xetql on 02.07.18.
//

#ifndef NBMPI_STRATEGY_HPP
#define NBMPI_STRATEGY_HPP


#include <random>
#include <queue>
#include "utils.hpp"
#include "probe.hpp"
template<class P>
class LBPolicy {
public:
    virtual bool should_load_balance() = 0;
};
template<class Policy>
class PolicyRunner : public LBPolicy<Policy> {
    std::unique_ptr<Policy> p;
public:
    template<class... Args> PolicyRunner(Args... args) : p(std::make_unique<Policy>(args...)) {}
    bool should_load_balance() { return p->apply(); };
};
template<class Policy>
class PolicyExecutor : public LBPolicy<Policy>{
    Probe* probe;
    Policy p;
public:
    PolicyExecutor(Probe* probe, Policy p) : probe(probe), p(p) {}
    bool should_load_balance() { return p(*probe); };
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
#endif //NBMPI_STRATEGY_HPP
