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

class PolicyRunner {
    lb::Criterion* criterion;
    Probe* probe;
public:
    PolicyRunner(Probe* probe, lb::Criterion* criterion) : probe(probe), criterion(criterion) {}
    bool should_load_balance() {
        return criterion->operator()(*probe);
    };
};

#endif //NBMPI_POLICY_HPP
