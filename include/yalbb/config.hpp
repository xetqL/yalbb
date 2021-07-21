//
// Created by xetql on 7/21/21.
//

#ifndef YALBB_CONFIG_HPP
#define YALBB_CONFIG_HPP
#include <tuple>
#include <vector>
#include "params.hpp"
#include "criterion.hpp"
#include "step_producer.hpp"

using Config = std::tuple<std::string, std::string, sim_param_t, lb::Criterion*>;
void load_default_configs(std::vector<Config>& configs, const sim_param_t& params) {

    configs.emplace_back("BBCriterion",  "BBCriterion",      params, new lb::BastienMenon);

    configs.emplace_back("Static",       "Static",           params, new lb::Static);

    // Automatic criterion
    configs.emplace_back("VanillaMenon", "VMenon",           params, new lb::VanillaMenon);
    configs.emplace_back("OfflineMenon", "OMenon",           params, new lb::OfflineMenon);
    configs.emplace_back("PositivMenon", "PMenon",           params, new lb::ImprovedMenonNoMax);
    configs.emplace_back("ZhaiMenon",    "ZMenon",           params, new lb::ZhaiMenon);
    configs.emplace_back("BBCriterion",  "BBCriterion",      params, new lb::BastienMenon);

    // Periodic
    configs.emplace_back("Periodic 1",       "Periodic_1",    params, new lb::Periodic(1));
    for(StepProducer<unsigned> producer({{25, 4}, {50, 10}, {100, 4}}); !producer.finished();){
        unsigned step = producer.next();
        configs.emplace_back(fmt("Periodic %d", step), fmt("Periodic_%d", step), params, new lb::Periodic(step));
    }
    // Procassini
    for(StepProducer<unsigned> producer({{25, 6},{50, 9}, {100, 5}, {200, 2}}); !producer.finished();){
        unsigned step = producer.next();
        configs.emplace_back(fmt("Procassini %d", step), fmt("Procassini_%d", step), params, new lb::Procassini(step / 100.0f));
    }
    // Marquez
    for(StepProducer<unsigned> producer({{500, 1},{100, 5}, {125, 4}, {250, 2},{500, 2}, {1000, 1}}); !producer.finished();){
        unsigned step = producer.next();
        configs.emplace_back(fmt("Marquez %d", step), fmt("Marquez_%d",step), params, new lb::Marquez{step / 100.0f});
    }
}

#endif //YALBB_CONFIG_HPP
