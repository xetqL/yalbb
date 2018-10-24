//
// Created by xetql on 9/25/18.
//

#ifndef NBMPI_GENERATE_HPP
#define NBMPI_GENERATE_HPP

#include <queue>
#include <memory>

#include "initial_conditions.hpp"
#include "params.hpp"

template<int N>
std::queue<std::pair<std::shared_ptr<initial_condition::RandomElementsGenerator<N>>, int>>
init_generator(std::shared_ptr<initial_condition::lennard_jones::RejectionCondition<N>> condition,
                int init_conf, const sim_param_t* params, const int MAX_TRIAL = 1000000) {
    unsigned int NB_CLUSTERS;
    std::vector<int> clusters;
    using ElementGeneratorCfg = std::pair<std::shared_ptr<initial_condition::RandomElementsGenerator<N>>, int>;
    std::queue<ElementGeneratorCfg> gen;
    switch (init_conf) {
        case 1: //uniformly distributed
            gen.push(std::make_pair(
                    std::make_shared<initial_condition::lennard_jones::UniformRandomElementsGenerator<N>>(
                            params->seed, MAX_TRIAL), params->npart));
            break;
        case 2: //Half full half empty
            gen.push(std::make_pair(
                    std::make_shared<initial_condition::lennard_jones::HalfLoadedRandomElementsGenerator<N>>(
                            params->simsize / 2, false, params->seed, MAX_TRIAL), params->npart));
            break;
        case 3: //Wall of particle
            gen.push(std::make_pair(
                    std::make_shared<initial_condition::lennard_jones::ParticleWallElementsGenerator<N>>(
                            params->simsize / 2, false, params->seed, MAX_TRIAL), params->npart));
            break;
        case 4: //cluster
            NB_CLUSTERS = 1;
            clusters.resize(NB_CLUSTERS);
            std::fill(clusters.begin(), clusters.end(), params->npart);
            gen.push(std::make_pair(
                    std::make_shared<initial_condition::lennard_jones::RandomElementsInNClustersGenerator<N>>(
                            clusters, params->seed, MAX_TRIAL), params->npart));
            break;
        case 5: //custom various density
            NB_CLUSTERS = 2;
            clusters.resize(NB_CLUSTERS);
            std::fill(clusters.begin(), clusters.end(), params->npart / 4);
            gen.push(std::make_pair(
                    std::make_shared<initial_condition::lennard_jones::RandomElementsInNClustersGenerator<N>>(
                            clusters, params->seed, MAX_TRIAL), params->npart / 4));
            gen.push(std::make_pair(
                    std::make_shared<initial_condition::lennard_jones::HalfLoadedRandomElementsGenerator<N>>(
                            params->simsize / 10, false, params->seed, MAX_TRIAL), 3 * params->npart / 4));
            break;
        case 6: //custom various density
            NB_CLUSTERS = 1;
            clusters.resize(NB_CLUSTERS);
            std::fill(clusters.begin(), clusters.end(), params->npart);
            gen.push(std::make_pair(
                    std::make_shared<initial_condition::lennard_jones::RandomElementsInNClustersGenerator<N>>(
                            clusters, params->seed, MAX_TRIAL), params->npart));
            break;
        default:
            throw std::runtime_error("Unknown particle distribution.");
    }
    return gen;
}

#endif //NBMPI_GENERATE_HPP
