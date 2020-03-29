//
// Created by xetql on 08.06.18.
//

#ifndef NBMPI_INITIAL_CONDITIONS_HPP
#define NBMPI_INITIAL_CONDITIONS_HPP

#include "ljpotential.hpp"
#include "spatial_elements.hpp"
#include "utils.hpp"

#include <memory>

namespace initial_condition {

static std::random_device __rd;
static std::mt19937 __gen(__rd()); //Standard mersenne_twister_engine seeded with rd()

template<class Candidate>
class RejectionCondition {
public:
    virtual const bool predicate(const Candidate& c) const = 0;
};

namespace lj {

template<int N>
class RejectionCondition : public initial_condition::RejectionCondition<elements::Element<N>> {
    const std::vector<elements::Element<N>>* others;
public:
    const Real sig;
    const Real min_r2;
    const Real T0;
    const Real xmin;
    const Real ymin;
    const Real zmin;
    const Real xmax;
    const Real ymax;
    const Real zmax;
    const sim_param_t* params;

    RejectionCondition(const std::vector<elements::Element<N>>* others,
                       const Real sig,
                       const Real min_r2,
                       const Real T0,
                       const Real xmin,
                       const Real ymin,
                       const Real zmin,
                       const Real xmax,
                       const Real ymax,
                       const Real zmax,
                       const sim_param_t* params) :
            others(others),
            sig(sig), min_r2(min_r2), T0(T0),
            xmin(xmin), ymin(ymin), zmin(zmin),
            xmax(xmax), ymax(ymax), zmax(zmax), params(params) {}

    const bool predicate(const elements::Element<N>& c) const override {
        if (N > 2) {
            return std::all_of(others->cbegin(), others->cend(), [&](auto o) {
                return xmin < c.position.at(0) && c.position.at(0) < xmax &&
                       ymin < c.position.at(1) && c.position.at(1) < ymax &&
                       zmin < c.position.at(2) && c.position.at(2) < zmax &&
                       elements::distance2<N>(c, o) >= min_r2;
            });
        }else
            return std::all_of(others->cbegin(), others->cend(), [&](auto o) {
                return xmin < c.position.at(0) && c.position.at(0) < xmax &&
                       ymin < c.position.at(1) && c.position.at(1) < ymax &&
                       elements::distance2<N>(c, o) >= min_r2;
            });
    }
};

template<int N>
class NoRejectionCondition : public RejectionCondition<N> {
public:
    NoRejectionCondition( const std::vector<elements::Element<N>>* others,
                          const Real sig,
                          const Real min_r2,
                          const Real T0,
                          const Real xmin,
                          const Real ymin,
                          const Real zmin,
                          const Real xmax,
                          const Real ymax,
                          const Real zmax) :
            RejectionCondition<N>(others, sig, min_r2, T0, xmin, ymin, zmin, xmax, ymax, zmax) {}

    const bool predicate(const elements::Element<N>& c) const override {
        return true;
    }
};

} // end of namespace lennard_jones

template<int N>
class RandomElementsGenerator {
public:
    virtual void generate_elements(std::vector<elements::Element<N>>& elements, const int n,
                                   const std::shared_ptr<lj::RejectionCondition<N>> cond) = 0;
};

namespace lj {

template<int N>
class RandomElementsInClustersGenerator : public RandomElementsGenerator<N> {
    const int max_particles_per_cluster, seed, max_trial;
public:
    int number_of_clusters_generated = 0;

    RandomElementsInClustersGenerator(const int max_particles_per_cluster, const int seed = __rd(), const int max_trial = 10000) :
            max_particles_per_cluster(max_particles_per_cluster), seed(seed), max_trial(max_trial) {}

    void generate_elements(std::vector<elements::Element<N>>& elements, const int n,
                           const std::shared_ptr<lj::RejectionCondition<N>> condition) override {
        int number_of_element_generated = 0;
        int clusters_to_generate = 1;
        int cluster_id = 0;
        std::normal_distribution<Real> temp_dist(0.0, condition->T0 * condition->T0);
        std::uniform_real_distribution<Real> udistx(condition->xmin, condition->xmax),
                udisty(condition->ymin, condition->ymax),
                udistz(condition->zmin, condition->zmax);
        std::mt19937 my_gen(seed);
        while(cluster_id < clusters_to_generate && elements.size() < n) {
            Real cluster_centerx = udistx(my_gen),
                    cluster_centery = udisty(my_gen),
                    cluster_centerz = udistz(my_gen);
            statistic::NormalSphericalDistribution<N, Real>
                    sphere_dist_position(condition->sig * (max_particles_per_cluster), cluster_centerx, cluster_centery, cluster_centerz);
            statistic::NormalSphericalDistribution<N, Real>
                    sphere_dist_velocity(2.0 * condition->T0 * condition->T0, 0, 0, 0);
            auto cluster_velocity = sphere_dist_velocity(my_gen);
            int trial = 0;
            int part_in_cluster = 0;
            while(trial < max_trial && part_in_cluster < max_particles_per_cluster && elements.size() < n) { // stop when you cant generate new particles with less than 10000 trials within a cluster
                auto element = elements::Element<N>(sphere_dist_position(my_gen), sphere_dist_velocity(my_gen), elements.size(), elements.size());
                if(condition->predicate(element)) {
                    trial = 0;
                    element.velocity = cluster_velocity;
                    elements.push_back(element);
                    number_of_element_generated++;
                    part_in_cluster++;
                } else
                    trial++;
            }
            cluster_id++;
            if(cluster_id == clusters_to_generate && elements.size() < n) clusters_to_generate++;
        }
        number_of_clusters_generated = cluster_id;
    }
};

template<int N>
class RandomElementsInNClustersGenerator : public RandomElementsGenerator<N> {
    std::vector<int> clusters;
    const int seed, max_trial;
public:
    RandomElementsInNClustersGenerator(std::vector<int> clusters, const int seed = __rd(), const int max_trial = 10000) :
            clusters(clusters), seed(seed), max_trial(max_trial) {}

    void generate_elements(std::vector<elements::Element<N>>& elements, const int n,
                           const std::shared_ptr<lj::RejectionCondition<N>> condition) override {
        Real x_sz = condition->xmax - condition->xmin;
        Real y_sz = condition->ymax - condition->ymin;
        Real z_sz = condition->zmax - condition->zmin;
        const int clusters_to_generate = clusters.size();

        int number_of_element_generated = 0;
        int cluster_id = 0;
        std::uniform_real_distribution<Real>
                udistx(condition->xmin+x_sz*0.05, condition->xmax-x_sz*0.05),
                udisty(condition->ymin+y_sz*0.05, condition->ymax-y_sz*0.05),
                udistz(condition->zmin+z_sz*0.05, condition->zmax-z_sz*0.05);
        std::mt19937 my_gen(seed);
        std::vector<int> K = clusters;
        int part_in_cluster = 0;
        Real cluster_centerx = udistx(my_gen),
                cluster_centery = udisty(my_gen),
                cluster_centerz = udistz(my_gen);

        statistic::NormalSphericalDistribution<N, Real>
                sphere_dist_velocity(2.0 * condition->T0 * condition->T0, 0, 0, 0);

        std::array<Real, N> cluster_velocity = sphere_dist_velocity(my_gen);

        while(cluster_id < clusters_to_generate && elements.size() < n) {
            Real sphere_dist_var = condition->sig * std::pow(K[cluster_id], 1.0/3.0) * 2.;
            statistic::UniformSphericalDistribution<N, Real>
                    sphere_dist_position(sphere_dist_var, cluster_centerx, cluster_centery, cluster_centerz);

            int trial = 0;
            while(trial < max_trial && part_in_cluster < clusters[cluster_id] && elements.size() < n) { // stop when you cant generate new particles with less than 10000 trials within a cluster
                auto element = elements::Element<N>(sphere_dist_position(my_gen), sphere_dist_velocity(my_gen), elements.size(), elements.size());
                if(condition->predicate(element)) {
                    trial = 0;
                    element.velocity = cluster_velocity;
                    elements.push_back(element);
                    number_of_element_generated++;
                    part_in_cluster++;
                } else trial++;
            }
            if(trial == max_trial) {
                std::cerr << "increase sphere size of cluster " << cluster_id << ", " << part_in_cluster<< "/"<<clusters[cluster_id] << std::endl;
                K[cluster_id] *= 4;
            } else{
                part_in_cluster = 0;
                cluster_centerx = udistx(my_gen);
                cluster_centery = udistx(my_gen);
                cluster_centerz = udistz(my_gen);
                cluster_velocity = sphere_dist_velocity(my_gen);
                cluster_id++;
            }
        }
    }
};

template<int N>
class UniformRandomElementsGenerator : public RandomElementsGenerator<N> {
    int seed;

    const int max_trial;
public:
    UniformRandomElementsGenerator(int seed, const int max_trial = 10000) : seed(seed), max_trial(max_trial) {}

    void generate_elements(std::vector<elements::Element<N>>& elements, const int n,
                           const std::shared_ptr<lj::RejectionCondition<N>> condition) override {
        int number_of_element_generated = 0;
        const Real dblT0Sqr = 2.0 * condition->T0 * condition->T0;
        std::normal_distribution<Real> temp_dist(0.0, dblT0Sqr);
        std::uniform_real_distribution<Real> utemp_dist(0.0, dblT0Sqr);
        std::uniform_real_distribution<Real>
            udistx(condition->xmin, condition->xmax),
            udisty(condition->ymin, condition->ymax),
            udistz(condition->zmin, condition->zmax);

        std::mt19937 my_gen(seed);
        int trial = 0;
        std::array<Real, N>  element_position, velocity;

        Integer lcxyz, lc[N];
        Real cut_off = condition->params->rc;
        lc[0] = std::round((condition->xmax - condition->xmin) / cut_off);
        lc[1] = std::round((condition->ymax - condition->ymin) / cut_off);
        lcxyz = lc[0] * lc[1];
        if constexpr (N==3){
            lc[2] = std::round((condition->zmax - condition->zmin) / cut_off);
            lcxyz *= lc[2];
        }
        const Integer EMPTY = -1;
        std::vector<Integer> head(lcxyz, -1), lscl(n, -1);
        Integer generated = 0;
        std::array<Real, N> singularity;
        std::generate(singularity.begin(), singularity.end(), [&my_gen, &udist=udistx](){return udist(my_gen);});
        while(generated < n) {
            while(trial < max_trial) {

                if constexpr (N==3){
                    element_position = { udistx(my_gen), udisty(my_gen), udistz(my_gen) };
                    auto strength    = utemp_dist(my_gen);
                    velocity         = {
                            ((condition->xmin + singularity[0]) - element_position[0]) * strength,
                            ((condition->ymin + singularity[1]) - element_position[1]) * strength,
                            ((condition->zmin + singularity[2]) - element_position[2]) * strength
                    };
                } else {
                    auto strength    = utemp_dist(my_gen);
                    element_position = { udistx(my_gen), udisty(my_gen)};
                    velocity         = {
                            ((condition->xmin + singularity[0]) - element_position[0]) * strength,
                            ((condition->ymin + singularity[1]) - element_position[1]) * strength,
                    };
                }

                auto element = elements::Element<N>(element_position, velocity, elements.size(), elements.size());

                bool accepted = true;

                std::array<Real, 3> delta_dim;
                Integer c, c1, ic[N], ic1[N], j;
                elements::Element<N> receiver;
                c = position_to_cell<N>(element.position, cut_off, lc[0], lc[1]);

                for (auto d = 0; d < N; ++d)
                    ic[d] = c / lc[d];
                for (ic1[0] = ic[0] - 1; ic1[0] < (ic[0]+1); ic1[0]++) {
                    for (ic1[1] = ic[1] - 1; ic1[1] < ic[1] + 1; ic1[1]++) {
                        if constexpr (N==3) {
                            for (ic1[2] = ic[2] - 1; ic1[2] < ic[2] + 1; ic1[2]++) {
                                if ((ic1[0] < 0 || ic1[0] >= lc[0]) || (ic1[1] < 0 || ic1[1] >= lc[1]) ||
                                    (ic1[2] < 0 || ic1[2] >= lc[2]))
                                    continue;
                                c1 = (ic1[0]) + (lc[0] * ic1[1]) + (lc[0] * lc[1] * ic1[2]);
                                j = head[c1];
                                while (j != EMPTY) {
                                    if (generated < j) {
                                        receiver = elements[j];
                                        accepted =
                                                accepted && elements::distance2(receiver, element) >= condition->min_r2;
                                    }
                                    j = lscl[j];
                                }
                            }
                        } else {
                            if ((ic1[0] < 0 || ic1[0] >= lc[0]) || (ic1[1] < 0 || ic1[1] >= lc[1]))
                                continue;
                            c1 = (ic1[0]) + (lc[0] * ic1[1]);
                            j = head[c1];
                            while (j != EMPTY) {
                                if (generated < j) {
                                    receiver = elements[j];
                                    accepted = accepted && elements::distance2(receiver, element) >= condition->min_r2;
                                }
                                j = lscl[j];
                            }
                        }
                    }
                }

                if(accepted) {
                    trial = 0;
                    elements.push_back(element);
                    algorithm::CLL_append(generated, c, element, &head, &lscl);
                    generated++;
                    break;
                } else {
                    trial++;
                }
            }
            if(trial == max_trial) break; // when you cant generate new particles with less than max trials stop.
        }
    }
};

template<int N>
class HalfLoadedRandomElementsGenerator : public RandomElementsGenerator<N> {
    double division_pos;
    const bool direction; //true is positive, false is negative
    int seed;
    const int max_trial;

public:
    HalfLoadedRandomElementsGenerator(double division_position, bool direction, int seed, const int max_trial = 10000) :
            division_pos(division_position), direction(direction), seed(seed), max_trial(max_trial) {}

    void generate_elements(std::vector<elements::Element<N>>& elements, const int n,
                           const std::shared_ptr<lj::RejectionCondition<N>> condition) override {
        //division_pos = condition->xmax < division_pos ? condition->xmax : division_pos;
        int number_of_element_generated = 0;
        int already_generated = elements.size();
        std::normal_distribution<Real> temp_dist(0.0, 2.0 * condition->T0 * condition->T0);
        std::uniform_real_distribution<Real>
                udistx(0.0, division_pos),
                udisty(condition->ymin, condition->ymax),
                udistz(condition->zmin, condition->zmax);

        statistic::NormalSphericalDistribution<N, Real>
                sphere_dist_velocity(2.0 * condition->T0 * condition->T0, 0, 0, 0);
        std::array<Real, N>  element_velocity;
        std::mt19937 my_gen(seed);
        if(N>2) {
            element_velocity = {direction ? temp_dist(my_gen) : -temp_dist(my_gen), 0.0, 0.0};
        } else {
            element_velocity = {direction ? temp_dist(my_gen) : -temp_dist(my_gen), 0.0};
        }

        int trial = 0;
        while(elements.size()-already_generated < n) {
            while(trial < max_trial) {
                std::array<Real, N>  element_position;
                if(N>2)
                    element_position = {udistx(my_gen), udisty(my_gen), udistz(my_gen)} ;
                else
                    element_position = {udistx(my_gen), udisty(my_gen)};

                auto element = elements::Element<N>(element_position, element_velocity, elements.size(), elements.size());
                if(condition->predicate(element)) {
                    trial = 0;
                    elements.push_back(element);
                    break;
                } else{
                    trial++;
                }
            }
            if(trial == max_trial) break; // when you cant generate new particles with less than max trials stop.
        }
    }
};

template<int N>
class ParticleWallElementsGenerator : public RandomElementsGenerator<N> {
    const Real pw_pos;
    const bool direction; //true is positive, false is negative
    int seed;
    const int max_trial;
public:
    ParticleWallElementsGenerator(double pw_position, bool direction, int seed, const int max_trial = 10000) :
            pw_pos(pw_position), direction(direction), seed(seed), max_trial(max_trial) {}

    void generate_elements(std::vector<elements::Element<N>>& elements, const int n,
                           const std::shared_ptr<lj::RejectionCondition<N>> condition) override {
        int number_of_element_generated = 0;
        std::normal_distribution<Real> temp_dist(0.0, 2.0 * condition->T0 * condition->T0);
        std::uniform_real_distribution<Real>
                udistx(condition->xmin, condition->xmax),
                udisty(condition->ymin, condition->ymax),
                udistz(condition->zmin, condition->zmax);
        std::mt19937 my_gen(seed);
        int trial = 0;
        std::array<Real, N>  element_velocity;
        if(N>2) {
            element_velocity = {direction ? temp_dist(my_gen) : -temp_dist(my_gen), 0.0, 0.0};
        } else {
            element_velocity = {direction ? temp_dist(my_gen) : -temp_dist(my_gen), 0.0};
        }

        std::array<Real, N>  element_position, velocity;

        Integer lcxyz, lc[N];
        Real cut_off = condition->params->rc;
        lc[0] = (condition->xmax - condition->xmin) / cut_off;
        lc[1] = (condition->ymax - condition->ymin) / cut_off;
        lcxyz = lc[0] * lc[1];
        if constexpr (N==3){
            lc[2] = (condition->zmax - condition->zmin) / cut_off;
            lcxyz *= lc[2];
        }
        const Integer EMPTY = -1;
        std::vector<Integer> head(lcxyz, -1), lscl(n, -1);
        Integer generated = 0;
        while(generated < n) {
            while(trial < max_trial) {
                if(N>2) {
                    element_position = {pw_pos, udisty(my_gen), udistz(my_gen)};
                } else {
                    element_position = {pw_pos, udisty(my_gen)};
                }
                auto element = elements::Element<N>(element_position, element_velocity, elements.size(), elements.size());

                bool accepted = true;

                std::array<Real, 3> delta_dim;
                Integer c, c1, ic[N], ic1[N], j;
                elements::Element<N> receiver;
                c = position_to_cell<N>(element.position, cut_off, lc[0], lc[1]);
                for (auto d = 0; d < N; ++d)
                    ic[d] = c / lc[d];
                for (ic1[0] = ic[0] - 1; ic1[0] < (ic[0]+1); ic1[0]++) {
                    for (ic1[1] = ic[1] - 1; ic1[1] < ic[1] + 1; ic1[1]++) {
                        if constexpr (N==3) {
                            for (ic1[2] = ic[2] - 1; ic1[2] < ic[2] + 1; ic1[2]++) {
                                if ((ic1[0] < 0 || ic1[0] >= lc[0]) || (ic1[1] < 0 || ic1[1] >= lc[1]) ||
                                    (ic1[2] < 0 || ic1[2] >= lc[2]))
                                    continue;
                                c1 = (ic1[0]) + (lc[0] * ic1[1]) + (lc[0] * lc[1] * ic1[2]);
                                j = head[c1];
                                while (j != EMPTY) {
                                    if (generated < j) {
                                        receiver = elements[j];
                                        accepted =
                                                accepted && elements::distance2(receiver, element) >= condition->min_r2;
                                    }
                                    j = lscl[j];
                                }
                            }
                        } else {
                            if ((ic1[0] < 0 || ic1[0] >= lc[0]) || (ic1[1] < 0 || ic1[1] >= lc[1]))
                                continue;
                            c1 = (ic1[0]) + (lc[0] * ic1[1]);
                            j = head[c1];
                            while (j != EMPTY) {
                                if (generated < j) {
                                    receiver = elements[j];
                                    accepted = accepted && elements::distance2(receiver, element) >= condition->min_r2;
                                }
                                j = lscl[j];
                            }
                        }
                    }
                }

                if(accepted) {
                    trial = 0;
                    elements.push_back(element);
                    algorithm::CLL_append(generated, c, element, &head, &lscl);
                    generated++;
                    break;
                } else {
                    trial++;
                }
            }
            if(trial == max_trial) break; // when you cant generate new particles with less than max trials stop.
        }
    }
};
} // end of namespace lennard_jones

template<int N>
void initialize_mesh_data(int npart, MESH_DATA<N>& mesh_data,
                          initial_condition::RandomElementsGenerator<N>* elements_generator,
                          const std::shared_ptr<lj::RejectionCondition<N>> condition) {
    elements_generator->generate_elements(mesh_data.els, npart, condition);

}

} // end of namespace initial_condition

#endif //NBMPI_INITIAL_CONDITIONS_HPP
