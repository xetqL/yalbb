//
// Created by xetql on 08.06.18.
//

#ifndef NBMPI_INITIAL_CONDITIONS_HPP
#define NBMPI_INITIAL_CONDITIONS_HPP
#include "spatial_elements.hpp"
#include "utils.hpp"

namespace initial_condition {

static std::random_device __rd;
static std::mt19937 __gen(__rd()); //Standard mersenne_twister_engine seeded with rd()

template<class Candidate>
class RejectionCondition {
public:
    virtual const bool predicate(const Candidate& c) const = 0;
};

namespace lennard_jones {
template<int N>
class RejectionCondition : public initial_condition::RejectionCondition<elements::Element<N>> {
    const std::vector<elements::Element<N>>* others;
public:
    const elements::ElementRealType sig;
    const elements::ElementRealType min_r2;
    const elements::ElementRealType T0;
    const elements::ElementRealType xmin;
    const elements::ElementRealType ymin;
    const elements::ElementRealType zmin;
    const elements::ElementRealType xmax;
    const elements::ElementRealType ymax;
    const elements::ElementRealType zmax;

    RejectionCondition(const std::vector<elements::Element<N>>* others,
                       const elements::ElementRealType sig,
                       const elements::ElementRealType min_r2,
                       const elements::ElementRealType T0,
                       const elements::ElementRealType xmin,
                       const elements::ElementRealType ymin,
                       const elements::ElementRealType zmin,
                       const elements::ElementRealType xmax,
                       const elements::ElementRealType ymax,
                       const elements::ElementRealType zmax) :
            others(others),
            sig(sig), min_r2(min_r2), T0(T0),
            xmin(xmin), ymin(ymin), zmin(zmin),
            xmax(xmax), ymax(ymax), zmax(zmax){}

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
} // end of namespace lennard_jones

template<int N>
class RandomElementsGenerator {
public:
    virtual void generate_elements(std::vector<elements::Element<N>>& elements, const int n, const lennard_jones::RejectionCondition<N>* cond) = 0;
};

namespace lennard_jones {

template<int N>
class RandomElementsInClustersGenerator : public RandomElementsGenerator<N> {
    const int max_particles_per_cluster, seed, max_trial;
public:
    int number_of_clusters_generated = 0;

    RandomElementsInClustersGenerator(const int max_particles_per_cluster, const int seed = __rd(), const int max_trial = 10000) :
            max_particles_per_cluster(max_particles_per_cluster), seed(seed), max_trial(max_trial) {}

    void generate_elements(std::vector<elements::Element<N>>& elements, const int n,
                           const lennard_jones::RejectionCondition<N>* condition) override {
        elements.clear();
        int number_of_element_generated = 0;
        int clusters_to_generate = 1;
        int cluster_id = 0;
        std::normal_distribution<elements::ElementRealType> temp_dist(0.0, condition->T0 * condition->T0);
        std::uniform_real_distribution<elements::ElementRealType> udistx(condition->xmin, condition->xmax),
                                                                  udisty(condition->ymin, condition->ymax),
                                                                  udistz(condition->zmin, condition->zmax);
        std::mt19937 my_gen(seed);
        while(cluster_id < clusters_to_generate && elements.size() < n) {
            elements::ElementRealType cluster_centerx = udistx(my_gen),
                                      cluster_centery = udistx(my_gen),
                                      cluster_centerz = udistz(my_gen);
            statistic::NormalSphericalDistribution<N, elements::ElementRealType>
                    sphere_dist_position(condition->sig * (max_particles_per_cluster), cluster_centerx, cluster_centery, cluster_centerz);
            statistic::NormalSphericalDistribution<N, elements::ElementRealType>
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

template<int N, int C>
class RandomElementsInNClustersGenerator : public RandomElementsGenerator<N> {
    std::array<int, C> clusters;
    const int seed, max_trial;
public:

    RandomElementsInNClustersGenerator(std::array<int, C> clusters, const int seed = __rd(), const int max_trial = 10000) :
            clusters(clusters), seed(seed), max_trial(max_trial) {}

    void generate_elements(std::vector<elements::Element<N>>& elements, const int n,
                           const lennard_jones::RejectionCondition<N>* condition) override {
        elements.clear();
        float x_sz = condition->xmax - condition->xmin;
        float y_sz = condition->ymax - condition->ymin;
        float z_sz = condition->zmax - condition->zmin;

        int number_of_element_generated = 0;
        const int clusters_to_generate = clusters.size();
        int cluster_id = 0;
        std::normal_distribution<elements::ElementRealType> temp_dist(0.0, condition->T0 * condition->T0);
        std::uniform_real_distribution<elements::ElementRealType>
                udistx(condition->xmin+x_sz*0.05, condition->xmax-x_sz*0.05),
                udisty(condition->ymin+y_sz*0.05, condition->ymax-y_sz*0.05),
                udistz(condition->zmin+z_sz*0.05, condition->zmax-z_sz*0.05);
        std::mt19937 my_gen(seed);
        std::array<int, C> K = clusters;
        int part_in_cluster = 0;
        elements::ElementRealType cluster_centerx = udistx(my_gen),
                                  cluster_centery = udistx(my_gen),
                                  cluster_centerz = udistz(my_gen);
        while(cluster_id < clusters_to_generate && elements.size() < n) {

            elements::ElementRealType sphere_dist_var = condition->sig * std::pow(K[cluster_id], 1.0/3.0) / 2.0;

            statistic::UniformSphericalDistribution<N, elements::ElementRealType>
                    sphere_dist_position(sphere_dist_var, cluster_centerx, cluster_centery, cluster_centerz);
            statistic::NormalSphericalDistribution<N, elements::ElementRealType>
                    sphere_dist_velocity(2.0 * condition->T0 * condition->T0, 0, 0, 0);
            auto cluster_velocity = sphere_dist_velocity(my_gen);
            int trial = 0;

            while(trial < max_trial && part_in_cluster < clusters[cluster_id] && elements.size() < n) { // stop when you cant generate new particles with less than 10000 trials within a cluster
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
            if(trial == max_trial) {
                std::cout << "increase sphere size of cluster " << cluster_id << ", " << part_in_cluster<< "/"<<clusters[cluster_id] << std::endl;
                K[cluster_id] *= 1.5;
            } else{
                part_in_cluster = 0;
                cluster_centerx = udistx(my_gen);
                cluster_centery = udistx(my_gen);
                cluster_centerz = udistz(my_gen);
                cluster_id++;
            }
        }
    }
};


template<int N>
class UniformRandomElementsGenerator : public RandomElementsGenerator<N> {
    const int max_trial;
public:
    UniformRandomElementsGenerator(const int max_trial = 10000) : max_trial(max_trial) {}

    void generate_elements(std::vector<elements::Element<N>>& elements, const int n,
                           const lennard_jones::RejectionCondition<N>* condition) override {
        elements.clear();
        int number_of_element_generated = 0;
        std::normal_distribution<elements::ElementRealType> temp_dist(0.0, condition->T0 * condition->T0);
        std::uniform_real_distribution<elements::ElementRealType> udistx(condition->xmin, condition->xmax),
                                                                  udisty(condition->ymin, condition->ymax),
                                                                  udistz(condition->zmin, condition->zmax);

        statistic::NormalSphericalDistribution<N, elements::ElementRealType>
                sphere_dist_velocity(2.0 * condition->T0 * condition->T0, 0, 0, 0);

        int trial = 0;
        while(elements.size() < n) {
            while(trial < max_trial) {
                std::array<elements::ElementRealType, N>  element_position;
                if(N>2)
                    element_position = {udistx(__gen), udisty(__gen), udistx(__gen)} ;
                else
                    element_position = {udistx(__gen), udisty(__gen)};

                auto element = elements::Element<N>(element_position, sphere_dist_velocity(__gen), elements.size(), elements.size());
                if(condition->predicate(element)) {
                    trial = 0;
                    std::generate(element.velocity.begin(), element.velocity.end(), [&temp_dist]{return temp_dist(__gen);});
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

} // end of namespace lennard_jones

template<int N>
void initialize_mesh_data(int npart, MESH_DATA<N>& mesh_data,
                    initial_condition::RandomElementsGenerator<N>* elements_generator,
                    const lennard_jones::RejectionCondition<N>& condition) {
    elements_generator->generate_elements(mesh_data.els, npart, &condition);
}

} // end of namespace initial_condition

#endif //NBMPI_INITIAL_CONDITIONS_HPP
