//
// Created by xetql on 08.01.18.
//

#include <cppcheck/cppcheck.hpp>
#include <random>
#include "../includes/utils.hpp"
#include "../includes/spatial_elements.hpp"
#include "../includes/physics.hpp"
#include "../includes/ljpotential.hpp"
#include <algorithm>
#include <unordered_map>
using namespace lennard_jones;
int main(int argc, char **argv) {

    constexpr int DIM=2;

    TestsRunner runner("MPI Load Balancing: Tests");

    std::random_device rd; //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<double> dist(0.1, 0.6);

    std::vector<elements::Element<DIM>> points;
    //populate points
    points = {
            elements::Element<2>::createc({0.0, 0.0}, {0,0}, 0, 0),
            elements::Element<2>::createc({sig, 0.0}, {0,0}, 1, 1),
            elements::Element<2>::createc({0.0, sig}, {0,0}, 2, 2),
            elements::Element<2>::createc({sig, sig}, {0,0}, 3, 3)
    };

    auto computation_of_lj_scalar_within_rcut = std::make_shared<UnitTest<std::vector<elements::Element<DIM>>>>(
      "LJ potential is =/= zero inside rcut", [] {
        std::vector<elements::Element<DIM>> points;
        std::random_device rd; //Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
        std::uniform_real_distribution<double> dist(0.0, 2.5*sig);
        for(size_t i = 0; i < 100; i++)
            points.push_back(elements::Element<2>::create_random(dist, gen, i, i));

        for(auto &force_recepter : points){
            for(auto &force_source : points){
                if(force_source.gid != force_recepter.gid){
                    double dx = force_source.position.at(0) - force_recepter.position.at(0);
                    double dy = force_source.position.at(1) - force_recepter.position.at(1);
                    double C_LJ = compute_LJ_scalar(dx*dx+dy*dy, eps, sig2);
                    force_recepter.acceleration[0] += (C_LJ * dx);
                    force_recepter.acceleration[1] += (C_LJ * dy);
                }
            }
        }
        return points;
      },
      [](auto const &points){
          return std::all_of(points.begin(), points.end(), [](auto p){
              return p.acceleration.at(0) != 0 && p.acceleration.at(1) != 0;});
      }
    );

    auto computation_of_lj_scalar_outside_rcut = std::make_shared<UnitTest<std::vector<elements::Element<DIM>>>>(
            "LJ potential is zero outside rcut", [] {
                std::vector<elements::Element<DIM>> points;
                points = {
                        elements::Element<2>::createc({0.0, 0.0}, {0,0}, 0, 0),
                        elements::Element<2>::createc({2.25*sig, 2.25*sig}, {0,0}, 1, 1),
                };
                for(auto &force_recepter : points){
                    for(auto &force_source : points){
                        if(force_source.gid != force_recepter.gid){
                            double dx = force_source.position.at(0) - force_recepter.position.at(0);
                            double dy = force_source.position.at(1) - force_recepter.position.at(1);
                            double C_LJ = compute_LJ_scalar(dx*dx+dy*dy, eps, sig2);
                            force_recepter.acceleration[0] += (C_LJ * dx);
                            force_recepter.acceleration[1] += (C_LJ * dy);
                        }
                    }
                }
                return points;
            },
            [](auto const &points){
                return std::all_of(points.begin(), points.end(), [](auto p){
                    return p.acceleration.at(0) == 0 && p.acceleration.at(1) == 0;});
            }
    );

    auto cell_linked_list_creation = std::make_shared<UnitTest<std::unordered_map<int,int>>>(
            "LinkedList indeed contains all the particles", [] {
                std::vector<elements::Element<DIM>> points;
                std::random_device rd; //Will be used to obtain a seed for the random number engine
                std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
                std::uniform_real_distribution<double> dist(0.0, 1.0);
                std::unordered_map<int, int> pklist;
                int nsub = 4;
                double lsub = 1.0 / nsub;
                std::vector<int> head(nsub*nsub);

                for(size_t i = 0; i < 100; i++)
                    points.push_back(elements::Element<2>::create_random(dist, gen, i, i));

                //create_cell_linkedlist(nsub, lsub, points, pklist, head);

                return pklist;
            },
            [](auto const &pklist){
                return pklist.size() == 100;
            }
    );

    auto reflection = std::make_shared<UnitTest<std::vector<elements::Element<2>>>>(
            "Reflection are applied on the border of the simulation and points are within the boundaries", [] {
                std::vector<elements::Element<DIM>> points;
                std::random_device rd; //Will be used to obtain a seed for the random number engine
                std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
                std::uniform_real_distribution<double> dist(1.0, 1.2);
                for(size_t i = 0; i < 1000; i++){
                    auto e = elements::Element<2>::create_random(dist, gen, i, i);
                    e.velocity[0] = dist(gen) / 10.0;
                    e.velocity[1] = dist(gen) / 10.0;
                    points.push_back(e);
                }
                apply_reflect(points, 1.0);
                return points;
            },
            [](auto const &points){
                return std::all_of(points.begin(), points.end(), [](auto p){
                    std::array<std::pair<double,double>, 2> domain = {std::make_pair(0.0, 1.0),std::make_pair(0.0, 1.0)};
                    return elements::is_inside<2>(p, domain);
                });
            }
    );

    runner.add_test(computation_of_lj_scalar_within_rcut);
    runner.add_test(computation_of_lj_scalar_outside_rcut);
    //runner.add_test(cell_linked_list_creation);
    runner.add_test(reflection);

    runner.run();

    return runner.are_tests_passed();
}