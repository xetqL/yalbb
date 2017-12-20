//
// Created by xetql on 13.12.17.
//
#include <cppcheck/cppcheck.hpp>
#include <random>
#include <map>

#include "../includes/spatial_bisection.hpp"
using namespace partitioning::geometric;
int main(int argc, char** argv){
    TestsRunner runner("Space Partitioning: Tests");

    std::random_device rd; //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::array<int , 4> result_should_be;
    unsigned int p = 8;
    for(int i = 0; i < 4; ++i) result_should_be.at(i) = p / 4;

    auto test_partition_balanced = std::make_shared<UnitTest<std::array<int, 4>>>
            ("Partitions contains the same number of elements", [&dist, &gen, &elements=p] {
                std::vector<Element<2>> points;
                //populate points
                for (unsigned int i = 0; i < elements; ++i) {
                    double x = dist(gen), y = dist(gen);
                    points.push_back({x,y});
                }

                //apply bisection
                SeqSpatialBisection<2> partitioner;
                auto partitions = partitioner.partition_data(points, 4);

                std::array<int, 4> result;
                std::fill(result.begin(), result.end(), 0);
                for(auto const& element : partitions->parts)
                    result[element.first] += 1;
                return result;
            }, result_should_be);


    runner.add_test(test_partition_balanced);

    runner.run();
    runner.summarize();

    return runner.are_tests_passed();
}