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
    unsigned int p = 256;
    for(int i = 0; i < 4; ++i) result_should_be.at(i) = p / 4;

    auto test_partition_balanced = std::make_shared<UnitTest<std::array<int, 4>>>
            ("Partitions contains the same number of elements", [&dist, &gen, &elements=p] {
                constexpr int DIM=2;
                std::vector<elements::Element<DIM>> points;
                //populate points
                for (unsigned int i = 0; i < elements; ++i) {
                    double x = dist(gen), y = dist(gen);
                    const elements::Element<2> e({x, y}, {0.0, 0.0});
                    points.push_back(e);
                }

                //apply bisection
                SeqSpatialBisection<DIM> partitioner;
                std::array<std::pair<double ,double>, DIM> domain_boundary = {
                        std::make_pair(0.0, 1.0),
                        std::make_pair(0.0, 1.0)
                };
                auto partitions = partitioner.partition_data(points, domain_boundary, 4);

                std::array<int, 4> result;
                std::fill(result.begin(), result.end(), 0);
                for(auto const& element : partitions->parts)
                    result[element.first] += 1;
                return result;
            }, result_should_be);

    auto  total_area_is_area_of_initial_domain = std::make_shared<UnitTest<std::vector<partitioning::geometric::Domain<2> > > >
            ("Total sub area equals initial total area", [&dist, &gen, &elements=p] {
                constexpr int DIM=2;
                std::vector<elements::Element<DIM>> points;
                //populate points
                for (unsigned int i = 0; i < elements; ++i) {
                    double x = dist(gen), y = dist(gen);
                    const elements::Element<2> e({x, y}, {0.0, 0.0});
                    points.push_back(e);
                }
                //apply bisection
                SeqSpatialBisection<DIM> partitioner;
                std::array<std::pair<double ,double>, DIM> domain_boundary = {
                        std::make_pair(0.0, 1.0),
                        std::make_pair(0.0, 1.0)
                };
                auto partitions = partitioner.partition_data(points, domain_boundary, 16);
                std::vector<partitioning::geometric::Domain<2> > d = partitions->domains;
                return d;
            }, [](auto const& subdomains) {
                double total_area = 1.0; //sqr domain
                double area_sum = 0.0;
                for(auto const& subdomain : subdomains){
                    const double width = subdomain.at(0).second - subdomain.at(0).first;
                    const double height= subdomain.at(1).second - subdomain.at(1).first;
                    area_sum += (double) (width * height);
                }
                return std::fabs(total_area - area_sum) <= std::numeric_limits<double>::epsilon();
            });

    auto  no_overlapping_region = std::make_shared<UnitTest<std::vector<partitioning::geometric::Domain<2> > > >
            ("Regions does not overlap", [&dist, &gen, &elements=p] {
                constexpr int DIM=2;
                std::vector<elements::Element<DIM>> points;
                //populate points
                for (unsigned int i = 0; i < elements; ++i) {
                    double x = dist(gen), y = dist(gen);
                    const elements::Element<2> e({x, y}, {0.0, 0.0});
                    points.push_back(e);
                }
                //apply bisection
                SeqSpatialBisection<DIM> partitioner;
                std::array<std::pair<double ,double>, DIM> domain_boundary = {
                        std::make_pair(0.0, 1.0),
                        std::make_pair(0.0, 1.0)
                };
                auto partitions = partitioner.partition_data(points, domain_boundary, 16);
                std::vector<partitioning::geometric::Domain<2> > d = partitions->domains;
                return d;
            }, [](auto const& subdomains) {
                for(size_t i = 0; i < subdomains.size(); ++i){
                    auto const subdomain_a = subdomains.at(i);
                    for(size_t j = i; j < subdomains.size(); ++j){
                        auto const subdomain_b = subdomains.at(j);
                        //Left or right boundary is within boundaries of another region
                        bool regionA_overlaps_regionB =
                                ((subdomain_a.at(0).first < subdomain_b.at(0).first && subdomain_b.at(0).first < subdomain_a.at(0).second) ||
                                 (subdomain_a.at(0).first < subdomain_b.at(0).second && subdomain_b.at(0).second < subdomain_a.at(0).second)) &&
                                ((subdomain_a.at(1).first < subdomain_b.at(1).first && subdomain_b.at(1).first < subdomain_a.at(1).second) ||
                                 (subdomain_a.at(1).first < subdomain_b.at(1).second && subdomain_b.at(1).second < subdomain_a.at(1).second));
                        bool regionB_overlaps_regionA =
                                ((subdomain_b.at(0).first < subdomain_a.at(0).first && subdomain_a.at(0).first < subdomain_b.at(0).second) ||
                                 (subdomain_b.at(0).first < subdomain_a.at(0).second && subdomain_a.at(0).second < subdomain_b.at(0).second)) &&
                                ((subdomain_b.at(1).first < subdomain_a.at(1).first && subdomain_a.at(1).first < subdomain_b.at(1).second) ||
                                 (subdomain_b.at(1).first < subdomain_a.at(1).second && subdomain_a.at(1).second < subdomain_b.at(1).second));
                        if (regionA_overlaps_regionB || regionB_overlaps_regionA){ return false; }
                    }
                }
                return true;
            });

    runner.add_test(test_partition_balanced);
    runner.add_test(total_area_is_area_of_initial_domain);
    runner.add_test(no_overlapping_region);

    runner.run();
    runner.summarize();

    return runner.are_tests_passed();
}