//
// Created by xetql on 19.12.17.
//

#include <cppcheck/cppcheck.hpp>
#include <algorithm>
#include <map>
#include <random>

#include "../includes/utils.hpp"

using namespace partitioning::utils;
int main(int argc, char** argv){
    TestsRunner runner("Utility functions: Tests");

    std::random_device rd; //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    /* test zip */

    std::vector<double> vec1(100), vec2(100);
    // generate random numbers
    std::generate(vec1.begin(), vec1.begin(), [&dist, &gen]{return dist(gen);});
    std::generate(vec2.begin(), vec2.begin(), [&dist, &gen]{return dist(gen);});

    auto is_zip_correct = std::make_shared<UnitTest<std::vector<std::pair<double, double> > > >(
            "Two vectors should be zipped in order", [&vec1, &vec2]{
                return zip(vec1, vec2);
            }, [&vec1, &vec2](auto result) {
                bool is_ok = true;
                for(size_t i = 0; i < result.size(); ++i){
                    if(result.at(i).first != vec1.at(i) || result.at(i).second != vec2.at(i) ) {
                        is_ok = false;
                        break;
                    }
                }
                return is_ok;
            }
    );

    /* test unzip */
    auto zipped = zip(vec1, vec2);
    auto is_unzip_correct = std::make_shared<UnitTest<std::pair< std::vector<double>, std::vector<double> > > >(
            "A zipped vector should be unzipped in order", [&zipped]{
                return unzip(zipped);
            }, [&zipped](auto result){
                bool is_ok = true;
                auto vec1 = result.first;
                auto vec2 = result.second;
                for(size_t i = 0; i < zipped.size(); ++i){
                    if(zipped.at(i).first != vec1.at(i) || zipped.at(i).second != vec2.at(i) ) {
                        is_ok = false;
                        break;
                    }
                }
                return is_ok;
            }
    );

    /* test flatten */

    std::vector<std::vector<double>> two_dim_vec(100);
    std::fill(two_dim_vec.begin(), two_dim_vec.end(), vec1);

    auto is_flatten_size_correct = std::make_shared<UnitTest< std::vector<double> > >(
            "A flattened vector's size is the sum of all its component", [&two_dim_vec]{
                return flatten(two_dim_vec);
            }, [&two_dim_vec](auto result){
                int size = std::accumulate(two_dim_vec.begin(), two_dim_vec.end(), 0,
                                           [](unsigned int acc_sz, std::vector<double> row){return  acc_sz + row.size();});
                return result.size() == (unsigned int) size;
            }
    );

    auto is_flatten_correct = std::make_shared<UnitTest< std::vector<double> > >(
            "A flattened vector contains the same data as the original one", [&two_dim_vec]{
                return flatten(two_dim_vec);
            }, [&two_dim_vec](auto result){
                std::vector<double>::iterator it = result.begin();
                for(auto const& row:two_dim_vec) {
                    if(!std::equal(row.begin(), row.end(), it, std::next(it, row.size()))) return false;
                    else std::advance(it, row.size());
                }
                return true;
            }
    );

    runner.add_test(is_unzip_correct);
    runner.add_test(is_zip_correct);
    runner.add_test(is_flatten_size_correct);
    runner.add_test(is_flatten_correct);

    runner.run();
    runner.summarize();

    return runner.are_tests_passed();
}