//
// Created by xetql on 12/27/17.
//

#include <cppcheck/cppcheck.hpp>
#include <iostream>
#include <mpi.h>

#include "../includes/utils.hpp"
#include "../includes/spatial_bisection.hpp"
#include "../includes/geometric_load_balancer.hpp"

using namespace partitioning::geometric;
int main(int argc, char **argv) {
    int rank, world, passed = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // TestsRunner is in silent mode if rank is not zero
    TestsRunner runner("MPI Load Balancing: Tests", rank != 0 );

    std::random_device rd; //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    gen.seed(0);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    int elements = 100;

    //generate the same 2D points on all processing elements
    std::vector<Element<2>> points;
    std::vector<Element<2>> recipient(elements);
    for (int i = 0; i < elements; ++i) {
        const double x = dist(gen), y = dist(gen);
        const Element<2> e({x, y}, {0.1, 2.0});
        points.push_back(e);
    }
    SeqSpatialBisection<2> partitioner;
    load_balancing::GeometricLoadBalancer<2> geometric_lb(partitioner, MPI_COMM_WORLD);
    // One is able to send a vector of 2D element to another processing element
    auto element2d_exchanged_correctly = std::make_shared<UnitTest<std::vector<Element<2>>>>
            ("Element 2d exchange correctly performed", [&elements, &points, &rank, &geometric_lb] {
                std::vector<Element<2>> recipient(elements);
                if (rank == 1) // send generated points
                    MPI_Send(&points.front(), elements, geometric_lb.get_element_datatype(), 0, 1, MPI_COMM_WORLD);
                else // retrieve sent points
                    MPI_Recv(&recipient.front(), elements, geometric_lb.get_element_datatype(), 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                return recipient;
            }, [&points, &rank](auto received_data) {
                bool same = true;
                if(rank == 0) // Check if what has been received is what has been generated earlier
                    for(size_t i = 0; i < received_data.size(); ++i){
                        same = points.at(i) == received_data.at(i) ? same : false;
                    }
                return same;
            });

    //do the same for 3d points
    int elements3d = 100;
    std::vector<Element<3>> points3d;
    std::vector<Element<3>> recipient3d(elements3d);

    for (int i = 0; i < elements3d; ++i) {
        const double x = dist(gen), y = dist(gen), z = dist(gen);
        const Element<3> e({x, y, z}, {0.1, 2.0, 1.0});
        points3d.push_back(e);
    }
    SeqSpatialBisection<3> partitioner3d;
    load_balancing::GeometricLoadBalancer<3> geometric_lb3d(partitioner3d, MPI_COMM_WORLD);

    // One is able to send a vector of 2D element to another processing element
    auto element3d_exchanged_correctly = std::make_shared<UnitTest<std::vector<Element<3>>>>
            ("Element 3d exchange correctly performed", [&elements3d, &points3d, &rank, &geometric_lb3d] {
                std::vector<Element<3>> recipient(elements3d);
                if (rank == 1) // send generated points
                    MPI_Send(&points3d.front(), elements3d, geometric_lb3d.get_element_datatype(), 0, 1, MPI_COMM_WORLD);
                else // retrieve sent points
                    MPI_Recv(&recipient.front(), elements3d, geometric_lb3d.get_element_datatype(), 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                return recipient;
            }, [&points3d, &rank](auto received_data) {
                bool same = true;
                if(rank == 0) // Check if what has been received is what has been generated earlier
                    for(size_t i = 0; i < received_data.size(); ++i){
                        same = points3d.at(i) == received_data.at(i) ? same : false;
                    }
                return same;
            });

    runner.add_test(element2d_exchanged_correctly);
    runner.add_test(element3d_exchanged_correctly);

    runner.run();

    if (rank == 0) {
        runner.summarize();
        passed = runner.are_tests_passed();
    }
    geometric_lb.stop();
    MPI_Finalize();
    return passed;
}