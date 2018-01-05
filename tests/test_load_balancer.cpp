//
// Created by xetql on 12/27/17.
//

#include <cppcheck/cppcheck.hpp>
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
    std::vector<elements::Element<2>> points;
    std::vector<elements::Element<2>> recipient(elements);
    for (int i = 0; i < elements; ++i) {
        points.push_back(elements::Element<2>::create_random(dist, gen, i));
    }
    std::array<std::pair<double ,double>, 2> domain_boundary = {
            std::make_pair(0.0, 1.0),
            std::make_pair(0.0, 1.0),
    };
    // One is able to send a vector of 2D element to another processing element
    auto element2d_exchanged_correctly = std::make_shared<UnitTest<std::vector<elements::Element<2>>>>
            ("elements::Elements 2d exchange correctly performed", [&points, &rank] {
                SeqSpatialBisection<2> partitioner;
                load_balancing::geometric::GeometricLoadBalancer<2> load_balancer(partitioner, MPI_COMM_WORLD);
                std::vector<elements::Element<2>> recipient(points.size());

                if (rank == 1) // send generated points
                    MPI_Send(&points.front(), points.size(), load_balancer.get_element_datatype(), 0, 1, MPI_COMM_WORLD);
                else if (rank == 0) // retrieve sent points
                    MPI_Recv(&recipient.front(), recipient.size(), load_balancer.get_element_datatype(), 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                load_balancer.stop();
                return recipient;
            }, [&points, &rank](auto received_data) {
                bool same = true;
                if(rank == 0){ // Check if what has been received is what has been generated earlier
                    for(size_t i = 0; i < received_data.size(); ++i){
                        same = points.at(i) == received_data.at(i) ? same : false;
                    }
                }
                return same;
            });

    //do the same for 3d points
    std::vector<elements::Element<3>> points3d;
    std::vector<elements::Element<3>> recipient3d(elements);

    for (int i = 0; i < elements; ++i) {

        points3d.push_back(elements::Element<3>::create_random(dist, gen, i));
    }

    // One is able to send a vector of 3D element to another processing element
    auto element3d_exchanged_correctly = std::make_shared<UnitTest<std::vector<elements::Element<3>>>>
            ("elements::Elements 3d exchange correctly performed", [&points3d, &rank] {
                SeqSpatialBisection<3> rcb_partitioner;
                load_balancing::geometric::GeometricLoadBalancer<3> load_balancer(rcb_partitioner, MPI_COMM_WORLD);
                std::vector<elements::Element<3>> recipient(points3d.size());
                if (rank == 1) // send generated points
                    MPI_Send(&points3d.front(), points3d.size(), load_balancer.get_element_datatype(), 0, 1, MPI_COMM_WORLD);
                else // retrieve sent points
                    MPI_Recv(&recipient.front(), recipient.size(), load_balancer.get_element_datatype(), 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                load_balancer.stop();
                return recipient;
            }, [&points3d, &rank](auto received_data) {
                bool same = true;
                if(rank == 0) // Check if what has been received is what has been generated
                    for(size_t i = 0; i < received_data.size(); ++i){
                        same = points3d.at(i) == received_data.at(i) ? same : false;

                    }
                return same;
            });
    SeqSpatialBisection<2> partitioner;
    load_balancing::geometric::GeometricLoadBalancer<2> load_balancer(partitioner, MPI_COMM_WORLD);

    auto load_balancer_exchange_data = std::make_shared<UnitTest<std::vector<elements::Element<2> > > >
            ("Data are divided among the processing elements", [&load_balancer, &points, &domain_boundary, &rank]{
                std::vector<elements::Element<2>> recipient(points.begin(), points.end());
                if(rank != 0) recipient.clear();
                load_balancer.load_balance(recipient, domain_boundary);
                return recipient;
            }, [&points, &load_balancer,&rank](auto recipient) {
                bool same = true;
                std::vector<elements::Element<2>> control(points.size());
                MPI_Gather(&recipient.front(), recipient.size(), load_balancer.get_element_datatype(), &control.front(), recipient.size(), load_balancer.get_element_datatype(), 0, MPI_COMM_WORLD);
                if(rank == 0) // Check if what has been received is what has been generated
                    same = std::all_of(control.begin(), control.end(), [&points](auto const& el) {return std::find(points.begin(), points.end(), el) != points.end(); }) ;

                return same;
            });

    std::array<std::pair<double, double>, 2> ranges = {
            std::make_pair(dist(gen), dist(gen)),
            std::make_pair(dist(gen), dist(gen))
    };

    auto can_send_range_through_mpi = std::make_shared<UnitTest<std::array<std::pair<double, double>, 2> > >
            ("Ranges are sent correctly via MPI", [&load_balancer, &ranges, &rank] () {
                SeqSpatialBisection<2> rcb_partitioner;
                load_balancing::geometric::GeometricLoadBalancer<2> load_balancer(rcb_partitioner, MPI_COMM_WORLD);
                std::array<std::pair<double, double>, 2> data;

                if (rank == 1) // send generated points
                    MPI_Send(&ranges.front(), ranges.size(), load_balancer.get_range_datatype(), 0, 666, MPI_COMM_WORLD);
                else // retrieve sent points
                    MPI_Recv(&data.front(), data.size(), load_balancer.get_range_datatype(), 1, 666, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if (rank == 0) // send generated points
                    MPI_Send(&data.front(), data.size(), load_balancer.get_range_datatype(), 1, 666, MPI_COMM_WORLD);
                else // retrieve sent points
                    MPI_Recv(&data.front(), data.size(), load_balancer.get_range_datatype(), 0, 666, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                return data;
            }, [&ranges, &load_balancer, &rank](auto recipient) {
                return std::all_of(recipient.begin(), recipient.end(), [&ranges](auto const& el) {return std::find(ranges.begin(), ranges.end(), el) != ranges.end(); }) ;
            });

    auto after_load_balancing_processing_elements_have_data_to_compute = std::make_shared<UnitTest<std::vector<elements::Element<2> > > >
            ("All the processing elements have something to compute", [&load_balancer, &points, &domain_boundary, &rank]{
                std::vector<elements::Element<2>> recipient(points.begin(), points.end());
                if(rank != 0) recipient.clear();
                load_balancer.load_balance(recipient, domain_boundary);
                return recipient;
            }, [&points, &load_balancer,&rank, &world](auto recipient) {
                bool same = true;
                int recipient_not_empty = recipient.size() > 0 ? 1 : 0;
                std::vector<int> control(world);
                MPI_Gather(&recipient_not_empty, 1, MPI_INT, &control.front(), 1, MPI_INT, 0, MPI_COMM_WORLD);
                if(rank == 0){ // Check if what has been received is what has been generated
                    same = std::accumulate(control.begin(), control.end(), 0) == world ? true : false;
                }
                return same;
            });

    auto processing_elements_have_real_data = std::make_shared<UnitTest<std::vector<elements::Element<2> > > >
            ("Real data are exchanged between the processing elements", [&load_balancer, &points, &domain_boundary, &rank]{
                std::vector<elements::Element<2>> recipient(points.begin(), points.end());
                if(rank != 0) recipient.clear();
                load_balancer.load_balance(recipient, domain_boundary);
                return recipient;
            }, [&points, &load_balancer,&rank,&world](auto recipient) {
                bool same = true;
                int recipient_contain_data = std::all_of(recipient.begin(), recipient.end(), [&points](auto const& el) {return std::find(points.begin(), points.end(), el) != points.end(); }) ? 1 : 0;
                std::vector<int> control(world);
                MPI_Gather(&recipient_contain_data, 1, MPI_INT, &control.front(), 1, MPI_INT, 0, MPI_COMM_WORLD);
                if(rank == 0){ // Check if what has been received is what has been generated
                    same = std::accumulate(control.begin(), control.end(), 0) == world ? true : false;
                }
                return same;
            });

    runner.add_test(element2d_exchanged_correctly);
    runner.add_test(element3d_exchanged_correctly);
    runner.add_test(after_load_balancing_processing_elements_have_data_to_compute);
    runner.add_test(processing_elements_have_real_data);
    runner.add_test(load_balancer_exchange_data);
    runner.add_test(can_send_range_through_mpi);

    runner.run();

    if (rank == 0) {
        runner.summarize();
        passed = runner.are_tests_passed();
    }

    load_balancer.stop();
    MPI_Finalize();
    return passed;
}