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
    partitioning::geometric::Domain<2> _domain_boundary = {
            std::make_pair(0.0, 1.0),
            std::make_pair(0.0, 1.0),
    };
    std::vector<partitioning::geometric::Domain<2>> domain_boundary = {_domain_boundary};
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

    //do the same for 3d
    /*
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
*/
    SeqSpatialBisection<2> partitioner;
    load_balancing::geometric::GeometricLoadBalancer<2> load_balancer(partitioner, MPI_COMM_WORLD);

    auto load_balancer_exchange_data = std::make_shared<UnitTest<std::vector<elements::Element<2> > > >
            ("Data are divided among the processing elements", [&load_balancer, &points, &rank]{
                partitioning::geometric::Domain<2> _domain_boundary = {
                        std::make_pair(0.0, 1.0),
                        std::make_pair(0.0, 1.0),
                };
                std::vector<partitioning::geometric::Domain<2>> domain_boundary = {_domain_boundary};
                SeqSpatialBisection<2> partitioner;
                load_balancing::geometric::GeometricLoadBalancer<2> load_balancer(partitioner, MPI_COMM_WORLD);
                std::vector<elements::Element<2>> recipient(points.begin(), points.end());
                if(rank != 0) recipient.clear();
                load_balancer.load_balance(recipient, domain_boundary);
                //std::for_each(domain_boundary.begin(), domain_boundary.end(), [](auto const& el){std::cout << to_string(el) << std::endl;});
                return recipient;
            }, [&points, &load_balancer,&rank](auto recipient) {
                
                bool same = true;
                std::vector<elements::Element<2>> control(points.size());
                MPI_Gather(&recipient.front(), recipient.size(), load_balancer.get_element_datatype(), &control.front(), recipient.size(), load_balancer.get_element_datatype(), 0, MPI_COMM_WORLD);
                if(rank == 0) // Check if what has been received is what has been generated
                    same = std::all_of(control.begin(), control.end(), [&points](auto const& el) {return std::find(points.begin(), points.end(), el) != points.end(); }) ;

                
                
                return same;
            });

    partitioning::geometric::Domain<2> ranges = {std::make_pair(1.0, dist(gen)), std::make_pair(1.0, dist(gen))};

    auto can_send_range_through_mpi = std::make_shared<UnitTest<std::array<std::pair<double, double>, 2> > >
            ("Ranges are sent correctly via MPI", [&load_balancer, &ranges, &rank] () {
                SeqSpatialBisection<2> rcb_partitioner;
                load_balancing::geometric::GeometricLoadBalancer<2> load_balancer(rcb_partitioner, MPI_COMM_WORLD);
                partitioning::geometric::Domain<2> data;

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
            ("All the processing elements have something to compute", [&load_balancer, &points, &rank]{
                partitioning::geometric::Domain<2> _domain_boundary = {
                        std::make_pair(0.0, 1.0),
                        std::make_pair(0.0, 1.0),
                };
                std::vector<partitioning::geometric::Domain<2>> domain_boundary = {_domain_boundary};
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
            ("Real data are exchanged between the processing elements", [&load_balancer, &points, &rank]{
                partitioning::geometric::Domain<2> _domain_boundary = {
                        std::make_pair(0.0, 1.0),
                        std::make_pair(0.0, 1.0),
                };
                std::vector<partitioning::geometric::Domain<2>> domain_boundary = {_domain_boundary};
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
    auto can_send_domain_through_mpi = std::make_shared<UnitTest<Domain<2> > >
            ("Domain are sent correctly via MPI", [&load_balancer, &ranges, &rank] () {
                SeqSpatialBisection<2> rcb_partitioner;
                load_balancing::geometric::GeometricLoadBalancer<2> load_balancer(rcb_partitioner, MPI_COMM_WORLD);
                partitioning::geometric::Domain<2> data;

                if (rank == 1) // send points
                    MPI_Send(&ranges, 1, load_balancer.get_domain_datatype(), 0, 666, MPI_COMM_WORLD);
                else // retrieve points
                    MPI_Recv(&data, 1,  load_balancer.get_domain_datatype(), 1, 666, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if (rank == 0) // send points
                    MPI_Send(&data, 1, load_balancer.get_domain_datatype(), 1, 666, MPI_COMM_WORLD);
                else // retrieve points
                    MPI_Recv(&data, 1, load_balancer.get_domain_datatype(), 0, 666, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                return data;
            }, [&ranges, &load_balancer, &rank](auto recipient) {
                return std::all_of(recipient.begin(), recipient.end(), [&ranges](auto const& el) {return std::find(ranges.begin(), ranges.end(), el) != ranges.end(); }) ;
            });



    auto load_balancer_can_migrate_data = std::make_shared<UnitTest< std::pair<partitioning::geometric::Domain<2>, std::vector<elements::Element<2> > > > >
            ("Data are migrated among the PE", [&rank]{
                SeqSpatialBisection<2> rcb_partitioner;
                load_balancing::geometric::GeometricLoadBalancer<2> load_balancer_2(rcb_partitioner, MPI_COMM_WORLD);
                std::vector<elements::Element<2>> _recipient;
                std::random_device _rd; //Will be used to obtain a seed for the random number engine
                std::mt19937 _gen(_rd()); //Standard mersenne_twister_engine seeded with rd()
                std::uniform_real_distribution<double> _dist(0.0+rank, 1.0+rank);

                for (unsigned int i = 0; i < 10; ++i) {
                    _recipient.push_back(elements::Element<2>::create_random(_dist, _gen, i));
                }

                std::vector<partitioning::geometric::Domain<2>> domains = {
                        {std::make_pair(1.0, 2.0), std::make_pair(1.0, 2.0)},
                        {std::make_pair(0.0, 1.0), std::make_pair(0.0, 1.0)},
                };
                load_balancer_2.migrate_particles(_recipient, domains);
                return std::make_pair(domains.at(rank), _recipient);
            }, [](auto recipient) {
                auto r = recipient.second;
                auto dom = recipient.first;
                return std::all_of(r.begin(), r.end(), [&dom](auto data){return elements::is_inside<2>(data, dom);});
            });

    auto load_balancer_can_exchange_data_with_neighbors = std::make_shared<UnitTest< std::pair<partitioning::geometric::Domain<2>, std::vector<elements::Element<2> > > > >
            ("Data are exchanged among the PE", [&rank]{
                SeqSpatialBisection<2> rcb_partitioner;
                load_balancing::geometric::GeometricLoadBalancer<2> load_balancer_2(rcb_partitioner, MPI_COMM_WORLD);
                std::vector<elements::Element<2>> _recipient;
                std::random_device _rd; //Will be used to obtain a seed for the random number engine
                std::mt19937 gen(_rd()); //Standard mersenne_twister_engine seeded with rd()
                std::uniform_real_distribution<double> dx(0.1, 1.1);
                std::uniform_real_distribution<double> dy(1.0-rank, 2.0-rank);
                for (unsigned int i = 0; i < 1000; ++i) {
                    elements::Element<2> e;
                    e.position[0]=dx(gen);
                    e.position[1]=dy(gen);
                    _recipient.push_back(e);
                }
                std::vector<partitioning::geometric::Domain<2>> domains = {
                        {std::make_pair(0.1, 1.1), std::make_pair(1.0, 2.0)},
                        {std::make_pair(0.1, 1.1), std::make_pair(0.0, 1.0)},
                };
                auto remote_data = load_balancer_2.exchange_data(_recipient, domains);
                //std::for_each(remote_data.begin(), remote_data.end(), [](auto const& el){std::cout << el << std::endl;});
                return std::make_pair(domains.at(rank), remote_data);
            }, [](auto recipient) {
                auto r = recipient.second;
                auto dom = recipient.first;
                return std::all_of(r.begin(), r.end(), [&dom](auto data){return !elements::is_inside<2>(data, dom);});
            });

    runner.add_test(element2d_exchanged_correctly);
    //runner.add_test(element3d_exchanged_correctly);
    runner.add_test(after_load_balancing_processing_elements_have_data_to_compute);
    runner.add_test(processing_elements_have_real_data);
    runner.add_test(load_balancer_exchange_data);
    runner.add_test(can_send_range_through_mpi);
    runner.add_test(can_send_domain_through_mpi);
    runner.add_test(load_balancer_can_migrate_data);
    runner.add_test(load_balancer_can_exchange_data_with_neighbors);

    runner.run();

    if (rank == 0) {
        runner.summarize();
        passed = runner.are_tests_passed();
    }

    load_balancer.stop();
    MPI_Finalize();
    return passed;
}