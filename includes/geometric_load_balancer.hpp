//
// Created by xetql on 18.12.17.
//

#ifndef NBMPI_LOADBALANCER_HPP
#define NBMPI_LOADBALANCER_HPP

#include <map>
#include <mpi.h>

#include "partitioner.hpp"
#include "spatial_elements.hpp"
#include "spatial_bisection.hpp"

namespace load_balancing {
    using ProcessingElementID=int;

    template<typename PartitionType, typename ContainedDataType, typename DomainContainer>
    class LoadBalancer {
    private:
        /**
         * Register a MPI data type for a 2D or 3D particle structure.
         */
        //TODO: The structure will integrate an id => add a new field in the data type.
        void register_datatype(){
            MPI_Datatype oldtype[1];

            int array_size = ContainedDataType::size() / 2;
            MPI_Type_contiguous(array_size, MPI_DOUBLE, &vec_datatype);
            MPI_Type_commit(&vec_datatype);

            MPI_Aint offset[1] = {0};
            int blockcount[1];
            blockcount[0] = 2;
            oldtype[0] = vec_datatype;

            MPI_Type_struct(1, blockcount, offset, oldtype, &whole_datatype);
            MPI_Type_commit(&whole_datatype);
        }

    protected:
        const partitioning::Partitioner<PartitionType, std::vector<ContainedDataType>, DomainContainer>* partitioner;
        const std::map<ProcessingElementID, std::vector<ContainedDataType>> storage_table;
        int world_size;
        int caller_rank;
        MPI_Datatype whole_datatype;
        MPI_Datatype vec_datatype;

    public:

        const MPI_Datatype get_element_datatype() const {
            return whole_datatype;
        }

        //Assume that the data are all located on PE 0.
        LoadBalancer(const partitioning::Partitioner<PartitionType, std::vector<ContainedDataType>, DomainContainer> &partitioner,
                     MPI_Comm comm) : partitioner(&partitioner){
            MPI_Comm_size(comm, &world_size);
            MPI_Comm_rank(comm, &caller_rank);
            register_datatype();
        };

        //Provide a table that specifies where is the data located (on which PE)
        LoadBalancer(const std::map<ProcessingElementID, std::vector<ContainedDataType>> &storage_table,
                     const partitioning::Partitioner<PartitionType, std::vector<ContainedDataType>, DomainContainer> &partitioner,
                     MPI_Comm comm) : partitioner(&partitioner), storage_table(storage_table){
            MPI_Comm_size(comm, &world_size);
            MPI_Comm_rank(comm, &caller_rank);
            register_datatype();
        };

        ~LoadBalancer(){}

        void stop(){
            MPI_Type_free(&whole_datatype);
            MPI_Type_free(&vec_datatype);
        }

        //TODO: It should somehow retrieves the data from the processing elements, at least makes it possible.
        //TODO: It should map the partitions as function of the PEs' pace
        virtual void load_balance(std::vector<ContainedDataType>& data, DomainContainer& domain_boundary){
            std::vector<int> counts(world_size);
            std::vector<int> displs(world_size);
            std::fill(counts.begin(), counts.end(), 0);

            //TODO: Have to detect if the algorithm is sequential or parallel
            if (caller_rank == 0){
                //partition the dataset in world_size partitions
                std::unique_ptr<PartitionType> partitions = partitioner->partition_data(data, domain_boundary, world_size);
                auto partitioned_data = partitions->parts;
                std::sort(partitioned_data.begin(), partitioned_data.end(), [](const auto & a, const auto & b) -> bool{
                    return a.first < b.first;
                });
                //counts the number to send to each processor
                displs[0] = 0;
                for(int i=0; i < world_size; ++i)
                    for(size_t cpt=displs[i]; cpt < partitioned_data.size(); ++cpt)
                        if(partitioned_data.at(cpt).first == i)
                            counts[i] += 1;
                        else {
                            displs[i+1] = cpt;
                            break;
                        }
                for(int i = 0; i < world_size; ++i){
                    MPI_Send(&data[displs[i]], counts[i], this->get_element_datatype(), i, 666, MPI_COMM_WORLD);
                }
            }

            // Check the number of elements sent to us
            MPI_Status status;
            MPI_Probe(0, 666, MPI_COMM_WORLD, &status);
            int my_data_size;
            MPI_Get_count(&status, this->get_element_datatype(), &my_data_size);
            // resize the data to fit the number of elementss
            data.resize(my_data_size);
            // receive the elements
            MPI_Recv(&data.front(), my_data_size, this->get_element_datatype(), 0 , 666, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

    };
    namespace geometric {
        template<int N>
        class GeometricLoadBalancer : public LoadBalancer<partitioning::geometric::PartitionsInfo<N>, elements::Element<N>, const std::array<std::pair<double, double>, N>> {
        public:
            /**
             * Create a load balancer based on geometric partitioning algorithms
             * @param partitioner Partitioning algorithm
             * @param comm Communication group
             */l
            GeometricLoadBalancer(
                    const partitioning::Partitioner<
                            partitioning::geometric::PartitionsInfo<N>,
                            std::vector<elements::Element<N>>,
                            const std::array<std::pair<double, double>, N>> &partitioner, MPI_Comm comm) : LoadBalancer<partitioning::geometric::PartitionsInfo<N>, elements::Element<N>, const std::array<std::pair<double, double>, N>>(partitioner, comm) {}
        };
    }

}

#endif //NBMPI_LOADBALANCER_HPP
