//
// Created by xetql on 18.12.17.
//

#ifndef NBMPI_LOADBALANCER_HPP
#define NBMPI_LOADBALANCER_HPP

#include <map>
#include <mpi.h>

#include "partitioner.hpp"
#include "spatial_bisection.hpp"

namespace load_balancing {
    using ProcessingElementID=int;

    template<typename PartitionType, typename ContainedDataType>
    class LoadBalancer {
    protected:
        const partitioning::Partitioner<PartitionType, std::vector<ContainedDataType>> partitioner;
        const std::map<ProcessingElementID, std::vector<ContainedDataType>> storage_table;
        int world_size;
        int caller_rank;

    public:
        //Assume that the data are all located on PE 0.
        LoadBalancer(const partitioning::Partitioner<PartitionType, std::vector<ContainedDataType>> &partitioner,
                     MPI_Comm comm) : partitioner(partitioner){
            MPI_Comm_size(comm, &world_size);
            MPI_Comm_rank(comm, &caller_rank);

        };

        //Provide a table that specifies where is the data located (on which PE)
        LoadBalancer(const std::map<ProcessingElementID, std::vector<ContainedDataType>> &storage_table,
                     const partitioning::Partitioner<PartitionType, std::vector<ContainedDataType>> &partitioner,
                     MPI_Comm comm) : partitioner(partitioner), storage_table(storage_table){
            MPI_Comm_size(comm, &world_size);
            MPI_Comm_rank(comm, &caller_rank);
        };

        //TODO: It should somehow retrieves the data from the processing elements, at least makes it possible.
        //TODO: It should map the partitions as function of the PEs' pace
        virtual void load_balance(std::vector<ContainedDataType>& data){
            const std::vector<int> counts(world_size);
            const std::vector<int> displs(world_size);
            std::fill(counts.begin(), counts.end(), 0);
            if (caller_rank = 0){
                //partition the dataset into the size of the world
                std::unique_ptr<PartitionType> partitions = partitioner.partition_data(data, world_size);
                auto partitioned_data = partitions->parts;
                std::sort(data.begin(), data.end(), [](const auto & a, const auto & b) -> bool{
                    return a.first < b.first;
                });
                //counts the number to send to each processor
                displs[0] = 0;
                for(unsigned int i=0; i < world_size; ++i){
                    for(unsigned int cpt=displs[i]; cpt < partitioned_data.size(); ++cpt){
                        if(partitioned_data.at(cpt).first == i)
                            counts[i] += 1;
                        else {
                            displs[i+1] = cpt;
                            break;
                        }
                    }
                }
            }
            //MPI_Scatterv(&data.front(), counts, displs, );
        }

    private:
        /**
         * Migrate data according to the partitions returned by the partitioner
         */
        virtual void migrate_data(){}
    };

    class GeometricLoadBalancer2D : public LoadBalancer<partitioning::geometric::Partitions2D, partitioning::geometric::Element2D> {
    public:
        GeometricLoadBalancer2D(
                const partitioning::Partitioner<
                        partitioning::geometric::Partitions2D,
                        partitioning::geometric::Element2D> &partitioner, MPI_Comm comm) : LoadBalancer(partitioner, comm) {}
    };

}

#endif //NBMPI_LOADBALANCER_HPP
