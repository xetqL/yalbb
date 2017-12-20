//
// Created by xetql on 18.12.17.
//

#ifndef NBMPI_PARTITIONER_HPP
#define NBMPI_PARTITIONER_HPP

#include <vector>
#include <memory>

namespace partitioning {
    template<typename PartitionsType, typename DataTypeContainer>
    class Partitioner {
    public:
        virtual std::unique_ptr<PartitionsType> partition_data(DataTypeContainer spatial_data,
                                                               int number_of_partitions) const = 0;
    };
}

#endif //NBMPI_PARTITIONER_HPP
