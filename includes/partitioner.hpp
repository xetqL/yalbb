//
// Created by xetql on 18.12.17.
//

#ifndef NBMPI_PARTITIONER_HPP
#define NBMPI_PARTITIONER_HPP

#include <vector>
#include <memory>

namespace partitioning {
    struct CommunicationDatatype {

        MPI_Datatype vec_datatype;
        MPI_Datatype elements_datatype;
        MPI_Datatype range_datatype;
        MPI_Datatype domain_datatype;

        CommunicationDatatype(const MPI_Datatype &vec,
                              const MPI_Datatype &elements,
                              const MPI_Datatype &range_datatype,
                              const MPI_Datatype &domain_datatype ) :
            vec_datatype(vec),
            elements_datatype(elements),
            range_datatype(range_datatype),
            domain_datatype(domain_datatype){}
        void free_datatypes(){
            MPI_Type_free(&vec_datatype);
            MPI_Type_free(&elements_datatype);
            MPI_Type_free(&range_datatype);
        }
    };
    template<typename PartitionsType, typename DataTypeContainer, typename DomainContainer>
    class Partitioner {
    public:
        virtual std::unique_ptr<PartitionsType> partition_data(DataTypeContainer spatial_data,
                                                               DomainContainer   &domain_boundary,
                                                               int number_of_partitions) const = 0;
        virtual CommunicationDatatype register_datatype() const = 0;

    };
}

#endif //NBMPI_PARTITIONER_HPP
