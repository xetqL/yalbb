//
// Created by xetql on 2/28/20.
//

#ifndef NBMPI_COMMUNICATION_DATATYPE_HPP
#define NBMPI_COMMUNICATION_DATATYPE_HPP

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



#endif //NBMPI_COMMUNICATION_DATATYPE_HPP
