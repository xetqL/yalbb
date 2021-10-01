//
// Created by xetql on 8/10/21.
//

#ifndef NBMPI_ELEMENT_HPP
#define NBMPI_ELEMENT_HPP

#include "type.hpp"
#include <array>
#include <mpi.h>

template<unsigned N>
struct BaseElement {
    Index gid = 0, lid = 0;
    std::array<Real, N> position {},  velocity {};
    static std::array<Real, N>* getElementPositionPtr(BaseElement<N>* e);
    static std::array<Real, N>* getElementVelocityPtr(BaseElement<N>* e);
    static MPI_Datatype register_datatype();
};

template<unsigned int N>
std::array<Real, N> *BaseElement<N>::getElementPositionPtr(BaseElement<N> *e) { return &(e->position); }

template<unsigned int N>
std::array<Real, N> *BaseElement<N>::getElementVelocityPtr(BaseElement<N> *e) { return &(e->velocity); }

template<unsigned int N>
MPI_Datatype BaseElement<N>::register_datatype(){
    constexpr const bool UseDoublePrecision = std::is_same<Real, double>::value;
    MPI_Datatype element_datatype, vec_datatype, oldtype_element[2];

    MPI_Aint offset[2], lb, intex;

    int blockcount_element[2];

    // register particle element type
    constexpr int array_size = N;
    auto mpi_raw_datatype = UseDoublePrecision ? MPI_DOUBLE : MPI_FLOAT;

    MPI_Type_contiguous(array_size, mpi_raw_datatype, &vec_datatype);

    MPI_Type_commit(&vec_datatype);

    blockcount_element[0] = 2; //gid, lid
    blockcount_element[1] = 2; //position, velocity

    oldtype_element[0] = MPI_LONG_LONG;
    oldtype_element[1] = vec_datatype;

    MPI_Type_get_extent(MPI_LONG_LONG, &lb, &intex);

    offset[0] = static_cast<MPI_Aint>(0);
    offset[1] = blockcount_element[0] * intex;

    MPI_Type_create_struct(2, blockcount_element, offset, oldtype_element, &element_datatype);

    MPI_Type_commit(&element_datatype);

    return element_datatype;
}

#endif //NBMPI_ELEMENT_HPP


