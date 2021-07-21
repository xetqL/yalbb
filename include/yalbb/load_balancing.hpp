//
// Created by xetql on 7/21/21.
//

#ifndef NBMPI_LOAD_BALANCING_HPP
#define NBMPI_LOAD_BALANCING_HPP
#include <mpi.h>
// extends these to use your own Load Balancing Method
namespace lb {
    template<class T=void>
    struct InitLB {
    };          // Init load balancer functor
    template<class T=void>
    struct Copier {
    };          // Copy ptr functor, used in optimal finder
    template<class T=void>
    struct Destroyer {
    };       // Destructor functor
    template<class T=void>
    struct DoPartition {
    };     // Do partitioning functor
    template<class T=void>
    struct IntersectDomain {
    }; // Domain intersection functor
    template<class T=void>
    struct AssignPoint {
    };     // Point assignation functor

    template<class LB, class GetPosF>
    struct DoLB {

        MPI_Datatype datatype;
        MPI_Comm APP_COMM;

        GetPosF getPositionPtrFunc;
        DoPartition<LB> doPart{};

        DoLB(MPI_Datatype datatype, MPI_Comm APP_COMM, GetPosF getPosF) :
                datatype(datatype), APP_COMM(APP_COMM), getPositionPtrFunc(getPosF) {}

        template<class T>
        void operator()(LB *zlb, T *mesh_data) {
            doPart(zlb, mesh_data, getPositionPtrFunc);
        }

    };
}
#endif //NBMPI_LOAD_BALANCING_HPP
