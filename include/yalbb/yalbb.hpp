//
// Created by xetql on 4/14/21.
//

#ifndef NBMPI_YALBB_HPP
#define NBMPI_YALBB_HPP

#include "parallel_utils.hpp"

struct YALBB {
    MPI_Comm comm;
    YALBB(int argc, char** argv) {
        MPI_Init(&argc, &argv);
        MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    }
    ~YALBB() {
        MPI_Finalize();
    }
};

#endif //NBMPI_YALBB_HPP
