//
// Created by xetql on 4/14/21.
//

#ifndef NBMPI_YALBB_HPP
#define NBMPI_YALBB_HPP

#include "parallel_utils.hpp"
#include "io.hpp"
struct YALBB {

    MPI_Comm comm;

    YALBB(int argc, char** argv) {
        int init;
        MPI_Initialized(&init);
        if(!init) MPI_Init(&argc, &argv);

        MPI_Comm_dup(MPI_COMM_WORLD, &comm);
        pcout = std::make_unique<io::ParallelOutput>(std::cout);
        pcerr = std::make_unique<io::ParallelOutput>(std::cerr);
    }

    ~YALBB() {
        MPI_Finalize();
    }

    io::ParallelOutput& get_parallel_cout(){
        return *pcout;
    }
    io::ParallelOutput& get_parallel_cerr(){
        return *pcerr;
    }
private:
    std::unique_ptr<io::ParallelOutput> pcout, pcerr;

};

#endif //NBMPI_YALBB_HPP
