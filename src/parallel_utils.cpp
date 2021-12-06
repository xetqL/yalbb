//
// Created by xetql on 4/29/20.
//
#include "parallel_utils.hpp"

std::vector<int>& get_invert_list(const std::vector<int>& sends_to_procs, int* num_found, MPI_Comm comm) {

    int worldsize = sends_to_procs.size();

    static std::vector<int> recv_from_proc(worldsize, 0);

    MPI_Alltoall(sends_to_procs.data(), 1, MPI_INT, recv_from_proc.data(), 1, MPI_INT, comm);

    *num_found = std::accumulate(recv_from_proc.begin(), recv_from_proc.end(), 0);

    return recv_from_proc;
}