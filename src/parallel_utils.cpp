//
// Created by xetql on 4/29/20.
//
#include "parallel_utils.hpp"

std::vector<int> get_invert_list(const std::vector<int>& sends_to_procs, int* num_found, MPI_Comm comm) {

    int worldsize, rank = 0;
    int how_many_to_import = 0;

    MPI_Comm_size(comm, &worldsize);
    MPI_Comm_rank(comm, &rank);

    std::vector<int> import_from_procs;

    // create requet to send/recv
    std::vector<MPI_Request> recv_req(worldsize, MPI_REQUEST_NULL);
    std::vector<MPI_Request> send_req(worldsize, MPI_REQUEST_NULL);

    // Send the HOW MANY to my neighbors
    for(int PE = 0; PE < worldsize; PE++) {
        MPI_Isend(&sends_to_procs.at(PE), 1, MPI_INT, PE, 2, comm, &send_req[PE]);
    }

    int recv_size = 0;
    for(int PE = 0; PE < worldsize; PE++) {
        MPI_Recv(&recv_size, 1, MPI_INT, PE, 2, comm, MPI_STATUS_IGNORE);
        if(PE != rank && recv_size > 0){
            import_from_procs.push_back(PE);
            how_many_to_import += recv_size;
        }
    }

    *num_found = how_many_to_import;
    MPI_Waitall(worldsize, send_req.data(), MPI_STATUSES_IGNORE);

    return import_from_procs;
}