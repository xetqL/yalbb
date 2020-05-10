//
// Created by xetql on 4/10/20.
//

#ifndef NBMPI_PARALLEL_UTILS_HPP
#define NBMPI_PARALLEL_UTILS_HPP

#include "utils.hpp"
#include "coordinate_translater.hpp"
#include "cll.hpp"

#include <mpi.h>
#include <vector>
#include <numeric>
#include <set>

using Real       = float;
using Time       = double;
using Rank       = int;
using Integer    = long long int;
using Complexity = Integer;
using Index      = Integer;

#define MPI_TIME MPI_DOUBLE
#define MPI_COMPLEXITY MPI_LONG_LONG

#define TIME_IT(a, name){\
 double start = MPI_Wtime();\
 a;\
 double end = MPI_Wtime();\
 auto diff = (end - start) / 1e-3;\
 std::cout << name << " took " << diff << " milliseconds" << std::endl;\
};\

#define START_TIMER(var)\
double var = MPI_Wtime();

#define RESTART_TIMER(v) \
v = MPI_Wtime() - v;

#define END_TIMER(var)\
var = MPI_Wtime() - var;

#define PAR_START_TIMER(var, comm)\
MPI_Barrier(comm);\
double var = MPI_Wtime();\

#define PAR_END_TIMER(var, comm)\
MPI_Barrier(comm);\
var = MPI_Wtime() - var;

struct Borders {
    std::vector<std::vector<Rank>> neighbors;
    std::vector<Index> bordering_cells;
};

// Get the how much data I have to import from other processing elements
// This function is a synchronization point.
std::vector<int> get_invert_list(const std::vector<int>& sends_to_procs, int* num_found, MPI_Comm comm);

template<class LoadBalancer, class BoxIntersectFunc>
Borders get_border_cells_index3d(
        LoadBalancer* LB,
        std::vector<Integer> *head,
        const BoundingBox<3>& bbox,
        const Real rc,
        BoxIntersectFunc boxIntersectFunc,
        MPI_Comm comm) {
    constexpr int N = 3;
    int caller_rank, wsize, num_found;

    MPI_Comm_rank(comm, &caller_rank);
    MPI_Comm_size(comm, &wsize);

    if (wsize == 1) return {};
    auto lc = get_cell_number_by_dimension<N>(bbox, rc);

    // number of bordering cell in 3D is xyz - (x-2)(y-2)(z-2)
    int nb_local_cells     = std::accumulate(lc.cbegin(), lc.cend(), 1, [](auto p, auto v){return p*v;});
    int nb_bordering_cells = std::min(nb_local_cells, nb_local_cells - std::accumulate(lc.cbegin(), lc.cend(), 1, [](auto p, auto v){return p * (v-2);}));

    // If I have no bordering cell (impossible, but we never know (: )
    if(nb_bordering_cells <= 0) return {};

    std::vector<Rank> PEs(wsize);
    std::array<double, 3> pos_in_double;
    Integer border_cell_cnt = 0;
    std::vector<Index> bordering_cell_index;
    bordering_cell_index.reserve(nb_bordering_cells);
    std::vector< std::vector<Rank> > neighbors; neighbors.reserve(nb_bordering_cells);
    double radius = 2.0*rc;
    for(Index cell_id = 0; cell_id < nb_local_cells; ++cell_id) {
        if(head->at(cell_id) != -1) {
            auto xyz = CoordinateTranslater::translate_linear_index_into_xyz_array<3>(cell_id, lc[0], lc[1]);
            put_in_double_array<3>(pos_in_double,
                                   CoordinateTranslater::translate_local_xyz_into_position<N>(xyz, bbox, rc));
            boxIntersectFunc(LB,
                             pos_in_double.at(0) + rc / 2.0 - radius, pos_in_double.at(1) + rc / 2.0 - radius,
                             pos_in_double.at(2) + rc / 2.0 - radius, pos_in_double.at(0) + rc / 2.0 + radius,
                             pos_in_double.at(1) + rc / 2.0 + radius, pos_in_double.at(2) + rc / 2.0 + radius,
                             &PEs.front(), &num_found);
            if (num_found) {
                bordering_cell_index.push_back(cell_id);
                neighbors.push_back(std::vector<Rank>());
                neighbors.at(border_cell_cnt).assign(PEs.begin(), PEs.begin() + num_found);
                border_cell_cnt++;
            }
        }
    }
    return {neighbors, bordering_cell_index};
}

template<class LoadBalancer, class BoxIntersectFunc>
Borders get_border_cells_index2d(
        LoadBalancer* LB,
        std::vector<Integer> *head,
        const BoundingBox<2>& bbox,
        const Real rc,
        BoxIntersectFunc boxIntersectFunc,
        MPI_Comm comm) {
    constexpr int N = 2;
    int caller_rank, wsize, num_found;

    MPI_Comm_rank(comm, &caller_rank);
    MPI_Comm_size(comm, &wsize);

    if (wsize == 1) return {};
    auto lc = get_cell_number_by_dimension<N>(bbox, rc);


    // number of bordering cell in 3D is xyz - (x-2)(y-2)(z-2)
    int nb_local_cells     = std::accumulate(lc.cbegin(), lc.cend(), 1, [](auto p, auto v){return p*v;});
    int nb_bordering_cells = std::min(nb_local_cells, nb_local_cells - std::accumulate(lc.cbegin(), lc.cend(), 1, [](auto p, auto v){return p * (v-2);}));

    // If I have no bordering cell (impossible, but we never know (: )
    if(nb_bordering_cells <= 0) return {};
    std::vector<Rank> PEs(wsize);
    std::array<double, 3> pos_in_double;
    Integer border_cell_cnt = 0;
    std::vector<Index> bordering_cell_index;
    bordering_cell_index.reserve(nb_bordering_cells);
    std::vector< std::vector<Rank> > neighbors; neighbors.reserve(nb_bordering_cells);
    double radius = rc/2.0+rc;
    for(Index cell_id = 0; cell_id < nb_local_cells; ++cell_id) {
        if(head->at(cell_id) != -1){
            auto xyz = CoordinateTranslater::translate_linear_index_into_xyz_array<3>(cell_id, lc[0], lc[1]);
            put_in_double_array<3>(pos_in_double,
                                   CoordinateTranslater::translate_local_xyz_into_position<N>(xyz, bbox, rc));
            boxIntersectFunc(LB,
                             pos_in_double.at(0) + rc / 2.0 - radius, pos_in_double.at(1) + rc / 2.0 - radius,
                             0.0, pos_in_double.at(0) + rc / 2.0 + radius,
                             pos_in_double.at(1) + rc / 2.0 + radius, 0.0,
                             &PEs.front(), &num_found);
            if (num_found) {
                bordering_cell_index.push_back(cell_id);
                neighbors.push_back({});
                neighbors.at(border_cell_cnt).assign(PEs.begin(), PEs.begin() + num_found);
                border_cell_cnt++;
            }
        }
    }
    return {neighbors, bordering_cell_index};
}
template<int N, class LoadBalancer, class BoxIntersectFunc>
Borders get_border_cells_index(
        LoadBalancer* LB,
        std::vector<Integer> *head,
        const BoundingBox<N>& bbox,
        const Real rc,
        BoxIntersectFunc boxIntersectFunc,
        MPI_Comm comm) {

    int size;
    MPI_Comm_size(comm, &size);
    if(size == 1) return {};

    if constexpr(N==3){
        return get_border_cells_index3d<LoadBalancer, BoxIntersectFunc>(LB, head, bbox, rc, boxIntersectFunc, comm);
    } else {
        return get_border_cells_index2d<LoadBalancer, BoxIntersectFunc>(LB, head, bbox, rc, boxIntersectFunc, comm);
    }
}

template<class T>
std::vector<T> exchange_data(
        const std::vector<T> &data,
        const std::vector<Integer>* head,
        const std::vector<Integer>* lscl,
        const Borders& bordering_cells,
        MPI_Datatype datatype,
        MPI_Comm LB_COMM,
        int &nb_elements_recv,
        int &nb_elements_sent) {

    const auto nb_elements = data.size();

    int wsize, caller_rank;
    MPI_Comm_size(LB_COMM, &wsize);
    MPI_Comm_rank(LB_COMM, &caller_rank);

    std::vector<T> buffer;
    std::vector<T> remote_data_gathered;

    if (wsize == 1)
        return remote_data_gathered;

    std::vector<std::vector<T > > data_to_migrate(wsize);
    std::for_each(data_to_migrate.begin(), data_to_migrate.end(),
                  [size = nb_elements, wsize](auto &buf) { buf.reserve(size / wsize); });

    int num_found, num_known = 0;

    //std::vector<int> export_gids, export_lids, export_procs;
    int cell_cnt = 0;
    for(auto cidx : bordering_cells.bordering_cells) {
        auto p = head->at(cidx);
        while(p != -1) {
            num_known += bordering_cells.neighbors.at(cell_cnt).size();
            p = lscl->at(p);
        }
        cell_cnt++;
    }

    cell_cnt = 0;
    num_known = 0;
    for(auto cidx : bordering_cells.bordering_cells){
        auto p = head->at(cidx);
        while(p != -1){
            for(auto rank : bordering_cells.neighbors.at(cell_cnt)) {
                if(rank != caller_rank) {
                    const auto& el = data[p];
                    data_to_migrate.at(rank).push_back(el);
                    num_known ++;
                }
            }
            p = lscl->at(p);
        }
        cell_cnt++;
    }
    std::vector<int> sends_to_proc(wsize);
    std::transform(data_to_migrate.cbegin(), data_to_migrate.cend(), std::begin(sends_to_proc), [](const auto& el){return el.size();});

    auto import_from_procs = get_invert_list(sends_to_proc, &num_found, LB_COMM);

    int nb_reqs = std::count_if(data_to_migrate.cbegin(), data_to_migrate.cend(), [](const auto& buf){return !buf.empty();});
    int cpt = 0;

    // Send the data to neighbors
    std::vector<MPI_Request> reqs(nb_reqs);
    nb_elements_sent = 0;
    for (size_t PE = 0; PE < wsize; PE++) {
        int send_size = data_to_migrate.at(PE).size();
        if (send_size) {
            nb_elements_sent += send_size;
            MPI_Isend(&data_to_migrate.at(PE).front(), send_size, datatype, PE, 400, LB_COMM, &reqs[cpt]);
            cpt++;
        }
    }

    // Import the data from neighbors
    nb_elements_recv = 0;
    remote_data_gathered.reserve(num_found);
    int recv_count = import_from_procs.size();
    MPI_Status status;
    int size;
    while(recv_count) {
        // Probe for next incoming message
        MPI_Probe(MPI_ANY_SOURCE, 400, LB_COMM, &status);
        // Get message size
        MPI_Get_count(&status, datatype, &size);
        nb_elements_recv += size;
        // Resize buffer if needed
        if(buffer.capacity() < size) buffer.reserve(size);
        // Receive data
        MPI_Recv(buffer.data(), size, datatype, status.MPI_SOURCE, 400, LB_COMM, MPI_STATUS_IGNORE);
        // Move to my data
        std::move(buffer.begin(), buffer.begin()+size, std::back_inserter(remote_data_gathered));
        // One less message to recover
        recv_count--;
    }

    MPI_Waitall(reqs.size(), &reqs.front(), MPI_STATUSES_IGNORE);

    return remote_data_gathered;
}

template<class T, class LoadBalancer, class PointAssignFunc>
typename std::vector<T>::const_iterator migrate_data(
        LoadBalancer* LB,
        std::vector<T> &data,
        PointAssignFunc pointAssignFunc,
        MPI_Datatype datatype,
        MPI_Comm LB_COMM) {
    int wsize;
    MPI_Comm_size(LB_COMM, &wsize);
    int caller_rank;
    MPI_Comm_rank(LB_COMM, &caller_rank);
    const auto prev_size = data.size();
    if(wsize == 1) return data.cend();
    auto nb_elements = data.size();

    std::vector< std::vector<T> > data_to_migrate(wsize);
    std::for_each(data_to_migrate.begin(), data_to_migrate.end(),
                  [size = nb_elements, wsize](auto &buf) { buf.reserve(size / wsize); });

    size_t data_id = 0;
    int PE, num_known = 0, num_found;
    std::vector<int> import_from_procs;
    {
        //std::vector<int> export_gids, export_lids, export_procs;
        //export_gids.reserve(nb_elements / wsize);
        //export_lids.reserve(nb_elements / wsize);
        //export_procs.reserve(nb_elements / wsize);
        while (data_id < nb_elements) {
            pointAssignFunc(LB, &data.at(data_id), &PE);
            if (PE != caller_rank) {
                //export_gids.push_back(data.at(data_id).gid);
                //export_lids.push_back(data.at(data_id).lid);
                //export_procs.push_back(PE);
                //if the current element has to be moved, then swap with the last and pop it out (dont need to move the pointer also)
                //swap iterator values in constant time
                std::iter_swap(data.begin() + data_id, data.end() - 1);
                //get the value and push it in the "to migrate" vector
                data_to_migrate.at(PE).push_back(*(data.end() - 1));
                //pop the head of the list in constant time
                data.pop_back();
                nb_elements--;
                num_known++;
            } else data_id++; //if the element must stay with me then check the next one
        }
        //export_gids.shrink_to_fit();
        //export_lids.shrink_to_fit();
        //export_procs.shrink_to_fit();
        std::vector<int> sends_to_proc(wsize);
        std::transform(data_to_migrate.cbegin(), data_to_migrate.cend(), std::begin(sends_to_proc), [](const auto& el){return el.size();});
        import_from_procs = get_invert_list(sends_to_proc, &num_found, LB_COMM);
    }

    data.reserve(nb_elements + num_found);

    /* Let's Migrate ma boi ! */

    int nb_reqs = std::count_if(data_to_migrate.cbegin(), data_to_migrate.cend(), [](const auto& buf){return !buf.empty();});

    int cpt = 0;

    std::vector<MPI_Request> reqs(nb_reqs);
    for (size_t PE = 0; PE < wsize; PE++) {
        int send_size = data_to_migrate.at(PE).size();
        if (send_size) {
            MPI_Isend(&data_to_migrate.at(PE).front(), send_size, datatype, PE, 300, LB_COMM,
                      &reqs[cpt]);
            cpt++;
        }
    }

    std::vector<T> buffer;

    int recv_count = import_from_procs.size();
    MPI_Status status;
    int size;
    while(recv_count) {
        // Probe for next incoming message
        MPI_Probe(MPI_ANY_SOURCE, 300, LB_COMM, &status);
        // Get message size
        MPI_Get_count(&status, datatype, &size);
        // Resize buffer if needed
        if(buffer.capacity() < size) buffer.reserve(size);
        // Receive data
        MPI_Recv(buffer.data(), size, datatype, status.MPI_SOURCE, 300, LB_COMM, MPI_STATUS_IGNORE);
        // Move to my data
        std::move(buffer.begin(), buffer.begin()+size, std::back_inserter(data));
        // One less message to recover
        recv_count--;
    }
    const int nb_data = data.size();
    //std::sort(data.begin(), data.end());
    for(int i = 0; i < nb_data; ++i) data[i].lid = i;
    MPI_Waitall(reqs.size(), &reqs.front(), MPI_STATUSES_IGNORE);
    return std::next(data.begin(), nb_data - prev_size);
}

template<class T>
inline void gather_elements_on(const int world_size,
                               const int my_rank,
                               const int nb_elements,
                               const std::vector<T> &local_el,
                               const int dest_rank,
                               std::vector<T> &dest_el,
                               const MPI_Datatype &sendtype,
                               const MPI_Comm &comm) {
    int nlocal = local_el.size();
    std::vector<int> counts(world_size, 0), displs(world_size, 0);
    MPI_Gather(&nlocal, 1, MPI_INT, &counts.front(), 1, MPI_INT, dest_rank, comm);
    for (int cpt = 0; cpt < world_size; ++cpt) displs[cpt] = cpt == 0 ? 0 : displs[cpt - 1] + counts[cpt - 1];
    if (my_rank == dest_rank) dest_el.resize(nb_elements);
    MPI_Gatherv(&local_el.front(), nlocal, sendtype,
                &dest_el.front(),
                &counts.front(),
                &displs.front(), sendtype, dest_rank, comm);
}

template<int N, class T, class GetPosFunc>
std::vector<T> get_ghost_data(
        std::vector<T>& els,
        GetPosFunc getPosPtrFunc,
        std::vector<Integer>* head, std::vector<Integer>* lscl,
        BoundingBox<N>& bbox, Borders borders, Real rc,
        MPI_Datatype datatype, MPI_Comm comm) {
    int r, s;
    MPI_Comm_size(comm, &s);
    if(s == 1) return {};
    auto remote_el = exchange_data<T>(els, head, lscl, borders, datatype, comm, r, s);
    return remote_el;
}
#endif //NBMPI_PARALLEL_UTILS_HPP
