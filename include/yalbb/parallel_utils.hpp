//
// Created by xetql on 4/10/20.
//

#ifndef NBMPI_PARALLEL_UTILS_HPP
#define NBMPI_PARALLEL_UTILS_HPP

#include "type.hpp"
#include "utils.hpp"
#include "coordinate_translater.hpp"
#include "cll.hpp"

#include <mpi.h>
#include <vector>
#include <numeric>
#include <set>

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
double var = MPI_Wtime();

#define PAR_END_TIMER(var, comm)\
MPI_Barrier(comm);\
var = MPI_Wtime() - var;
    
template<class T>
constexpr MPI_Datatype get_mpi_type() {
    if constexpr (std::is_same<T, float>::value) return MPI_FLOAT;
    if constexpr (std::is_same<T, double>::value) return MPI_DOUBLE;
    if constexpr (std::is_same<T, int>::value) return MPI_INT;
    if constexpr (std::is_same<T, unsigned int>::value) return MPI_UNSIGNED;
    if constexpr (std::is_same<T, long>::value) return MPI_LONG;
    if constexpr (std::is_same<T, long int>::value) return MPI_LONG_INT;
    if constexpr (std::is_same<T, long double>::value) return MPI_LONG_DOUBLE;
    if constexpr (std::is_same<T, long long>::value) return MPI_LONG_LONG;
    if constexpr (std::is_same<T, long long int>::value) return MPI_LONG_LONG_INT;
    if constexpr (std::is_same<T, unsigned long>::value) return MPI_UNSIGNED_LONG;
    if constexpr (std::is_same<T, unsigned long long>::value) return MPI_UNSIGNED_LONG_LONG;
    if constexpr (std::is_same<T, short>::value) return MPI_SHORT;
    if constexpr (std::is_same<T, short int>::value) return MPI_SHORT_INT;
    if constexpr (std::is_same<T, char>::value) return MPI_CHAR;
    return MPI_DATATYPE_NULL;
}

struct Borders {
    std::vector<std::vector<Rank>> neighbors;
    std::vector<Index> bordering_cells;
};

// Get the how much data I have to import from other processing elements
// This function is a synchronization point.
std::vector<int> get_invert_list(const std::vector<int>& sends_to_procs, int* num_found, MPI_Comm comm);

template<class T>
typename std::vector<T>::const_iterator do_migration(int nb_elements, std::vector<T>& data, std::vector<std::vector<T>>& data_to_migrate, MPI_Datatype datatype, MPI_Comm comm) {
    int wsize, PE, nb_import;	
    MPI_Comm_size(comm, & wsize);	
    //export
    std::vector<int> export_counts(wsize), export_displs(wsize, 0);
    std::transform(data_to_migrate.cbegin(), data_to_migrate.cend(), std::begin(export_counts), [](const auto& el){return el.size();});
    for(PE = 1; PE < wsize; ++PE) export_displs[PE] = export_displs[PE - 1] + export_counts[PE - 1];
    auto nb_export = std::accumulate(export_counts.cbegin(), export_counts.cend(), 0);
    std::vector<T> export_buf; export_buf.reserve(nb_export);
    for(const auto& migration_buf : data_to_migrate)
        export_buf.insert(export_buf.end(), migration_buf.cbegin(), migration_buf.cend());
    // import
    std::vector<int> import_counts = get_invert_list(export_counts, &nb_import, comm), import_displs(wsize, 0);
    for(PE = 1; PE < wsize; ++PE) import_displs[PE] = import_displs[PE - 1] + import_counts[PE - 1];
    data.reserve(nb_elements + nb_import);
    std::vector<T> import_buf(nb_import);
    MPI_Alltoallv(export_buf.data(), export_counts.data(), export_displs.data(), datatype,
                  import_buf.data(), import_counts.data(), import_displs.data(), datatype, comm);
    std::move(import_buf.begin(), import_buf.end(), std::back_inserter(data));
    return std::next(data.begin(), nb_elements + nb_import);
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
    if(wsize == 1) return data.cend();
    auto nb_elements = data.size();

    std::vector< std::vector<T> > data_to_migrate(wsize);
    std::for_each(data_to_migrate.begin(), data_to_migrate.end(),
                  [size = nb_elements, wsize](auto &buf) { buf.reserve(size / wsize); });

    size_t data_id = 0;
    int PE, num_known = 0;

    while (data_id < nb_elements) {
        pointAssignFunc(LB, &data.at(data_id), &PE);
        if (PE != caller_rank) {
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
    return do_migration<T>(nb_elements, data, data_to_migrate, datatype, LB_COMM);
}

template<class T>
inline std::vector<int> gather_elements_on(const int world_size,
                               const int my_rank,
                               const int nb_elements,
                               const std::vector<T> &local_el,
                               const int dest_rank,
                               std::vector<T> &dest_el,
                               const MPI_Datatype &sendtype,
                               const MPI_Comm &comm) {
    int nlocal = local_el.size();
    std::vector<int> el_rank(nlocal, my_rank);
    std::vector<int> counts(world_size, 0), displs(world_size, 0);
    MPI_Gather(&nlocal, 1, MPI_INT, &counts.front(), 1, MPI_INT, dest_rank, comm);
    for (int cpt = 0; cpt < world_size; ++cpt) displs[cpt] = cpt == 0 ? 0 : displs[cpt - 1] + counts[cpt - 1];
    if (my_rank == dest_rank) dest_el.resize(nb_elements);
    if (my_rank == dest_rank) el_rank.resize(nb_elements);
    std::vector<int> all_el_rank(nb_elements);
    MPI_Gatherv(&local_el.front(), nlocal, sendtype,
                &dest_el.front(),
                &counts.front(),
                &displs.front(), sendtype, dest_rank, comm);
    MPI_Gatherv(&el_rank.front(), nlocal, MPI_INT,
                &all_el_rank.front(),
                &counts.front(),
                &displs.front(), MPI_INT, dest_rank, comm);
    return all_el_rank;
}

template<int N, class T, class LB, class GetPosFunc, class BoxIntersectFunc>
std::vector<T> get_ghost_data(
        LB* lb,
        std::vector<T>& data,
        GetPosFunc getPosPtrFunc,
        BoxIntersectFunc boxIntersect,
        Real rc,
        MPI_Datatype datatype,
        MPI_Comm LB_COMM) {

    const auto nb_elements = data.size();
    int wsize, caller_rank;
    MPI_Comm_size(LB_COMM, &wsize);
    MPI_Comm_rank(LB_COMM, &caller_rank);

    if (wsize == 1) return {};
    int num_found;

    std::array<double, 3> pos_in_double;
    std::vector<std::vector<T*> > data_to_migrate(wsize);
    std::for_each(data_to_migrate.begin(), data_to_migrate.end(),
            [size = nb_elements, wsize](auto &buf) { buf.reserve(size / wsize); });
    double radius = CUTOFF_RADIUS_FACTOR * rc;
    std::vector<int> PEs(wsize);
    for (size_t i = 0; i < nb_elements; ++i) {
        T* ptr_el = &data.at(i);
        put_in_3d_double_array<N>(pos_in_double, *getPosPtrFunc(ptr_el));
        boxIntersect(lb,
                     pos_in_double.at(0) - radius, pos_in_double.at(1) - radius, pos_in_double.at(2) - radius,
                     pos_in_double.at(0) + radius, pos_in_double.at(1) + radius, pos_in_double.at(2) + radius,
                     &PEs.front(), &num_found);
        for (int j = 0; j < num_found; ++j) {
            if(PEs[j] != caller_rank) data_to_migrate.at(PEs[j]).push_back(ptr_el);
        }
    }

    // build exportation lists
    std::vector<int> export_counts(wsize), export_displs(wsize, 0);
    std::transform(data_to_migrate.cbegin(), data_to_migrate.cend(), std::begin(export_counts), [](const auto& el){return el.size();});
    const int nb_export = std::accumulate(export_counts.cbegin(), export_counts.cend(), 0);
    for (int PE = 1; PE < wsize; ++PE) export_displs[PE] = export_displs[PE - 1] + export_counts[PE - 1];
    std::vector<T>   export_buf;
    export_buf.reserve(nb_export);
    for(const auto& migration_buf : data_to_migrate) {
        std::transform(migration_buf.cbegin(), migration_buf.cend(), std::back_inserter(export_buf), [](T* ptr_el){return *ptr_el;});
    }

    // build importation lists
    int nb_import;
    std::vector<int> import_counts = get_invert_list(export_counts, &nb_import, LB_COMM), import_displs(wsize, 0);
    for (int PE = 1; PE < wsize; ++PE) import_displs[PE] = import_displs[PE - 1] + import_counts[PE - 1];
    std::vector<T>   import_buf;
    import_buf.reserve(nb_import);
    MPI_Alltoallv(export_buf.data(), export_counts.data(), export_displs.data(), datatype,
                  import_buf.data(), import_counts.data(), import_displs.data(), datatype, LB_COMM);
    return import_buf;

}

template<int N, class T, class LB, class BoxIntersectFunc>
std::vector<T> retrieve_ghosts(
        LB* lb,
        std::vector<T>& data,
        const BoundingBox<N>& bbox,
        BoxIntersectFunc boxIntersect,
        Real rc,
        MPI_Datatype datatype,
        MPI_Comm LB_COMM) {

    const auto nb_elements = data.size();
    int wsize, caller_rank;
    MPI_Comm_size(LB_COMM, &wsize);
    MPI_Comm_rank(LB_COMM, &caller_rank);

    if (wsize == 1) return {};
    int num_found;

    std::array<double, 3> pos_in_double;
    std::vector<std::vector<T*> > data_to_migrate(wsize);
    std::for_each(data_to_migrate.begin(), data_to_migrate.end(),
                  [size = nb_elements, wsize](auto &buf) { buf.reserve(size / wsize); });
    double radius = CUTOFF_RADIUS_FACTOR * rc;
    std::vector<int> PEs(wsize);

    if constexpr(N == 3)
        boxIntersect(lb, bbox.at(0) - radius, bbox.at(2) - radius, bbox.at(4) - radius, bbox.at(1) + radius, bbox.at(3) + radius, bbox.at(5) + radius, &PEs.front(), &num_found);
    else
        boxIntersect(lb, bbox.at(0) - radius, bbox.at(2) - radius, 0.0, bbox.at(1) + radius, bbox.at(3) + radius, 0.0, &PEs.front(), &num_found);

    // build exportation lists
    std::vector<int> export_counts(wsize), export_displs(wsize, 0);
    auto nb_external_procs = std::count_if(PEs.cbegin(), PEs.cend(), [caller_rank](auto pe){return pe != caller_rank;});
    const int nb_export = nb_elements * nb_external_procs;

    std::vector<T>     export_buf;
    export_buf.reserve(nb_export);

    for (int j = 0; j < num_found; ++j) {
        const auto PE = PEs[j];
        if(PE != caller_rank) {
            export_counts.at(PE) = nb_elements;
            export_buf.insert(export_buf.end(), data.begin(), data.end());
        }
    }

    for (int PE = 1; PE < wsize; ++PE)
        export_displs[PE] = export_displs[PE - 1] + export_counts[PE - 1];

    // build importation lists
    int nb_import;
    std::vector<int> import_counts = get_invert_list(export_counts, &nb_import, LB_COMM), import_displs(wsize, 0);
    for (int PE = 1; PE < wsize; ++PE) import_displs[PE] = import_displs[PE - 1] + import_counts[PE - 1];
    std::vector<T> import_buf(nb_import);
    MPI_Alltoallv(export_buf.data(), export_counts.data(), export_displs.data(), datatype,
                  import_buf.data(), import_counts.data(), import_displs.data(), datatype, LB_COMM);

    return import_buf;

}

template<int N, class T, class LB, class BoxIntersectFunc>
std::vector<T> retrieve_ghosts(
        LB* lb,
        std::vector<T>& data,
        const BoundingBox<N>& bbox,
        BoxIntersectFunc& boxIntersect,
        Real rc,
        const std::vector<Integer>& head,
        const std::vector<Integer>& lscl,
        MPI_Datatype datatype,
        MPI_Comm LB_COMM,
        int* n_neighbors) {

    const auto nb_elements = data.size();
    int wsize, caller_rank;
    MPI_Comm_size(LB_COMM, &wsize);
    MPI_Comm_rank(LB_COMM, &caller_rank);

    if (wsize == 1) return {};
    int num_found;

    std::array<double, 3> pos_in_double{};
    std::vector<std::vector<T*> > data_to_migrate(wsize);
    std::for_each(data_to_migrate.begin(), data_to_migrate.end(),
                  [size = nb_elements, wsize](auto &buf) { buf.reserve(size / wsize); });

    double radius = 2.0*rc;

    std::vector<int> PEs(wsize);

    auto lc = get_cell_number_by_dimension<N>(bbox, rc);
    const auto ncells = head.size();
    for (auto c = 0; c < ncells; ++c) {
        auto j = head.at(c);
        if(j != EMPTY) {
            auto[x, y, z] = CoordinateTranslater::translate_linear_index_into_xyz(c, lc[0], lc[1]);
            auto[px,py,pz]= CoordinateTranslater::translate_local_xyz_into_position<N>({x,y,z}, bbox, rc);
            boxIntersect(lb, px - radius, py - radius, pz - radius, px + radius, py + radius, pz + radius, &PEs.front(), &num_found);
            while(j != EMPTY && num_found) {
                for (unsigned k = 0; k < num_found; ++k) {
                    data_to_migrate.at(PEs.at(k)).push_back(&data.at(j));
                }
                j = lscl.at(j);
            }
        }
    }

    // build exportation lists
    std::vector<int> export_counts(wsize), export_displs(wsize, 0);

    auto nb_external_procs = std::count_if(PEs.cbegin(), PEs.cend(), [caller_rank](auto pe){return pe != caller_rank;});

    const int nb_export = std::accumulate(data_to_migrate.cbegin(), data_to_migrate.cend(), 0,
                                          [](auto prev, const auto& mlist){return prev + mlist.size();});

    std::vector<T>   export_buf;
    export_buf.reserve(nb_export);

    for (unsigned PE = 0; PE < wsize; ++PE) {
        //const auto PE = PEs[j];
        if(PE != caller_rank) {
            export_counts.at(PE) = data_to_migrate.at(PE).size();
            for(auto beg = data_to_migrate.at(PE).begin(); beg != data_to_migrate.at(PE).end(); beg++){
                export_buf.push_back(**beg);
            }
        }
    }
    for (int PE = 1; PE < wsize; ++PE)
        export_displs[PE] = export_displs[PE - 1] + export_counts[PE - 1];

    // build importation lists
    int nb_import;
    std::vector<int> import_counts = get_invert_list(export_counts, &nb_import, LB_COMM), import_displs(wsize, 0);
    for (int PE = 1; PE < wsize; ++PE) import_displs[PE] = import_displs[PE - 1] + import_counts[PE - 1];

    auto neighbor_map = export_counts;
    for(auto i = 0; i < wsize; ++i) {
        neighbor_map.at(i) += import_counts.at(i);
    }
    neighbor_map.at(caller_rank) = 0;
    *n_neighbors = std::accumulate(neighbor_map.cbegin(), neighbor_map.cend(), 0, [](auto prev, auto v){return prev + (v > 1 ? 1 : v);});

    std::vector<T> import_buf(nb_import);
    MPI_Alltoallv(export_buf.data(), export_counts.data(), export_displs.data(), datatype,
                  import_buf.data(), import_counts.data(), import_displs.data(), datatype, LB_COMM);
    MPI_Allreduce(MPI_IN_PLACE, n_neighbors, 1, MPI_INT, MPI_MAX, LB_COMM);
    return import_buf;

}

#endif //NBMPI_PARALLEL_UTILS_HPP
