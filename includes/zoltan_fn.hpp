//
// Created by xetql on 02.03.18.
//

#ifndef NBMPI_ZOLTAN_FN_HPP
#define NBMPI_ZOLTAN_FN_HPP

#include "spatial_elements.hpp"
#include "params.hpp"

#include <cassert>
#include <random>
#include <string>
#include <vector>
#include <zoltan.h>
#include <set>

#define ENABLE_AUTOMATIC_MIGRATION true
template<int N>
std::array<double, N> get_as_double_array(const std::array<Real, N>& real_array){
    if constexpr(N==2)
        return {(double) real_array[0], (double) real_array[1]};
    else
        return {(double) real_array[0], (double) real_array[1], (double) real_array[2]};
}

using Rank = int;
using Index = Integer;
auto MPI_INDEX = MPI_LONG_LONG;

struct Borders {
    std::vector<std::vector<Rank>> neighbors;
    std::vector<Index> bordering_cells;
};

template<int N>
Borders __get_border_cells_index(Zoltan_Struct* load_balancer, const BoundingBox<N>& bbox, const Real rc) {
    int caller_rank, wsize, num_found;
    MPI_Comm_rank(MPI_COMM_WORLD, &caller_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &wsize) ;
    if (wsize == 1) return {};
    auto lc = get_cell_number_by_dimension<N>(bbox, rc);

    std::vector<Index> bordering_cell_index;
    // number of bordering cell in 3D is xyz - (x-2)(y-2)(z-2)
    int nb_local_cells     = std::accumulate(lc.cbegin(), lc.cend(), 1, [](auto p, auto v){return p*v;});
    int nb_bordering_cells = std::min(nb_local_cells, nb_local_cells - std::accumulate(lc.cbegin(), lc.cend(), 1, [](auto p, auto v){return p * (v-2);}));

    // If I have no bordering cell (impossible, but we never know (: )
    if(nb_bordering_cells <= 0) return {};

    bordering_cell_index.reserve(nb_bordering_cells);

    std::vector<Rank> PEs(wsize);
    std::vector< std::vector<Rank> > neighbors(nb_bordering_cells);

    Integer border_cell_cnt = 0;
    for(Index cell_id = 0; cell_id < nb_local_cells; ++cell_id) {
        auto[x, y, z] = CoordinateTranslater::translate_linear_index_into_xyz(cell_id, lc[0], lc[1]);
        if(x == 0 || y == 0 || z == 0) {
            std::array<double, N> pos_in_double =
                    get_as_double_array<N>(CoordinateTranslater::translate_local_index_into_position<N>(cell_id, bbox, rc));
            if constexpr (N == 3) {
                Zoltan_LB_Box_Assign(load_balancer,
                                     pos_in_double.at(0) + rc/2.0 - rc, pos_in_double.at(1) + rc/2.0 - rc,
                                     pos_in_double.at(2) + rc/2.0 - rc, pos_in_double.at(0) + rc/2.0 + rc,
                                     pos_in_double.at(1) + rc/2.0 + rc, pos_in_double.at(2) + rc/2.0 + rc,
                                     &PEs.front(), &num_found);
            } else {
                Zoltan_LB_Box_Assign(load_balancer,
                                     pos_in_double.at(0) + rc/2.0 - rc, pos_in_double.at(1) + rc/2.0 - rc,
                                     pos_in_double.at(2) + rc/2.0 - rc, pos_in_double.at(0) + rc/2.0 + rc,
                                     &PEs.front(), &num_found);
            }
            if(num_found){
                bordering_cell_index.push_back(cell_id);
                neighbors.at(border_cell_cnt).assign(PEs.begin(), PEs.begin() + num_found);
                border_cell_cnt++;
            }
        }
    }
    return {neighbors, bordering_cell_index};
}

template<int N>
Borders get_border_cells_index(Zoltan_Struct* load_balancer, const BoundingBox<N>& bbox, const Real rc) {
    int caller_rank, wsize, num_found;
    MPI_Comm_rank(MPI_COMM_WORLD, &caller_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &wsize) ;
    if (wsize == 1) return {};
    auto lc = get_cell_number_by_dimension<N>(bbox, rc);

    std::vector<Index> bordering_cell_index;
    // number of bordering cell in 3D is xyz - (x-2)(y-2)(z-2)
    int nb_local_cells     = std::accumulate(lc.cbegin(), lc.cend(), 1, [](auto p, auto v){return p*v;});
    int nb_bordering_cells = std::min(nb_local_cells, nb_local_cells - std::accumulate(lc.cbegin(), lc.cend(), 1, [](auto p, auto v){return p * (v-2);}));

    // If I have no bordering cell (impossible, but we never know (: )
    if(nb_bordering_cells <= 0) return {};

    bordering_cell_index.reserve(nb_bordering_cells);

    std::vector<Rank> PEs(wsize);
    std::vector< std::vector<Rank> > neighbors(nb_bordering_cells);

    Integer border_cell_cnt = 0;
    for(Integer z = 0; z < lc[2]; ++z){
        for(Integer y = 0; y < lc[1]; ++y){
            Integer condition = !(y^0)|!(y^(lc[1]-1)) | !(z^0)|!(z^(lc[2]-1));
            for(Integer x = 0; x < lc[0]; x += bitselect(condition, (Integer) 1, lc[0] - 1)) {
                Index cell_id = CoordinateTranslater::translate_xyz_into_linear_index<N>({x,y,z}, bbox, rc);
                std::array<double, N> pos_in_double =
                        get_as_double_array<N>(CoordinateTranslater::translate_local_index_into_position<N>(cell_id, bbox, rc));
                if constexpr (N == 3) {
                    Zoltan_LB_Box_Assign(load_balancer,
                                         pos_in_double.at(0) + rc/2.0 - rc, pos_in_double.at(1) + rc/2.0 - rc,
                                         pos_in_double.at(2) + rc/2.0 - rc, pos_in_double.at(0) + rc/2.0 + rc,
                                         pos_in_double.at(1) + rc/2.0 + rc, pos_in_double.at(2) + rc/2.0 + rc,
                                         &PEs.front(), &num_found);
                } else {
                    Zoltan_LB_Box_Assign(load_balancer,
                                         pos_in_double.at(0) + rc/2.0 - rc, pos_in_double.at(1) + rc/2.0 - rc,
                                         pos_in_double.at(2) + rc/2.0 - rc, pos_in_double.at(0) + rc/2.0 + rc,
                                         &PEs.front(), &num_found);
                }
                if(num_found){
                    bordering_cell_index.push_back(cell_id);
                    neighbors.at(border_cell_cnt).assign(PEs.begin(), PEs.begin() + num_found);
                    border_cell_cnt++;
                }
            }
        }
    }
    return {neighbors, bordering_cell_index};
}

template<int N>
void init_mesh_data(int rank, int nprocs, MESH_DATA<N>& mesh_data, sim_param_t* params) {

    if (rank == 0) {
        double min_r2 = params->sig_lj*params->sig_lj;

        //std::random_device rd; //Will be used to obtain a seed for the random number engine
        std::mt19937 gen(params->seed); //Standard mersenne_twister_engine seeded with rd()
        std::uniform_real_distribution<elements::ElementRealType> udist(0.0, params->simsize);
        //std::normal_distribution<double> ndist(params->simsize/2, 0.5);

        std::vector<elements::Element<N>> elements(params->npart);

        elements::Element<N>::create_random_n(elements, udist, gen, [=](auto point, auto other){
            return elements::distance2<N>(point, other) >= min_r2;
        });

        elements::init_particles_random_v(elements, params->T0);
        mesh_data.els = elements;
    } else {}
}

template<int N>
int get_number_of_objects(void *data, int *ierr) {
    MESH_DATA<N> *mesh= (MESH_DATA<N> *)data;
    *ierr = ZOLTAN_OK;
    return mesh->els.size();
}

template<int N>
void get_object_list(void *data, int sizeGID, int sizeLID,
                     ZOLTAN_ID_PTR globalID, ZOLTAN_ID_PTR localID,
                     int wgt_dim, float *obj_wgts, int *ierr) {
    size_t i;
    MESH_DATA<N> *mesh= (MESH_DATA<N> *)data;
    *ierr = ZOLTAN_OK;
    /* In this example, return the IDs of our objects, but no weights.
     * Zoltan will assume equally weighted objects.
     */
    for (i=0; i < mesh->els.size(); i++){
        globalID[i] = mesh->els[i].gid;
        localID[i] = i;
    }
}
template<int N>
int get_num_geometry(void *data, int *ierr) {
    *ierr = ZOLTAN_OK;
    return N;
} 

template<int N>
void get_geometry_list(void *data, int sizeGID, int sizeLID,
                       int num_obj,
                       ZOLTAN_ID_PTR globalID, ZOLTAN_ID_PTR localID,
                       int num_dim, double *geom_vec, int *ierr) {
    int i;

    MESH_DATA<N> *mesh= (MESH_DATA<N> *)data;

    if ( (sizeGID != 1) || (sizeLID != 1) || (num_dim > 3)){
        *ierr = ZOLTAN_FATAL;
        return;
    }

    *ierr = ZOLTAN_OK;

    for (i=0;  i < num_obj; i++){
        geom_vec[N * i] = mesh->els[i].position.at(0);
        geom_vec[N * i + 1] = mesh->els[i].position.at(1);

        if(N == 3) geom_vec[N * i + 2] = mesh->els[i].position.at(2);
    }
}



Zoltan_Struct* zoltan_create_wrapper(bool automatic_migration, MPI_Comm comm, int num_global_part = -1, int part_on_me = -1) {
    std::string ngp = std::to_string(num_global_part);
    std::string pom = std::to_string(part_on_me);

    auto zz = Zoltan_Create(MPI_COMM_WORLD);

    Zoltan_Set_Param(zz, "DEBUG_LEVEL", "0");
    Zoltan_Set_Param(zz, "LB_METHOD", "RCB");
    Zoltan_Set_Param(zz, "DETERMINISTIC", "1");
    Zoltan_Set_Param(zz, "NUM_GID_ENTRIES", "1");

    if(num_global_part >= 1) Zoltan_Set_Param(zz, "NUM_GLOBAL_PARTS", ngp.c_str());
    if(part_on_me >= 1) Zoltan_Set_Param(zz, "NUM_LOCAL_PARTS",  pom.c_str());

    Zoltan_Set_Param(zz, "NUM_LID_ENTRIES", "1");
    Zoltan_Set_Param(zz, "OBJ_WEIGHT_DIM", "0");
    Zoltan_Set_Param(zz, "RETURN_LISTS", "ALL");

    Zoltan_Set_Param(zz, "RCB_OUTPUT_LEVEL", "0");
    Zoltan_Set_Param(zz, "RCB_RECTILINEAR_BLOCKS", "1");
    Zoltan_Set_Param(zz, "KEEP_CUTS", "1");

    if(automatic_migration)
        Zoltan_Set_Param(zz, "AUTO_MIGRATE", "TRUE");

    return zz;
}

Zoltan_Struct* zoltan_create_wrapper(bool automatic_migration = false) {
    return zoltan_create_wrapper(automatic_migration, MPI_COMM_WORLD);
}

template<int N>
int cpt_obj_size( void *data,
                  int num_gid_entries,
                  int num_lid_entries,
                  ZOLTAN_ID_PTR global_id,
                  ZOLTAN_ID_PTR local_id,
                  int *ierr) {
    ierr = ZOLTAN_OK;
    return sizeof(int) * 2 + sizeof(elements::ElementRealType) * N * 3 /*pos, vel, acc*/;
}

template<int N>
void pack_particles(void *data,
                    int num_gid_entries,
                    int num_lid_entries,
                    ZOLTAN_ID_PTR global_id,
                    ZOLTAN_ID_PTR local_id,
                    int dest,
                    int size,
                    char *buf,
                    int *ierr) {
    auto all_mesh_data = (MESH_DATA<N>*) data;
    memcpy(buf, &(all_mesh_data->els[(int)(*local_id)]), sizeof(class elements::Element<N>));
    all_mesh_data->els[(int)(*local_id)].gid = -1;
}

template<int N>
void unpack_particles ( void *data,
                        int num_gid_entries,
                        ZOLTAN_ID_PTR global_id,
                        int size,
                        char *buf,
                        int *ierr) {
    auto all_mesh_data = (MESH_DATA<N>*) data;
    elements::Element<N> e;
    memcpy(&e, buf, sizeof(int) * 2 + sizeof(elements::ElementRealType) * N * 3);
    all_mesh_data->els.push_back(e);
}

template<int N>
void post_migrate_particles (
        void *data,
        int num_gid_entries, int num_lid_entries, int num_import,
        ZOLTAN_ID_PTR import_global_ids, ZOLTAN_ID_PTR import_local_ids,
        int *import_procs, int num_export,
        ZOLTAN_ID_PTR export_global_ids, ZOLTAN_ID_PTR export_local_ids,
        int *export_procs, int *ierr) {
    auto all_mesh_data = (MESH_DATA<N>*) data;
    size_t size = all_mesh_data->els.size();
    size_t i = 0;
    while(i < size) {
        if(all_mesh_data->els[i].gid == -1){
            std::iter_swap(all_mesh_data->els.begin() + i, all_mesh_data->els.end() - 1);
            all_mesh_data->els.pop_back();
            size--;
        } else {
            i++;
        }
    }
    size = all_mesh_data->els.size();
    for(size_t i = 0; i < size; i++){
        all_mesh_data->els[i].lid = i;
    }
}

template<int N>
void zoltan_fn_init(Zoltan_Struct* zz, MESH_DATA<N>* mesh_data, bool automatic_migration = false){
    Zoltan_Set_Num_Obj_Fn(   zz, get_number_of_objects<N>, mesh_data);
    Zoltan_Set_Obj_List_Fn(  zz, get_object_list<N>,       mesh_data);
    Zoltan_Set_Num_Geom_Fn(  zz, get_num_geometry<N>,      mesh_data);
    Zoltan_Set_Geom_Multi_Fn(zz, get_geometry_list<N>,     mesh_data);
    if(automatic_migration) {
        Zoltan_Set_Obj_Size_Fn(zz, cpt_obj_size<N>, mesh_data);
        Zoltan_Set_Pack_Obj_Fn(zz, pack_particles<N>, mesh_data);
        Zoltan_Set_Unpack_Obj_Fn(zz, unpack_particles<N>, mesh_data);
        Zoltan_Set_Post_Migrate_Fn(zz, post_migrate_particles<N>, mesh_data);
    }
}

template<typename T>
inline T dto(double v) {
    T ret = (T) v;

    if(std::isinf(ret)){
        if(ret == -INFINITY){
            ret = std::numeric_limits<T>::lowest();
        } else {
            ret = std::numeric_limits<T>::max();
        }
    }

    return ret;
}


template<int N>
typename std::vector<elements::Element<N>>::const_iterator zoltan_migrate_particles(
        std::vector<elements::Element<N>> &data,
        Zoltan_Struct *load_balancer,
        const Integer* head,
        const Integer* lscl,
        const Borders& bordering_cells,
        MPI_Datatype datatype,
        MPI_Comm LB_COMM)
{

    int wsize;
    MPI_Comm_size(LB_COMM, &wsize);
    int caller_rank;
    MPI_Comm_rank(LB_COMM, &caller_rank);

    if(wsize == 1) return data.cend();

    auto nb_elements = data.size();
    const auto prev_size = data.size();
    std::vector< std::vector<elements::Element<N>> > data_to_migrate(wsize);
    std::for_each(data_to_migrate.begin(), data_to_migrate.end(),
                  [size = nb_elements, wsize](auto &buf) { buf.reserve(size / wsize); });
    size_t data_id = 0;

    int PE, num_known = 0;
    ZOLTAN_ID_PTR found_gids, found_lids;
    int *found_procs, *found_parts, num_found;

    std::vector<int> export_gids, export_lids, export_procs;
    int cell_cnt = 0;
    for(auto cidx : bordering_cells.bordering_cells) {
        auto p = head[cidx];
        while(p != -1) {
            num_known ++;
            p = lscl[p];
        }
        cell_cnt++;
    }
    export_gids.reserve(num_known);
    export_lids.reserve(num_known);
    export_procs.reserve(num_known);
    cell_cnt = 0;
    num_known = 0;
    for(auto cidx : bordering_cells.bordering_cells){
        auto p = head[cidx];
        while(p != -1) {
            if(p < nb_elements) {
                const elements::Element<N>& el = data.at(p);
                auto pos_in_double = get_as_double_array<N>(el.position);
                Zoltan_LB_Point_Assign(load_balancer, &pos_in_double.front(), &PE);
                if(PE != caller_rank){
                    export_gids.push_back(el.gid);
                    export_lids.push_back(p);
                    export_procs.push_back(PE);
                    data_to_migrate.at(PE).push_back(el);
                    num_known ++;
                }
            }
            p = lscl[p];
        }
        cell_cnt++;
    }
    export_gids.shrink_to_fit();
    export_lids.shrink_to_fit();
    export_procs.shrink_to_fit();

    std::sort(export_lids.begin(), export_lids.end(), std::greater<int>());

    for(auto lid : export_lids) {
        std::iter_swap(data.begin()+lid, data.end()-1);
        data.pop_back();
    }

    ZOLTAN_ID_PTR known_gids = (ZOLTAN_ID_PTR) export_gids.data();
    ZOLTAN_ID_PTR known_lids = (ZOLTAN_ID_PTR) export_lids.data();

    Zoltan_Invert_Lists(load_balancer, export_procs.size(), known_gids, known_lids, export_procs.data(), export_procs.data(),
                        &num_found, &found_gids, &found_lids, &found_procs, &found_parts);

    data.reserve(data.size() + num_found);

    std::set<int> import_from_procs(found_procs, found_procs+num_found);

    /* Let's Migrate ma boi ! */
    if(num_found)
        Zoltan_LB_Free_Part(&found_gids, &found_lids, &found_procs, &found_parts);

    int nb_reqs = std::count_if(data_to_migrate.cbegin(), data_to_migrate.cend(), [](const auto& buf){return !buf.empty();});

    int cpt = 0;

    std::vector<MPI_Request> reqs(nb_reqs);
    for (size_t PE = 0; PE < wsize; PE++) {
        int send_size = data_to_migrate.at(PE).size();
        if (send_size) {
            MPI_Isend(&data_to_migrate.at(PE).front(), send_size, datatype, PE, 300, LB_COMM, &reqs[cpt]);
            cpt++;
        }
    }

    std::vector<elements::Element<N>> buffer;

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
    for(int i = 0; i < nb_data; ++i) data.at(i).lid = i;
    data.shrink_to_fit();

    MPI_Waitall(reqs.size(), &reqs.front(), MPI_STATUSES_IGNORE);
    return std::next(data.cbegin(), nb_data - prev_size);
}

template<int N>
typename std::vector<elements::Element<N>>::const_iterator Zoltan_Migrate_Particles(
        std::vector<elements::Element<N>> &data,
        Zoltan_Struct *load_balancer,
        MPI_Datatype datatype,
        MPI_Comm LB_COMM) {
    int wsize;
    MPI_Comm_size(LB_COMM, &wsize);
    int caller_rank;
    MPI_Comm_rank(LB_COMM, &caller_rank);
    const auto prev_size = data.size();
    if(wsize == 1) return data.cend();
    auto nb_elements = data.size();

    std::vector< std::vector<elements::Element<N>> > data_to_migrate(wsize);
    std::for_each(data_to_migrate.begin(), data_to_migrate.end(),
                  [size = nb_elements, wsize](auto &buf) { buf.reserve(size / wsize); });

    size_t data_id = 0;
    int PE, num_known = 0;
    ZOLTAN_ID_PTR found_gids, found_lids;
    int *found_procs, *found_parts, num_found;

    {
        std::vector<int> export_gids, export_lids, export_procs;
        export_gids.reserve(nb_elements / wsize);
        export_lids.reserve(nb_elements / wsize);
        export_procs.reserve(nb_elements / wsize);

        while (data_id < nb_elements) {
            auto pos_in_double = get_as_double_array<N>(data.at(data_id).position);
            Zoltan_LB_Point_Assign(load_balancer, &pos_in_double.front(), &PE);
            if (PE != caller_rank) {
                export_gids.push_back(data.at(data_id).gid);
                export_lids.push_back(data.at(data_id).lid);
                export_procs.push_back(PE);
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
         export_gids.shrink_to_fit();
         export_lids.shrink_to_fit();
        export_procs.shrink_to_fit();

        ZOLTAN_ID_PTR known_gids = (ZOLTAN_ID_PTR) &export_gids.front();
        ZOLTAN_ID_PTR known_lids = (ZOLTAN_ID_PTR) &export_lids.front();

        Zoltan_Invert_Lists(load_balancer, num_known, known_gids, known_lids, &export_procs[0], &export_procs[0],
                                       &num_found, &found_gids, &found_lids, &found_procs, &found_parts);
    }

    std::set<int> import_from_procs(found_procs, found_procs+num_found);

    data.reserve(nb_elements + num_found);

    /* Let's Migrate ma boi ! */

    if(num_found > 0)
        Zoltan_LB_Free_Part(&found_gids, &found_lids, &found_procs, &found_parts);

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

    std::vector<elements::Element<N>> buffer;

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
    for(int i = 0; i < nb_data; ++i) data[i].lid = i;

    MPI_Waitall(reqs.size(), &reqs.front(), MPI_STATUSES_IGNORE);

    return std::next(data.begin(), nb_data - prev_size);
}


template<int N>
std::vector<elements::Element<N>> zoltan_exchange_data(
        const std::vector<elements::Element<N>> &data,
        Zoltan_Struct *load_balancer,
        const Integer* head,
        const Integer* lscl,
        const Borders& bordering_cells,
        MPI_Datatype datatype,
        MPI_Comm LB_COMM,
        int &nb_elements_recv,
        int &nb_elements_sent) {

    const auto nb_elements = data.size();

    int wsize, caller_rank;
    MPI_Comm_size(LB_COMM, &wsize);
    MPI_Comm_rank(LB_COMM, &caller_rank);

    std::vector<elements::Element<N>> buffer;
    std::vector<elements::Element<N>> remote_data_gathered;

    if (wsize == 1)
        return remote_data_gathered;

    std::vector<std::vector<elements::Element<N> > > data_to_migrate(wsize);
    std::for_each(data_to_migrate.begin(), data_to_migrate.end(),
                  [size = nb_elements, wsize](auto &buf) { buf.reserve(size / wsize); });

    int num_found, num_known = 0;
    ZOLTAN_ID_PTR found_gids, found_lids;
    int *found_procs, *found_parts;

    std::vector<int> export_gids, export_lids, export_procs;
    int cell_cnt = 0;
    for(auto cidx : bordering_cells.bordering_cells) {
        auto p = head[cidx];
        while(p != -1) {
            num_known += bordering_cells.neighbors.at(cell_cnt).size();
            p = lscl[p];
        }
        cell_cnt++;
    }
     export_gids.resize(num_known);
     export_lids.resize(num_known);
    export_procs.resize(num_known);

    cell_cnt = 0;
    num_known = 0;
    for(auto cidx : bordering_cells.bordering_cells){
        auto p = head[cidx];
        while(p != -1){
            for(auto rank : bordering_cells.neighbors.at(cell_cnt)) {
                if(rank != caller_rank){
                    const auto& el = data[p];
                     export_gids[num_known] = el.gid;
                     export_lids[num_known] = el.lid;
                    export_procs[num_known] = rank;
                    data_to_migrate.at(rank).push_back(el);
                    num_known ++;
                }
            }
            p = lscl[p];
        }
        cell_cnt++;
    }

    auto known_gids = (ZOLTAN_ID_PTR) export_gids.data();
    auto known_lids = (ZOLTAN_ID_PTR) export_lids.data();

    // Compute who has to send me something via Zoltan.
    int err = Zoltan_Invert_Lists(load_balancer, num_known, known_gids, known_lids, export_procs.data(), export_procs.data(),
                        &num_found, &found_gids, &found_lids, &found_procs, &found_parts);

    std::set<int> import_from_procs(found_procs, found_procs+num_found);

    if(num_found)
        Zoltan_LB_Free_Part(&found_gids, &found_lids, &found_procs, &found_parts);

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

template<int N>
std::vector<elements::Element<N>> zoltan_exchange_data(
        std::vector<elements::Element<N>> &data,
        Zoltan_Struct *load_balancer,
        const CommunicationDatatype datatype,
        const MPI_Comm LB_COMM,
        int &nb_elements_recv,
        int &nb_elements_sent,
        double cell_size = 0.000625) {

    const auto nb_elements = data.size();

    int wsize, caller_rank;
    MPI_Comm_size(LB_COMM, &wsize);
    MPI_Comm_rank(LB_COMM, &caller_rank);

    std::vector<elements::Element<N>> buffer;
    std::vector<elements::Element<N>> remote_data_gathered;

    if (wsize == 1) return remote_data_gathered;

    std::vector<std::vector<elements::Element<N> > > data_to_migrate(wsize);
    std::for_each(data_to_migrate.begin(), data_to_migrate.end(),
                  [size = nb_elements, wsize](auto &buf) { buf.reserve(size / wsize); });

    int num_found, num_known = 0;
    ZOLTAN_ID_PTR found_gids, found_lids;
    int *found_procs, *found_parts;
    {
        std::vector<int> PEs(wsize, -1),
                export_gids, export_lids, export_procs;
        export_gids.reserve(nb_elements / wsize);
        export_lids.reserve(nb_elements / wsize);
        export_procs.reserve(nb_elements / wsize);
        size_t data_id = 0;
        while (data_id < nb_elements) {
            auto pos_in_double = get_as_double_array<N>(data.at(data_id).position);
            if constexpr (N == 3) {
                Zoltan_LB_Box_Assign(load_balancer,
                                     pos_in_double.at(0) - cell_size, pos_in_double.at(1) - cell_size,
                                     pos_in_double.at(2) - cell_size,
                                     pos_in_double.at(0) + cell_size, pos_in_double.at(1) + cell_size,
                                     pos_in_double.at(2) + cell_size,
                                     &PEs.front(), &num_found);
            } else {
                Zoltan_LB_Box_Assign(load_balancer,
                                     pos_in_double.at(0) - cell_size, pos_in_double.at(1) - cell_size, 0.0,
                                     pos_in_double.at(0) + cell_size, pos_in_double.at(1) + cell_size, 0.0,
                                     &PEs.front(), &num_found);
            }
            for (int PE_idx = 0; PE_idx < num_found; ++PE_idx) {
                int PE = PEs[PE_idx];
                if (PE != caller_rank) {
                    export_gids.push_back(data.at(data_id).gid);
                    export_lids.push_back(data.at(data_id).lid);
                    export_procs.push_back(PE);
                    data_to_migrate.at(PE).push_back(data.at(data_id));
                    num_known++;
                }
            }
            data_id++; //if the element must stay with me then check the next one
        }

        export_gids.shrink_to_fit();
        export_lids.shrink_to_fit();
        export_procs.shrink_to_fit();

        ZOLTAN_ID_PTR known_gids = (ZOLTAN_ID_PTR) export_gids.data();
        ZOLTAN_ID_PTR known_lids = (ZOLTAN_ID_PTR) export_lids.data();

        // Compute who has to send me something via Zoltan.
        Zoltan_Invert_Lists(load_balancer, num_known, known_gids, known_lids, export_procs.data(), export_procs.data(),
                            &num_found, &found_gids, &found_lids, &found_procs, &found_parts);
    }

    std::vector<int> num_import_from_procs(wsize, 0);
    std::vector<int> import_from_procs;

    if(num_found)
        import_from_procs.reserve(num_found);
    // Compute how many elements I have to import from others, and from whom.
    for (size_t i = 0; i < num_found; ++i) {
        num_import_from_procs[found_procs[i]]++;
        if (std::find(import_from_procs.begin(), import_from_procs.end(), found_procs[i]) == import_from_procs.end())
            import_from_procs.push_back(found_procs[i]);
    }

    Zoltan_LB_Free_Part(&found_gids, &found_lids, &found_procs, &found_parts);

    int nb_reqs = std::count_if(data_to_migrate.cbegin(), data_to_migrate.cend(), [](const auto& buf){return !buf.empty();});
    int cpt = 0;

    // Send the data to neighbors
    std::vector<MPI_Request> reqs(nb_reqs);
    nb_elements_sent = 0;
    for (size_t PE = 0; PE < wsize; PE++) {
        int send_size = data_to_migrate.at(PE).size();
        if (send_size) {
            nb_elements_sent += send_size;
            MPI_Isend(&data_to_migrate.at(PE).front(), send_size, datatype.elements_datatype, PE, 400, LB_COMM,
                      &reqs[cpt]);
            cpt++;
        }
    }

    // Import the data from neighbors
    nb_elements_recv = 0;
    remote_data_gathered.reserve(num_found);
    for (int proc_id : import_from_procs) {
        auto size = num_import_from_procs[proc_id];
        nb_elements_recv += size;
        buffer.resize(size);
        MPI_Recv(buffer.data(), size, datatype.elements_datatype, proc_id, 400, LB_COMM, MPI_STATUS_IGNORE);
        std::move(buffer.begin(), buffer.end(), std::back_inserter(remote_data_gathered));
    }
    MPI_Waitall(reqs.size(), &reqs.front(), MPI_STATUSES_IGNORE);
    return remote_data_gathered;
}


template<int N>
void migrate_zoltan(std::vector<elements::Element<N>> &data, int numImport, int numExport, int *exportProcs,
                    unsigned int *exportGlobalGids,
                    const CommunicationDatatype datatype,
                    const MPI_Comm LB_COMM) {
    int wsize;
    MPI_Comm_size(LB_COMM, &wsize);
    int caller_rank;
    MPI_Comm_rank(LB_COMM, &caller_rank);
    std::vector<elements::Element<N> > buffer;
    std::map<int, std::shared_ptr<std::vector<elements::Element<N> > > > data_to_migrate;

    for (int i = 0; i < numExport; ++i)
        if (data_to_migrate.find(exportProcs[i]) == data_to_migrate.end())
            data_to_migrate[exportProcs[i]] = std::make_shared<std::vector<elements::Element<N>>>();

    for (int i = 0; i < numExport; ++i) {
        auto PE = exportProcs[i];
        auto gid = exportGlobalGids[i];

        //check within the remaining elements which belong to the current PE
        size_t data_id = 0;
        while (data_id < data.size()) {
            if (gid == (size_t) data[data_id].gid) {
                //if the current element has to be moved, then swap with the last and pop it out (dont need to move the pointer also)
                //swap iterator values in constant time
                std::iter_swap(data.begin() + data_id, data.end() - 1);
                //get the value and push it in the "to migrate" vector
                data_to_migrate[PE]->push_back(*(data.end() - 1));
                //pop the head of the list in constant time
                data.pop_back();
            } else data_id++; //if the element must stay with me then check the next one
        }
    }

    std::vector<MPI_Request> reqs(data_to_migrate.size());

    int cpt = 0;
    for (auto const &pe_data : data_to_migrate) {
        int send_size = pe_data.second->size();
        MPI_Isend(&pe_data.second->front(), send_size, datatype.elements_datatype, pe_data.first, 300, LB_COMM,
                  &reqs[cpt]);
        cpt++;
    }
    int collectData = 0;

    while (collectData < numImport) {// receive the data in any order
        int source_rank, size;
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, 300, LB_COMM, &status);
        source_rank = status.MPI_SOURCE;
        MPI_Get_count(&status, datatype.elements_datatype, &size);
        collectData += size;
        buffer.resize(size);
        MPI_Recv(&buffer.front(), size, datatype.elements_datatype, source_rank, 300, LB_COMM, &status);
        std::move(buffer.begin(), buffer.end(), std::back_inserter(data));

    }
    MPI_Waitall(cpt, &reqs.front(), MPI_STATUSES_IGNORE);

}

template <int N>
void Zoltan_LB(MESH_DATA<N>* mesh_data, Zoltan_Struct* load_balancer) {

    // ZOLTAN VARIABLES
    int changes, numGidEntries, numLidEntries, numImport, numExport;
    ZOLTAN_ID_PTR importGlobalGids, importLocalGids, exportGlobalGids, exportLocalGids;
    int *importProcs, *importToPart, *exportProcs, *exportToPart;
    // END OF ZOLTAN VARIABLES

    zoltan_fn_init(load_balancer, mesh_data, ENABLE_AUTOMATIC_MIGRATION);
    Zoltan_LB_Partition(load_balancer,      /* input (all remaining fields are output) */
                        &changes,           /* 1 if partitioning was changed, 0 otherwise */
                        &numGidEntries,     /* Number of integers used for a global ID */
                        &numLidEntries,     /* Number of integers used for a local ID */
                        &numImport,         /* Number of vertices to be sent to me */
                        &importGlobalGids,  /* Global IDs of vertices to be sent to me */
                        &importLocalGids,   /* Local IDs of vertices to be sent to me */
                        &importProcs,       /* Process rank for source of each incoming vertex */
                        &importToPart,      /* New partition for each incoming vertex */
                        &numExport,         /* Number of vertices I must send to other processes*/
                        &exportGlobalGids,  /* Global IDs of the vertices I must send */
                        &exportLocalGids,   /* Local IDs of the vertices I must send */
                        &exportProcs,       /* Process to which I send each of the vertices */
                        &exportToPart);     /* Partition to which each vertex will belong */

    Zoltan_LB_Free_Part(&importGlobalGids, &importLocalGids, &importProcs, &importToPart);
    Zoltan_LB_Free_Part(&exportGlobalGids, &exportLocalGids, &exportProcs, &exportToPart);
}
#endif //NBMPI_ZOLTAN_FN_HPP
