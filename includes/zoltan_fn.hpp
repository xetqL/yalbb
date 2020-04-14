//
// Created by xetql on 02.03.18.
//

#ifndef NBMPI_ZOLTAN_FN_HPP
#define NBMPI_ZOLTAN_FN_HPP

#include "parallel_utils.hpp"
#include "params.hpp"
#include "spatial_elements.hpp"

#include <cassert>
#include <random>
#include <string>
#include <vector>
#include <zoltan.h>
#include <set>

#define ENABLE_AUTOMATIC_MIGRATION true

auto MPI_INDEX = MPI_LONG_LONG;

template<int N>
int get_number_of_objects(void *data, int *ierr) {
    MESH_DATA<elements::Element<N>> *mesh= (MESH_DATA<elements::Element<N>> *)data;
    *ierr = ZOLTAN_OK;
    return mesh->els.size();
}

template<int N>
void get_object_list(void *data, int sizeGID, int sizeLID,
                     ZOLTAN_ID_PTR globalID, ZOLTAN_ID_PTR localID,
                     int wgt_dim, float *obj_wgts, int *ierr) {
    size_t i;
    auto mesh= (MESH_DATA<elements::Element<N>> *)data;
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

    auto mesh= (MESH_DATA<elements::Element<N>> *)data;

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


    Zoltan_Set_Param(zz, "AUTO_MIGRATE", "TRUE");

    return zz;
}

Zoltan_Struct* zoltan_create_wrapper(MPI_Comm comm, bool automatic_migration = false) {
    return zoltan_create_wrapper(automatic_migration, comm);
}

template<int N>
int cpt_obj_size( void *data,
                  int num_gid_entries,
                  int num_lid_entries,
                  ZOLTAN_ID_PTR global_id,
                  ZOLTAN_ID_PTR local_id,
                  int *ierr) {
    ierr = ZOLTAN_OK;
    return sizeof(int) * 2 + sizeof(Real) * N * 3 /*pos, vel, acc*/;
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
    auto all_mesh_data = (MESH_DATA<elements::Element<N>>*) data;
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
    auto all_mesh_data = (MESH_DATA<elements::Element<N>>*) data;
    elements::Element<N> e;
    memcpy(&e, buf, /*gid, lid*/ sizeof(Integer) * 2 + /*position[x,y,(z)], velocity[x,y,(z)]*/sizeof(Real) * N * 2);
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
    auto all_mesh_data = (MESH_DATA<elements::Element<N>>*) data;
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
void zoltan_fn_init(Zoltan_Struct* zz, MESH_DATA<elements::Element<N>>* mesh_data){
    Zoltan_Set_Num_Obj_Fn(   zz, get_number_of_objects<N>, mesh_data);
    Zoltan_Set_Obj_List_Fn(  zz, get_object_list<N>,       mesh_data);
    Zoltan_Set_Num_Geom_Fn(  zz, get_num_geometry<N>,      mesh_data);
    Zoltan_Set_Geom_Multi_Fn(zz, get_geometry_list<N>,     mesh_data);
    Zoltan_Set_Obj_Size_Fn(zz, cpt_obj_size<N>, mesh_data);
    Zoltan_Set_Pack_Obj_Fn(zz, pack_particles<N>, mesh_data);
    Zoltan_Set_Unpack_Obj_Fn(zz, unpack_particles<N>, mesh_data);
    Zoltan_Set_Post_Migrate_Fn(zz, post_migrate_particles<N>, mesh_data);
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

template <int N>
void Zoltan_Do_LB(MESH_DATA<elements::Element<N>>* mesh_data, Zoltan_Struct* load_balancer) {

    // ZOLTAN VARIABLES
    int changes, numGidEntries, numLidEntries, numImport, numExport;
    ZOLTAN_ID_PTR importGlobalGids, importLocalGids, exportGlobalGids, exportLocalGids;
    int *importProcs, *importToPart, *exportProcs, *exportToPart;
    // END OF ZOLTAN VARIABLES

    zoltan_fn_init(load_balancer, mesh_data);
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
