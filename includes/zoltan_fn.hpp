//
// Created by xetql on 02.03.18.
//

#ifndef NBMPI_ZOLTAN_FN_HPP
#define NBMPI_ZOLTAN_FN_HPP

#include "spatial_elements.hpp"
#include "params.hpp"
#include "spatial_bisection.hpp"
#include "geometric_load_balancer.hpp"

#include <cassert>
#include <random>
#include <string>
#include <vector>
#include <zoltan.h>

#define ENABLE_AUTOMATIC_MIGRATION true

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

Zoltan_Struct* zoltan_create_wrapper(bool automatic_migration = false) {
    auto zz = Zoltan_Create(MPI_COMM_WORLD);
    Zoltan_Set_Param(zz, "DEBUG_LEVEL", "0");
    Zoltan_Set_Param(zz, "LB_METHOD", "RCB");
    Zoltan_Set_Param(zz, "DETERMINISTIC", "1");
    Zoltan_Set_Param(zz, "NUM_GID_ENTRIES", "1");
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
    if(automatic_migration){
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

template <int N>
inline void zoltan_load_balance(MESH_DATA<N>* mesh_data,
                         std::vector<partitioning::geometric::Domain<N>>& domain_boundaries,
                         Zoltan_Struct* load_balancer,
                         const int nproc,
                         const sim_param_t* params,
                         const partitioning::CommunicationDatatype& datatype,
                         const MPI_Comm comm,
                         bool automatic_migration = false){
    int wsize;
    MPI_Comm_size(comm, &wsize);
    if(wsize == 1) return;

    // ZOLTAN VARIABLES
    int changes, numGidEntries, numLidEntries, numImport, numExport;
    ZOLTAN_ID_PTR importGlobalGids, importLocalGids, exportGlobalGids, exportLocalGids;
    int *importProcs, *importToPart, *exportProcs, *exportToPart, dim;
    double xmin, ymin, zmin, xmax, ymax, zmax;
    // END OF ZOLTAN VARIABLES

    zoltan_fn_init(load_balancer, mesh_data, automatic_migration);
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

    if(changes)
        for(int part = 0; part < nproc; ++part) {
            Zoltan_RCB_Box(load_balancer, part, &dim, &xmin, &ymin, &zmin, &xmax, &ymax, &zmax);
            auto domain = partitioning::geometric::borders_to_domain<N>(dto<elements::ElementRealType>(xmin),
                                                                        dto<elements::ElementRealType>(ymin),
                                                                        dto<elements::ElementRealType>(zmin),
                                                                        dto<elements::ElementRealType>(xmax),
                                                                        dto<elements::ElementRealType>(ymax),
                                                                        dto<elements::ElementRealType>(zmax),
                                                                        params->simsize);
            domain_boundaries[part] = domain;
        }

    if(!automatic_migration)
        load_balancing::geometric::migrate_zoltan<N>(mesh_data->els, numImport, numExport, exportProcs,
                                                     exportGlobalGids, datatype, comm);

    Zoltan_LB_Free_Part(&importGlobalGids, &importLocalGids, &importProcs, &importToPart);
    Zoltan_LB_Free_Part(&exportGlobalGids, &exportLocalGids, &exportProcs, &exportToPart);
}

template<int N>
std::vector<partitioning::geometric::Domain<N>>
retrieve_domain_boundaries(Zoltan_Struct *zz, int nproc, const sim_param_t *params) {
    int dim;
    double xmin, ymin, zmin, xmax, ymax, zmax;
    std::vector<partitioning::geometric::Domain<N>> domain_boundaries(nproc);
    for (int part = 0; part < nproc; ++part) {
        Zoltan_RCB_Box(zz, part, &dim, &xmin, &ymin, &zmin, &xmax, &ymax, &zmax);
        auto domain = partitioning::geometric::borders_to_domain<N>(dto<elements::ElementRealType>(xmin),
                                                                    dto<elements::ElementRealType>(ymin),
                                                                    dto<elements::ElementRealType>(zmin),
                                                                    dto<elements::ElementRealType>(xmax),
                                                                    dto<elements::ElementRealType>(ymax),
                                                                    dto<elements::ElementRealType>(zmax),
                                                                    params->simsize);
        domain_boundaries[part] = domain;
    }
    return domain_boundaries;
}
#endif //NBMPI_ZOLTAN_FN_HPP
