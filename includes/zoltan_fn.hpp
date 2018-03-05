//
// Created by xetql on 02.03.18.
//

#ifndef NBMPI_ZOLTAN_FN_HPP
#define NBMPI_ZOLTAN_FN_HPP

#include "spatial_elements.hpp"
#include "ljpotential.hpp"
#include "params.hpp"

#include <cassert>
#include <random>
#include <string>
#include <vector>
#include <zoltan.h>

template<int N>
struct _MESH_DATA {
    int numMyPoints;
    std::vector<elements::Element<2>> els;
};

using MESH_DATA = _MESH_DATA<2>;

void init_mesh_data(int rank, int nprocs, MESH_DATA* mesh_data, sim_param_t* params) {

    if (rank == 0) {
        double min_r2 = 1e-2*1e-2;

        //std::random_device rd; //Will be used to obtain a seed for the random number engine
        std::mt19937 gen(params->seed); //Standard mersenne_twister_engine seeded with rd()
        std::uniform_real_distribution<double> udist(0.0, params->simsize);

        std::vector<elements::Element<2>> elements(params->npart);

        elements::Element<2>::create_random_n(elements, udist, gen, [min_r2](auto point, auto other){
            return elements::distance2<2>(point, other) >= min_r2;
        });

        elements::init_particles_random_v(elements, params->T0);
        mesh_data->els = elements;
        mesh_data->numMyPoints = params->npart;
    } else {
        mesh_data->numMyPoints = 0;
    }
}

int get_number_of_objects(void *data, int *ierr) {
    MESH_DATA *mesh= (MESH_DATA *)data;
    *ierr = ZOLTAN_OK;
    return mesh->numMyPoints;
}

void get_object_list(void *data, int sizeGID, int sizeLID,
                     ZOLTAN_ID_PTR globalID, ZOLTAN_ID_PTR localID,
                     int wgt_dim, float *obj_wgts, int *ierr) {
    int i;
    MESH_DATA *mesh= (MESH_DATA *)data;
    *ierr = ZOLTAN_OK;
    /* In this example, return the IDs of our objects, but no weights.
     * Zoltan will assume equally weighted objects.
     */
    for (i=0; i < mesh->numMyPoints; i++){
        globalID[i] = mesh->els[i].identifier;
        localID[i] = i;
    }
}

int get_num_geometry(void *data, int *ierr) {
    *ierr = ZOLTAN_OK;
    return 2;
} 

void get_geometry_list(void *data, int sizeGID, int sizeLID,
                       int num_obj,
                       ZOLTAN_ID_PTR globalID, ZOLTAN_ID_PTR localID,
                       int num_dim, double *geom_vec, int *ierr) {
    int i;

    MESH_DATA *mesh= (MESH_DATA *)data;

    if ( (sizeGID != 1) || (sizeLID != 1) || (num_dim != 2)){
        *ierr = ZOLTAN_FATAL;
        return;
    }

    *ierr = ZOLTAN_OK;

    for (i=0;  i < num_obj; i++){
        geom_vec[2*i] = mesh->els[i].position.at(0);
        geom_vec[2*i + 1] = mesh->els[i].position.at(1);
    }

    return;
}

int get_obj_size(
        void *data,
        int num_gid_entries,
        int num_lid_entries,
        ZOLTAN_ID_PTR global_id,
        ZOLTAN_ID_PTR local_id,
        int *ierr) {
    return elements::Element<2>::byte_size() + 5;
}

void pack_object (
        void *data,
        int num_gid_entries,
        int num_lid_entries,
        ZOLTAN_ID_PTR global_id,
        ZOLTAN_ID_PTR local_id,
        int dest,
        int size,
        char *buf,
        int *ierr) {
    assert((int) *global_id != -1);

    MESH_DATA *mesh_data = (MESH_DATA*) data;
    std::ostringstream packed_obj;
    int id = *global_id;
    size_t lid;
    for(size_t i = 0; i < mesh_data->els.size(); ++i){
        if(mesh_data->els[i].identifier == id){
            lid = i;
            break;
        }
    }
    auto el = mesh_data->els[lid];
    std::string comm_buf = el.to_communication_buffer();
    packed_obj << std::setfill ('*') << std::setw (size) << comm_buf;
    packed_obj.str().copy(buf, size);

    *ierr = ZOLTAN_OK;

}

void post_migration (
        void *data,
        int num_gid_entries,
        int num_lid_entries,
        int num_import,
        ZOLTAN_ID_PTR import_global_ids,
        ZOLTAN_ID_PTR import_local_ids,
        int *import_procs,
        int *import_to_part,
        int num_export,
        ZOLTAN_ID_PTR export_global_ids,
        ZOLTAN_ID_PTR export_local_ids,
        int *export_procs,
        int *export_to_part,
        int *ierr){

    MESH_DATA *mesh_data = (MESH_DATA*) data;
    decltype(mesh_data->els) cleaned_els;
    std::vector<ZOLTAN_ID_TYPE> gids(export_global_ids, export_global_ids + num_export);
    for(auto el: mesh_data->els){
        if(std::find(gids.begin(), gids.end(), el.identifier) == gids.end()) cleaned_els.push_back(el);
    }
    mesh_data->els = cleaned_els;
}

void unpack_object (
        void *data,
        int num_gid_entries,
        ZOLTAN_ID_PTR global_id,
        int size,
        char *buf,
        int *ierr){
    assert((int) *global_id != -1);

    MESH_DATA *mesh_data = (MESH_DATA*) data;

    int i;
    for(i = 0; i < size; ++i) if(buf[i] == '*') continue; else break;
    std::string el_str(buf);
    el_str.erase(0, i);
    std::string::size_type sz;
    elements::Element<2> e = elements::Element<2>();

    // position
    auto pos = el_str.find(";");
    auto token = el_str.substr(0, pos);

    double px = std::stod (token, &sz);
    double py = std::stod (token.substr(sz));
    el_str.erase(0, pos + 1);

    //velocity
    pos = el_str.find(";");
    token = el_str.substr(0, pos);
    double vx = std::stod (token, &sz);
    double vy = std::stod (token.substr(sz));
    el_str.erase(0, pos + 1);

    //acceleration
    pos = el_str.find(";");
    token = el_str.substr(0, pos);
    double ax = std::stod (token, &sz);
    double ay = std::stod (token.substr(sz));

    //identifier
    el_str.erase(0, pos + 1);
    pos = el_str.find("!");
    token = el_str.substr(0, pos);
    int id = std::stoi(token, &sz);

    e.position[0] = px; e.position[1] = py;
    e.velocity[0] = vx; e.velocity[1] = vy;
    e.acceleration[0] = ax; e.acceleration[1] = ay;
    e.identifier = id;

    mesh_data->els.push_back(e);

    *ierr = ZOLTAN_OK;
}

Zoltan_Struct* zoltan_create_wrapper() {
    auto zz = Zoltan_Create(MPI_COMM_WORLD);

    Zoltan_Set_Param(zz, "DEBUG_LEVEL", "0");
    Zoltan_Set_Param(zz, "LB_METHOD", "RCB");
    Zoltan_Set_Param(zz, "NUM_GID_ENTRIES", "1");
    Zoltan_Set_Param(zz, "NUM_LID_ENTRIES", "1");
    Zoltan_Set_Param(zz, "OBJ_WEIGHT_DIM", "0");
    Zoltan_Set_Param(zz, "RETURN_LISTS", "ALL");

    Zoltan_Set_Param(zz, "RCB_OUTPUT_LEVEL", "0");
    Zoltan_Set_Param(zz, "RCB_RECTILINEAR_BLOCKS", "1");
    Zoltan_Set_Param(zz, "KEEP_CUTS", "1");
    return zz;
}

void zoltan_fn_init(Zoltan_Struct* zz, MESH_DATA* mesh_data){
    Zoltan_Set_Num_Obj_Fn(zz, get_number_of_objects, mesh_data);
    Zoltan_Set_Obj_List_Fn(zz, get_object_list, mesh_data);
    Zoltan_Set_Num_Geom_Fn(zz, get_num_geometry, mesh_data);
    Zoltan_Set_Geom_Multi_Fn(zz, get_geometry_list, mesh_data);

    Zoltan_Set_Obj_Size_Fn(zz, get_obj_size, mesh_data);
    Zoltan_Set_Unpack_Obj_Fn(zz, unpack_object, mesh_data);
    Zoltan_Set_Pack_Obj_Fn(zz, pack_object, mesh_data);
    Zoltan_Set_Post_Migrate_PP_Fn(zz, post_migration, mesh_data);
}

#endif //NBMPI_ZOLTAN_FN_HPP
