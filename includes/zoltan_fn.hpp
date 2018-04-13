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
struct MESH_DATA {
    std::vector<elements::Element<N>> els;
};

template<int N>
void init_mesh_data(int rank, int nprocs, MESH_DATA<N>& mesh_data, sim_param_t* params) {

    if (rank == 0) {
        double min_r2 = params->sig_lj*params->sig_lj;

        //std::random_device rd; //Will be used to obtain a seed for the random number engine
        std::mt19937 gen(params->seed); //Standard mersenne_twister_engine seeded with rd()
        std::uniform_real_distribution<double> udist(0.0, params->simsize);
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

    return;
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

template<int N>
void zoltan_fn_init(Zoltan_Struct* zz, MESH_DATA<N>* mesh_data){
    Zoltan_Set_Num_Obj_Fn(   zz, get_number_of_objects<N>, mesh_data);
    Zoltan_Set_Obj_List_Fn(  zz, get_object_list<N>,       mesh_data);
    Zoltan_Set_Num_Geom_Fn(  zz, get_num_geometry<N>,      mesh_data);
    Zoltan_Set_Geom_Multi_Fn(zz, get_geometry_list<N>,     mesh_data);
}

#endif //NBMPI_ZOLTAN_FN_HPP
