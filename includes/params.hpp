//
// Created by xetql on 2/5/18.
//

#ifndef PARAMS_H
#define PARAMS_H
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include "zupply.hpp"

/*@T
 * \section{System parameters}
 *
 * The [[sim_param_t]] structure holds the parameters that
 * describe the simulation.  These parameters are filled in
 * by the [[get_params]] function (described later).
 *@c*/
struct sim_param_t {
    std::string fname;   /* File name (run.out)        */
    int   npart;   /* Number of particles (500)  */
    int   nframes; /* Number of frames (200)     */
    int   npframe; /* Steps per frame (100)      */
    int   world_size; /* wsize     */
    float dt;      /* Time step (1e-4)           */
    float eps_lj;  /* Strength for L-J (1)       */
    float sig_lj;  /* Radius for L-J   (1e-2)    */
    float G;       /* Gravitational strength (1) */
    float T0;      /* Initial temperature (1)    */
    float simsize; /* Borders of the simulation  */
    float rc;
    bool  record;  /* record the simulation in a binary file */
    int   seed;    /* seed used in the RNG */
    int   particle_init_conf = 1;
    int   id = 0;
    int nb_best_path;
    std::string uuid;
    int verbosity;
};
void print_params(std::ostream& stream, const sim_param_t& params);
void print_params(const sim_param_t& params);
std::optional<sim_param_t> get_params(int argc, char** argv);

/*@q*/
#endif /* PARAMS_H */

