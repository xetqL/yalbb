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
#include <optional>

/*@T
 * \section{System parameters}
 *
 * The [[sim_param_t]] structure holds the parameters that
 * describe the simulation.  These parameters are filled in
 * by the [[get_params]] function (described later).
 *@c*/
struct sim_param_t {

    int   npart;        /* Number of particles (500)  */
    int   nframes;      /* Number of frames (200)     */
    int   npframe;      /* Steps per frame (100)      */
    float dt;           /* Time step (1e-4)           */
    float eps_lj;       /* Strength for L-J (1)       */
    float sig_lj;       /* Radius for L-J   (1e-2)    */
    float G;            /* Gravitational strength (1) */
    float T0;           /* Initial temperature (1)    */
    float simsize;      /* Borders of the simulation  */
    float rc;           /* factor multiplying sigma for cutoff */
    float bounce;       /* shock absorption factor (0=no bounce, 1=full bounce) */
    bool  record;       /* record the simulation in a binary file */
    bool  import  = false;
    bool  monitor = true;
    int   seed;         /* seed used in the RNG */
    int   particle_init_conf = 1;
    int   id = 0;
    int   nb_best_path = 1;

    std::string uuid;
    std::string simulation_name;
    std::string fname;  /* File name (run.out)        */

    int verbosity;

};
namespace {
    #define show(x) #x << "=" << x
}
template<class T>
void print_params(T& stream, const sim_param_t& params){
    stream << "[Global]" << std::endl;
    stream << show(params.simulation_name) << std::endl;
    stream << show(params.npart) << std::endl;
    stream << show(params.seed) << std::endl;
    stream << show(params.id) << std::endl;
    stream << "\n";
    stream << "[Physics]" << std::endl;
    stream << show(params.T0) << std::endl;
    stream << show(params.sig_lj) << std::endl;
    stream << show(params.eps_lj) << std::endl;
    stream << show(params.G) << std::endl;
    stream << show(params.bounce) << std::endl;
    stream << show(params.dt) << std::endl;
    stream << show(params.rc) << std::endl;
    stream << "\n";

    stream << "[Box]" << std::endl;
    stream << show(params.simsize) << std::endl;
    stream << "\n";

    stream << "[Iterations]" << std::endl;
    stream << show(params.nframes) << std::endl;
    stream << show(params.npframe) << std::endl;
    stream << "Total=" << (params.nframes * params.npframe) << std::endl;
    stream << "\n";

    stream << "[LBSolver]" << std::endl;
    stream << show(params.nb_best_path) << std::endl;
    stream << "\n";

    stream << "[Storing]" << std::endl;
    stream << show(params.monitor) << std::endl;
    stream << show(params.record) << std::endl;
    stream << "\n";

    stream << "[Miscellaneous]" << std::endl;
    stream << show(params.verbosity) << std::endl;
}

void print_params(const sim_param_t& params);
std::optional<sim_param_t> get_params(int argc, char** argv);

/*@q*/
#endif /* PARAMS_H */

