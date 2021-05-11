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
#include "zupply.hpp"

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

    float simsize;      /* Borders of the simulation  */
    float rc;           /* factor multiplying sigma for cutoff */

    bool  record;       /* record the simulation in a binary file */
    bool  import  = false;
    bool  monitor = true;
    int   seed;         /* seed used in the RNG */
    int   id = 0;
    int   nb_best_path = 1;
    std::string simulation_name;
    std::string fname;  /* File name (run.out)        */
    int verbosity;
};

namespace {
    #define show(x) #x << "=" << x
}

template<class T>
void print_params(T& stream, const sim_param_t* params) {
    stream << "[Global]" << std::endl;
    stream << show(params->simulation_name) << std::endl;
    stream << show(params->npart) << std::endl;
    stream << show(params->seed) << std::endl;
    stream << show(params->id) << std::endl;
    stream << "\n";

    // stream << "[Physics]" << std::endl;
    // stream << show(params->T0) << std::endl;
    // stream << show(params->sig_lj) << std::endl;
    // stream << show(params->eps_lj) << std::endl;
    // stream << show(params->G) << std::endl;
    // stream << show(params->bounce) << std::endl;
    // stream << show(params->dt) << std::endl;
    // stream << show(params->rc) << std::endl;
    // stream << "\n";

    stream << "[Box]" << std::endl;
    stream << show(params->simsize) << std::endl;
    stream << "\n";

    stream << "[Iterations]" << std::endl;
    stream << show(params->nframes) << std::endl;
    stream << show(params->npframe) << std::endl;
    stream << "Total=" << (params->nframes * params->npframe) << std::endl;
    stream << "\n";

    stream << "[LBSolver]" << std::endl;
    stream << show(params->nb_best_path) << std::endl;
    stream << "\n";

    stream << "[Storing]" << std::endl;
    stream << show(params->monitor) << std::endl;
    stream << show(params->record) << std::endl;
    stream << "\n";

    stream << "[Miscellaneous]" << std::endl;
    stream << show(params->verbosity) << std::endl;
}
template<class T>
void print_params(T& stream, const sim_param_t params) {
    stream << "[Global]" << std::endl;
    stream << show(params.simulation_name) << std::endl;
    stream << show(params.npart) << std::endl;
    stream << show(params.seed) << std::endl;
    stream << show(params.id) << std::endl;
    stream << "\n";

    // stream << "[Physics]" << std::endl;
    // stream << show(params.T0) << std::endl;
    // stream << show(params.sig_lj) << std::endl;
    // stream << show(params.eps_lj) << std::endl;
    // stream << show(params.G) << std::endl;
    // stream << show(params.bounce) << std::endl;
    // stream << show(params.dt) << std::endl;
    // stream << show(params.rc) << std::endl;
    // stream << "\n";

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


template<class Param>
struct TParser {
    using value_type = Param;
    zz::cfg::ArgParser parser;
    std::unique_ptr<sim_param_t> params = std::make_unique<Param>();
    TParser() {
        // Helper
        parser.add_opt_version('V', "version", "MiniLB v1.0:\nMiniLB is a fast parallel (MPI) n-body mini code for load balancing brenchmarking.");
        parser.add_opt_help('h', "help"); // use -h or --help
        parser.add_opt_flag('v', "verbose", "this is the verbose option");

        // Optimal Search
        parser.add_opt_value('B', "best", params->nb_best_path, 1, "Number of Best path to retrieve (A*)", "INT").require();

        // Config
        parser.add_opt_value('N', "name", params->simulation_name, std::string(), "Simulation name", "STRING");
        parser.add_opt_value('i', "id", params->id, 0, "Simulation id", "INT").require();
        parser.add_opt_flag('r', "record", "Record the simulation", &params->record);
        parser.add_opt_value('S', "seed", params->seed, rand(), "Random seed", "INT").require();
        parser.add_opt_value('I', "import", params->fname, std::string("particles.in"), "import particles from this file", "STRING");

        // Compute
        parser.add_opt_value('n', "nparticles", params->npart, 500, "Number of particles", "INT").require();
        parser.add_opt_value('f', "npframe", params->npframe, 100, "steps per frame", "INT").require();
        parser.add_opt_value('F', "nframes", params->nframes, 100, "number of frames", "INT").require();
        parser.add_opt_value('l', "lattice", params->rc, 2.5f, "Lattice size", "FLOAT");
        parser.add_opt_value('t', "dt", params->dt, 1e-5f, "Time step", "FLOAT");
        parser.add_opt_value('w', "width", params->simsize, 1.0f, "Simulation box width", "FLOAT");
    }
    /**
     * Apply modifications on values retrieved by the parser
     */
    virtual void post_parsing() = 0;

    std::unique_ptr<Param> get_params(int argc, char** argv) {
        parser.parse(argc, argv);
        if (parser.count_error() > 0) {
            std::cout << parser.get_error() << std::endl;
            std::cout << parser.get_help() << std::endl;
            return nullptr;
        }
        this->params->verbosity = parser.count("verbose");
        this->post_parsing();
        return std::unique_ptr<Param>(static_cast<Param*>(this->params.release()));
    }
};

std::optional<sim_param_t> get_params(int argc, char** argv);


/*@q*/
#endif /* PARAMS_H */

