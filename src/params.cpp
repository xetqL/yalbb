//
// Created by xetql on 4/29/20.
//
#include "zupply.hpp"
#include "params.hpp"

std::optional<sim_param_t> get_params(int argc, char** argv){
    sim_param_t params;
    zz::cfg::ArgParser parser;

    // Helper
    parser.add_opt_version('V', "version", "MiniLB v1.0:\nMiniLB is a fast parallel (MPI) n-body mini code for load balancing brenchmarking.");
    parser.add_opt_help('h', "help"); // use -h or --help

    // Optimal Search
    parser.add_opt_value('B', "best", params.nb_best_path, 1, "Number of Best path to retrieve (A*)", "INT").require();


    // Config
    parser.add_opt_value('N', "name", params.simulation_name, std::string(), "Simulation name", "STRING");
    parser.add_opt_value('i', "id", params.id, 0, "Simulation id", "INT").require();
    parser.add_opt_flag('r', "record", "Record the simulation", &params.record);
    parser.add_opt_value('S', "seed", params.seed, rand(), "Random seed", "INT").require();
    parser.add_opt_value('I', "import", params.fname, std::string("particles.in"), "import particles from this file", "STRING");

    // Compute
    parser.add_opt_value('n', "nparticles", params.npart, 500, "Number of particles", "INT").require();
    parser.add_opt_value('f', "npframe", params.npframe, 100, "steps per frame", "INT").require();
    parser.add_opt_value('F', "nframes", params.nframes, 100, "number of frames", "INT").require();
    parser.add_opt_value('l', "lattice", params.rc, 2.5f, "Lattice size", "FLOAT");
    parser.add_opt_value('t', "dt", params.dt, 1e-5f, "Time step", "FLOAT");
    parser.add_opt_value('w', "width", params.simsize, 1.0f, "Simulation box width", "FLOAT");

    bool output;
    auto &verbose = parser.add_opt_flag('v', "verbose", "Set verbosity", &output);
    parser.parse(argc, argv);

    if (parser.count_error() > 0) {
        std::cout << parser.get_error() << std::endl;
        std::cout << parser.get_help() << std::endl;
        return std::nullopt;
    }



    return params;
}