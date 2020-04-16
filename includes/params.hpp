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

/*@T
 * \section{System parameters}
 *
 * The [[sim_param_t]] structure holds the parameters that
 * describe the simulation.  These parameters are filled in
 * by the [[get_params]] function (described later).
 *@c*/
typedef struct sim_param_t {
    char* fname;   /* File name (run.out)        */
    int   npart;   /* Number of particles (500)  */
    int   nframes; /* Number of frames (200)     */
    int   npframe; /* Steps per frame (100)      */
    float dt;      /* Time step (1e-4)           */
    float frozen_factor;      /* Time step (1e-4)           */
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
    char*   lb_dataset;
    int   lb_policy = 1;
    short computation_method;
    unsigned int world_size;
    unsigned int lb_interval;
    int one_shot_lb_call;
    unsigned int nb_best_path;
    std::string uuid;
    bool verbose = true;
    bool start_with_lb = false;
} sim_param_t;

void print_params(std::ostream& stream, const sim_param_t& params){
    stream << "==============================================" << std::endl;
    stream << "= Parameters: " << std::endl;
    stream << "= Particles: " << params.npart << std::endl;
    stream << "= Seed: " << params.seed << std::endl;
    stream << "= id: " << params.id << std::endl;
    stream << "= PEs: " << params.world_size << std::endl;
    stream << "= Simulation size: " << params.simsize << std::endl;
    stream << "= Number of time-steps: " << params.nframes << "x" << params.npframe << std::endl;
    stream << "= Initial conditions: " << std::endl;
    stream << "= SIG:" << params.sig_lj << std::endl;
    stream << "= EPS:  " << params.eps_lj << std::endl;
    stream << "= Borders: collisions " << std::endl;
    stream << "= Gravity:  " << params.G << std::endl;
    stream << "= Temperature: " << params.T0 << std::endl;
    stream << "==============================================" << std::endl;
}
void print_params(const sim_param_t& params) {
    print_params(std::cout, params);
}

/*@T
 * \section{Option processing}
 *
 * The [[print_usage]] command documents the options to the [[nbody]]
 * driver program, and [[default_params]] sets the default parameter
 * values.  You may want to add your own options to control
 * other aspects of the program.  This is about as many options as
 * I would care to handle at the command line --- maybe more!  Usually,
 * I would start using a second language for configuration (e.g. Lua)
 * to handle anything more than this.
 *@c*/
static void print_usage() {
    fprintf(stderr,
            "Lennard-Jones n-body simulation\n"
            "\t-B: Number of Best path to retrieve (A*)\n"
            "\t-C: Initial particle configuration 1: Uniform (default), 2:Half, 3:Wall, 4: Cluster\n"
            "\t-d: simulation dimension (0-1;0-1)\n"
            "\t-D: Frozen Decrease factor (0.0 default, i.e., do not freeze simulation over time)\n"
            "\t-e: epsilon parameter in LJ potential (1)\n"
            "\t-f: steps per frame (100)\n"
            "\t-F: number of frames (400)\n"
            "\t-g: gravitational field strength (1)\n"
            "\t-h: print this message\n"
            "\t-i: sim id (default: 0) \n"
            "\t-I: Load balancing call interval (0, never) \n"
            "\t-l: lattice size (3.5*sig_lj) \n"
            "\t-n: number of particles (500)\n"
            "\t-o: output file name (run.out)\n"
            "\t-P: LB Policy 1: Random, 2: Threshold, 3: Fixed (-I interval), 4: Reproduce from file\n"
            "\t-r: record the simulation (false)\n"
            "\t-R: LB reproduction file (default: none)\n"
            "\t-s: distance parameter in LJ potential (1e-2)\n"
            "\t-S: RNG seed\n"
            "\t-t: time step (1e-4)\n"
            "\t-T: initial temperature (1)\n"
    );
}

static void default_params(sim_param_t* params) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution dist(0, 100'000'000);
    params->fname = (char*) "run.out";
    params->npart = 500;
    params->nframes = 400;
    params->npframe = 100;
    params->dt = 1e-4;
    params->eps_lj = 1;
    params->sig_lj = 1e-2;
    params->rc     = 3.5f * params->sig_lj;
    params->G = 1;
    params->T0 = 1;
    params->id = 0;
    params->simsize = 1.0;
    params->record = false;
    params->computation_method = (short) 2;
    params->world_size = (unsigned int) 1;
    params->seed = dist(mt); //by default a random number
    params->lb_interval = 0;
    params->one_shot_lb_call = 0;
    params->nb_best_path = 1;
    params->start_with_lb = false;
    //boost::uuids::random_generator gen;
    //boost::uuids::uuid u = gen(); // generate unique id for this simulation
    //params->uuid = boost::uuids::to_string(u);
    params->frozen_factor = 1.0;

}

/*@T
 *
 * The [[get_params]] function uses the [[getopt]] package
 * to handle the actual argument processing.  Note that
 * [[getopt]] is {\em not} thread-safe!  You will need to
 * do some synchronization if you want to use this function
 * safely with threaded code.
 *@c*/
int get_params(int argc, char** argv, sim_param_t* params) {
    extern char* optarg;
    const char* optstring = "rLho:n:F:f:t:e:s:S:g:T:i:I:d:m:p:B:C:P:R:D:l:";
    int c;

#define get_int_arg(c, field) \
        case c: params->field = atoi(optarg); break
#define get_flt_arg(c, field) \
        case c: params->field = (float) atof(optarg); break
#define get_bool_arg(c, field) \
        case c: params->field = true; break;
    default_params(params);
    while ((c = getopt(argc, argv, optstring)) != -1) {
        switch (c) {
            case 'h':
                print_usage();
                return -1;
            case 'o':
                strcpy(params->fname = new char[(strlen(optarg) + 1)], optarg);
                break;
            case 'R':
                strcpy(params->lb_dataset = new char[(strlen(optarg) + 1)], optarg);
                break;

            get_flt_arg('l', rc);

            get_int_arg('B', nb_best_path);

            get_int_arg('i', id);

            get_int_arg('C', particle_init_conf);

            get_flt_arg('d', simsize);

            get_flt_arg('D', frozen_factor);

            get_flt_arg('e', eps_lj);

            get_int_arg('f', npframe);

            get_int_arg('F', nframes);

            get_flt_arg('g', G);

            get_flt_arg('I', lb_interval);

            get_bool_arg('L', start_with_lb);

            get_int_arg('m', computation_method);

            get_int_arg('n', npart);

            get_int_arg('p', world_size);
            get_int_arg('P', lb_policy);

            get_bool_arg('r', record);

            get_flt_arg('s', sig_lj);
            get_int_arg('S', seed);

            get_flt_arg('t', dt);
            get_flt_arg('T', T0);
            default:
                fprintf(stderr, "Unknown option\n");
                print_usage();
                return -1;
        }
    }
    return 0;
}

/*@q*/
#endif /* PARAMS_H */

