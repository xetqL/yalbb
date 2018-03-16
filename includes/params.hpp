//
// Created by xetql on 2/5/18.
//

#ifndef PARAMS_H
#define PARAMS_H
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    float eps_lj;  /* Strength for L-J (1)       */
    float sig_lj;  /* Radius for L-J   (1e-2)    */
    float G;       /* Gravitational strength (1) */
    float T0;      /* Initial temperature (1)    */
    float simsize; /* Borders of the simulation  */
    bool  record;  /* record the simulation in a binary file */
    int   seed;    /* seed used in the RNG */
    short computation_method; /* which computation method to use 1 is brute-force, 2 is cell linked list, 3 Fast Multipole Expansion (todo) */
    unsigned int world_size;
    unsigned int lb_interval;
    int one_shot_lb_call;
} sim_param_t;


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
                    "\t-h: print this message\n"
                    "\t-o: output file name (run.out)\n"
                    "\t-n: number of particles (500)\n"
                    "\t-F: number of frames (200)\n"
                    "\t-f: steps per frame (100)\n"
                    "\t-t: time step (1e-4)\n"
                    "\t-e: epsilon parameter in LJ potential (1)\n"
                    "\t-s: distance parameter in LJ potential (1e-2)\n"
                    "\t-S: RNG seed\n"
                    "\t-g: gravitational field strength (1)\n"
                    "\t-T: initial temperature (1)\n"
                    "\t-I: Load balancing call interval (0, never) \n"
                    "\t-d: simulation dimension (0-1;0-1)\n"
                    "\t-r: record the simulation in a binary file (movie)\n"
                    "\t-m: computation method(1 (BF), 2 (CLL), 3 (FME))\n"
                    "\t-p: number of processing elements\n"
                    "\t-C: Call the load balancer only at this iteration\n");
}

static void default_params(sim_param_t* params) {
    std::random_device rd;
    params->fname = (char*) "run.out";
    params->npart = 500;
    params->nframes = 400;
    params->npframe = 100;
    params->dt = 1e-4;
    params->eps_lj = 1;
    params->sig_lj = 1e-2;
    params->G = 1;
    params->T0 = 1;
    params->simsize = 1.0;
    params->record = false;
    params->computation_method = (short) 2;
    params->world_size = (unsigned int) 1;
    params->seed = rd(); //by default a random number
    params->lb_interval = 0;
    params->one_shot_lb_call = -1;
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
    const char* optstring = "rho:n:F:f:t:e:s:S:g:T:I:d:m:p:C:";
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
            get_bool_arg('r', record);
            get_int_arg('n', npart);
            get_int_arg('F', nframes);
            get_int_arg('f', npframe);
            get_flt_arg('t', dt);
            get_flt_arg('e', eps_lj);
            get_flt_arg('s', sig_lj);
            get_flt_arg('g', G);
            get_flt_arg('T', T0);
            get_flt_arg('I', lb_interval);
            get_flt_arg('d', simsize);
            get_int_arg('m', computation_method);
            get_int_arg('p', world_size);
            get_int_arg('S', seed);
            get_int_arg('C', one_shot_lb_call);
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

