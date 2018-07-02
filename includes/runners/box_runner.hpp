//
// Created by xetql on 05.03.18.
//

#ifndef NBMPI_BOXRUNNER_HPP
#define NBMPI_BOXRUNNER_HPP

#include <sstream>
#include <fstream>
#include <iomanip>
#include <mpi.h>
#include <map>
#include <unordered_map>
#include <zoltan.h>
#include <cstdlib>
#include <gsl/gsl_statistics.h>

#include "../astar.hpp"
#include "../ljpotential.hpp"
#include "../report.hpp"
#include "../physics.hpp"
#include "../nbody_io.hpp"
#include "../utils.hpp"
#include "../geometric_load_balancer.hpp"
#include "../params.hpp"
#include "../spatial_elements.hpp"
#include "../graph.hpp"
#include "../metrics.hpp"
#include "../zoltan_fn.hpp"

#ifndef WINDOW_SIZE
#define WINDOW_SIZE 30
#endif

#ifndef N_FEATURES
#define N_FEATURES 14
#endif

#ifndef N_LABEL
#define N_LABEL 1
#endif

#ifndef TICK_FREQ
#define TICK_FREQ 1 // MPI_Wtick()
#endif

#endif //NBMPI_BOXRUNNER_HPP
