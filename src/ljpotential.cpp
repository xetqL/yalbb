//
// Created by xetql on 4/29/20.
//
#include "utils.hpp"

Real compute_LJ_scalar(Real r2, Real eps, Real sig2, Real rc2) {
    if (r2 < rc2) { /* r_cutoff = 2.5 *sigma */
        Real z = sig2 / r2;
        Real u = z * z*z;
        return 24 * eps / r2 * u * (1 - 2 * u);
    }
    return 0;
}