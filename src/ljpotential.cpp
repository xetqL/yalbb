//
// Created by xetql on 4/29/20.
//
#include "utils.hpp"
/** FROM LAMMPS
   def fljc(r2):
    r2 = 1.0/r2
    r6 = r2*r2*r2
    lj1 = 48.0 * eps * sig**12.0
    lj2 = 24.0 * eps * sig**6.0;
    return r6 * (lj1*r6 - lj2)
 */
Real compute_LJ_scalar(Real r2, Real eps, Real sig, Real rc2) {
    if (r2 < rc2) { /* r_cutoff = 2.5 *sigma */
        Real lj1 = 48.0 * eps * std::pow(sig,12.0);
        Real lj2 = 24.0 * eps * std::pow(sig,6.0);
        Real r2i = 1.0 / r2;
        Real r6i = r2i*r2i*r2i;
        return r6i * (lj1 * r6i - lj2);
    }
    return 0;
}