//
// Created by xetql on 4/29/20.
//
#include "utils.hpp"

Real compute_LJ_force(Real r2, Real eps, Real sig){
    const Real lj1 = 48.0 * eps * std::pow(sig,12.0);
    const Real lj2 = 24.0 * eps * std::pow(sig,6.0);
    const Real r2i = 1.0 / r2;
    const Real r6i = r2i*r2i*r2i;
    return r6i * (lj1 * r6i - lj2) * r2i;
}
