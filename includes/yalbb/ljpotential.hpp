//
// Created by xetql on 04.01.18.
//

#ifndef NBMPI_LJPOTENTIAL_HPP
#define NBMPI_LJPOTENTIAL_HPP

#include "utils.hpp"

Real compute_LJ_scalar(Real r2, Real eps, Real sig2, Real rc2);

template<int N, class T, class GetPosPtrFunc>
std::array<Real, N> lj_compute_force(const T* receiver, const T* source, Real eps, Real sig2, Real rc, GetPosPtrFunc getPosPtr) {
    Real delta = 0.0;

    std::array<Real, N> delta_dim;
    std::array<Real, N> force;

    const auto rec_pos = getPosPtr(const_cast<T*>(receiver));
    const auto sou_pos = getPosPtr(const_cast<T*>(source));

    for (int dim = 0; dim < N; ++dim) delta_dim[dim] = rec_pos->at(dim) - sou_pos->at(dim);
    for (int dim = 0; dim < N; ++dim) delta += (delta_dim[dim] * delta_dim[dim]);

    const Real min_r2 = (rc*rc) / 10000.0;

    delta = std::max(delta, min_r2);

    Real C_LJ = -compute_LJ_scalar(delta, eps, sig2, rc*rc);
    C_LJ = std::max(C_LJ, (Real) -15000.0);

    for (int dim = 0; dim < N; ++dim) {
        force[dim] = (C_LJ * delta_dim[dim]);
    }

    return force;
}

#endif //NBMPI_LJPOTENTIAL_HPP
