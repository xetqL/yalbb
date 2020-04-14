//
// Created by xetql on 04.01.18.
//

#ifndef NBMPI_PHYSICS_HPP
#define NBMPI_PHYSICS_HPP

#include <limits>

Real compute_LJ_scalar(Real r2, Real eps, Real sig2) {
    if (r2 < 6.25 * sig2) { /* r_cutoff = 2.5 *sigma */
        Real z = sig2 / r2;
        Real u = z * z*z;
        return 24 * eps / r2 * u * (1 - 2 * u);
    }
    return 0;
}

template<int N, class T, class GetPosPtrFunc, class GetVelPtrFunc>
void leapfrog1(const Real dt, const Real cut_off, const std::vector<Real>& acc, std::vector<T>& elements
        ,GetPosPtrFunc getPosPtr, GetVelPtrFunc getVelPtr) {
    int i = 0;
    constexpr Real two = 2.0;
    constexpr Real maxSpeedPercentage = 0.9;
    for(auto &el : elements){
        std::array<Real, N>* pos = getPosPtr(el);//(getPosFunc(el));
        std::array<Real, N>* vel = getVelPtr(el);//(getVelFunc(el));
        for(size_t dim = 0; dim < N; ++dim) {
            vel->at(dim) += acc.at(N*i+dim) * dt / two;
            /**
             * This is a mega-giga hotfix.
             * Let say that particles are so close that they produce so much force on them such that the timestep
             * is too big to prevent them to cross the min radius. If a particle cross the min radius of another one
             * it creates an almost infinity repulsive force that breaks everything. */
            if(std::abs(vel->at(dim) * dt) >= cut_off ) {
                vel->at(dim) = maxSpeedPercentage * cut_off / dt; //max speed is 90% of cutoff per timestep
            }
            pos->at(dim) += vel->at(dim) * dt;
        }
        //setPosFunc(el, pos);
        //setVelFunc(el, vel);
    }
}


template<int N, class T, class GetVelPtrFunc>
void leapfrog2(const Real dt, const std::vector<Real>& acc, std::vector<T>& elements, GetVelPtrFunc getVelPtr) {
    int i = 0;
    constexpr Real two = 2.0;
    for(auto &el : elements){
        std::array<Real, N>* vel = getVelPtr(el); //getVelFunc(el);
        for(size_t dim = 0; dim < N; ++dim) {
            vel->at(dim) += acc.at(N*i+dim) * dt / two;
        }
        i++;
    }
}

/**
 * Reflection at the boundary
 * @param wall
 * @param x
 * @param v
 * @param a
 */
void reflect(Real wall, Real* x, Real* v) {
    constexpr Real two = 2.0;
    *x = two * wall - (*x);
    *v = -(*v);
}

template<int N, class T, class GetPosPtrFunc, class GetVelPtrFunc>
void apply_reflect(std::vector<T> &elements, const Real simsize, GetPosPtrFunc getPosPtr, GetVelPtrFunc getVelPtr) {

    for(auto &element: elements) {
        size_t dim = 0;
        std::array<Real, N>* pos  = getPosPtr(element);
        std::array<Real, N>* vel  = getVelPtr(element);
        while(dim < N) {
            if(element.position.at(dim) < 0.0)
                reflect(0.0, &pos->at(dim), &vel->at(dim));
            if(pos->at(dim) >= simsize)
                reflect(simsize-std::numeric_limits<Real>::epsilon(), &pos->at(dim), &vel->at(dim));
            dim++;
        }
    }
}
#endif //NBMPI_PHYSICS_HPP
