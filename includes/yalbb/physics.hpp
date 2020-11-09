//
// Created by xetql on 04.01.18.
//

#ifndef NBMPI_PHYSICS_HPP
#define NBMPI_PHYSICS_HPP

#include <limits>
#include "parallel_utils.hpp"
#include "cll.hpp"
#include "boundary.hpp"

template<int N, class T, class GetPosPtrFunc, class GetVelPtrFunc>
void leapfrog1(const Real dt, const Real cut_off, const std::vector<Real>& acc, std::vector<T>& elements
        ,GetPosPtrFunc getPosPtr, GetVelPtrFunc getVelPtr)  {
    int i = 0;
    constexpr Real two = 2.0;
    constexpr Real maxSpeedPercentage = 0.5;
    for(auto &el : elements){
        std::array<Real, N>* pos = getPosPtr(&el);//(getPosFunc(el));
        std::array<Real, N>* vel = getVelPtr(&el);//(getVelFunc(el));
        for(size_t dim = 0; dim < N; ++dim) {
            vel->at(dim) += acc.at(N*i+dim) * dt / two;
            /**
             * This is a mega-giga hotfix.
             * Let say that particles are so close that they produce so much force on them such that the timestep
             * is too big to prevent them to cross the min radius. If a particle cross the min radius of another one
             * it creates an almost infinity repulsive force that breaks everything. */
            if(std::abs(vel->at(dim) * dt) >= cut_off || std::isnan(vel->at(dim))) {
                vel->at(dim) = maxSpeedPercentage * cut_off / dt; //max speed is 90% of cutoff per timestep
            }
            pos->at(dim) += vel->at(dim) * dt;
        }
        i++;
    }
}


template<int N, class T, class GetVelPtrFunc>
void leapfrog2(const Real dt, const std::vector<Real>& acc, std::vector<T>& elements, GetVelPtrFunc getVelPtr) {
    int i = 0;
    constexpr Real two = 2.0;
    for(auto &el : elements) {
        std::array<Real, N>* vel = getVelPtr(&el); //getVelFunc(el);
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
void reflect(Real wall, Real bf, Real* x, Real* v);

template<int N, class T, class GetPosPtrFunc, class GetVelPtrFunc>
void apply_reflect(std::vector<T> &elements, const Real simsize, const Real bf, GetPosPtrFunc getPosPtr, GetVelPtrFunc getVelPtr) {
    for(auto &element: elements) {
        size_t dim = 0;
        std::array<Real, N>* pos  = getPosPtr(&element);
        std::array<Real, N>* vel  = getVelPtr(&element);
        while(dim < N) {
            if(element.position.at(dim) < 0.0)
                reflect(0.0, bf, &pos->at(dim), &vel->at(dim));
            if(simsize-pos->at(dim) <= 0.00000001f)
                reflect(simsize-0.00000001f, bf, &pos->at(dim), &vel->at(dim));
            dim++;
        }
    }
}

template<int N, class T, class GetPosPtrFunc, class GetVelPtrFunc>
void apply_reflect(std::vector<T> &elements, const Boundary<N>& boundary, GetPosPtrFunc getPosPtr, GetVelPtrFunc getVelPtr) {
    for(auto &element: elements) {
        size_t dim = 0;
        std::array<Real, N>* pos  = getPosPtr(&element);
        std::array<Real, N>* vel  = getVelPtr(&element);
        std::visit([&](const auto& boundary){ boundary.collide(pos, vel); }, boundary);
    }
}

template<int N, class T, class SetPosFunc, class SetVelFunc, class GetForceFunc>
Complexity nbody_compute_step(
        std::vector<Real>& flocal,
        std::vector<T>& elements,
        std::vector<T>& remote_el,
        SetPosFunc getPosPtrFunc,                  // function to get force of an entity
        SetVelFunc getVelPtrFunc,                  // function to get force of an entity
        std::vector<Integer> *head,                // the cell starting point
        std::vector<Integer> *lscl,                // the particle linked list
        BoundingBox<N>& bbox,                      // IN:OUT bounding box of particles
        GetForceFunc getForceFunc,                 // function to compute force between entities
        const Boundary<N>& boundary,
        const Real cutoff,                         // simulation parameters
        const Real dt,
        const Real simwidth,
        const Real Gforce,
        const Real bf) {

    std::fill(flocal.begin(), flocal.end(), (Real) 0.0);

    const auto size = flocal.size();

    for(int i = N-1; i < size; i += N) flocal.at(i) = Gforce;

    leapfrog1<N, T>(dt, cutoff, flocal, elements, getPosPtrFunc, getVelPtrFunc);

    apply_reflect<N, T>(elements, simwidth, bf, getPosPtrFunc, getVelPtrFunc);

    Complexity cmplx = CLL_compute_forces<N, T>(&flocal, elements, remote_el, getPosPtrFunc, bbox, cutoff, head, lscl, getForceFunc);

    leapfrog2<N, T>(dt, flocal, elements, getVelPtrFunc);

    return cmplx;
};

#endif //NBMPI_PHYSICS_HPP
