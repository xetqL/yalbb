//
// Created by xetql on 04.01.18.
//

#ifndef NBMPI_PHYSICS_HPP
#define NBMPI_PHYSICS_HPP

#include <limits>
#include "parallel_utils.hpp"
#include "cll.hpp"

namespace {
    std::vector<Real> acc;
}

template<int N, class T, class GetPosPtrFunc, class GetVelPtrFunc>
void leapfrog1(const Real dt, const Real cut_off, const std::vector<Real>& acc, std::vector<T>& elements
        ,GetPosPtrFunc getPosPtr, GetVelPtrFunc getVelPtr)  {
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
        i++;
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
void reflect(Real wall, Real* x, Real* v);

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

template<int N, class T, class SetPosFunc, class SetVelFunc, class GetForceFunc>
Complexity nbody_compute_step(
        std::vector<T>&        elements,
        std::vector<T>& remote_el,
        SetPosFunc getPosPtrFunc,                  // function to get force of an entity
        SetVelFunc getVelPtrFunc,                  // function to get force of an entity
        std::vector<Integer> *head,                // the cell starting point
        std::vector<Integer> *lscl,                // the particle linked list
        BoundingBox<N>& bbox,                      // IN:OUT bounding box of particles
        GetForceFunc getForceFunc,                 // function to compute force between entities
        const Borders& borders,                    // bordering cells and neighboring processors
        const Real cutoff,
        const Real dt,
        const Real simwidth) {               // simulation parameters

    std::fill(acc.begin(), acc.end(), (Real) 0.0);

    const size_t nb_elements = elements.size();

    if(const auto n_cells = get_total_cell_number<N>(bbox, cutoff); head->size() < n_cells) {
        head->resize(n_cells);
    }
    if(const auto n_force_elements = N*elements.size(); acc.size() < n_force_elements) {
        acc.resize(N*n_force_elements);
    }
    if(const auto n_particles = elements.size()+remote_el.size();  lscl->size() < n_particles) {
        lscl->resize(n_particles);
    }

    CLL_init<N, T>({ {elements.data(), nb_elements}, {remote_el.data(), remote_el.size()} }, getPosPtrFunc, bbox, cutoff, head, lscl);

    Complexity cmplx = CLL_compute_forces<N, T>(&acc, elements, remote_el, getPosPtrFunc, bbox, cutoff, head, lscl, getForceFunc);

    leapfrog2<N, T>(dt, acc, elements, getVelPtrFunc);
    leapfrog1<N, T>(dt, cutoff, acc, elements, getPosPtrFunc, getVelPtrFunc);
    apply_reflect<N, T>(elements, simwidth, getPosPtrFunc, getVelPtrFunc);

    return cmplx;
};

#endif //NBMPI_PHYSICS_HPP
