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
    for(auto &el : elements){
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
void reflect(Real wall, Real* x, Real* v);

template<int N, class T, class GetPosPtrFunc, class GetVelPtrFunc>
void apply_reflect(std::vector<T> &elements, const Real simsize, GetPosPtrFunc getPosPtr, GetVelPtrFunc getVelPtr) {
    for(auto &element: elements) {
        size_t dim = 0;
        std::array<Real, N>* pos  = getPosPtr(&element);
        std::array<Real, N>* vel  = getVelPtr(&element);
        while(dim < N) {
            if(element.position.at(dim) < 0.0)
                reflect(0.0, &pos->at(dim), &vel->at(dim));
            if(simsize-pos->at(dim) <= 0.000001f)
                reflect(simsize-0.000001f, &pos->at(dim), &vel->at(dim));
            dim++;
        }
    }
}

template<int N, class T, class SetPosFunc, class SetVelFunc, class GetForceFunc>
Complexity nbody_compute_step(
        std::vector<T>& elements,
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

    const size_t
          n_local_particles = elements.size(),
          n_remote_particles= remote_el.size(),
          n_total_particles = n_local_particles+n_remote_particles,
          n_allocated_particles= lscl->size(),
          n_allocated_force_components= acc.size(),
          n_allocated_cells = head->size();

    if(const auto n_cells = get_total_cell_number<N>(bbox, cutoff); n_allocated_cells < n_cells) {
        head->resize(n_cells);
    } else if(n_allocated_cells >= 2 * n_cells){
        head->resize(n_allocated_cells/2.0 + n_cells /2.0);
    }

    if(const auto n_force_components = N*n_local_particles; n_allocated_force_components < n_force_components) {
        acc.resize(n_force_components);
    } else if (n_allocated_force_components >= 2.0*n_force_components) {
        acc.resize(n_allocated_force_components/2.0 + n_force_components/2.0);
    }

    if(n_allocated_particles < n_total_particles) {//resize up
        lscl->resize(n_total_particles);
    } else if ( n_allocated_particles >= 2 * n_total_particles ){ //resize down
        lscl->resize(n_allocated_particles / 2.0 + n_total_particles / 2.0);
    }

    leapfrog1<N, T>(dt, cutoff, acc, elements, getPosPtrFunc, getVelPtrFunc);

    apply_reflect<N, T>(elements, simwidth, getPosPtrFunc, getVelPtrFunc);

    CLL_init<N, T>({ {elements.data(), elements.size()}, {remote_el.data(), remote_el.size()} }, getPosPtrFunc, bbox, cutoff, head, lscl);

    Complexity cmplx = CLL_compute_forces<N, T>(&acc, elements, remote_el, getPosPtrFunc, bbox, cutoff, head, lscl, getForceFunc);

    leapfrog2<N, T>(dt, acc, elements, getVelPtrFunc);

    return cmplx;
};

#endif //NBMPI_PHYSICS_HPP
