//
// Created by xetql on 04.01.18.
//

#ifndef NBMPI_PHYSICS_HPP
#define NBMPI_PHYSICS_HPP

#include <limits>
#include "spatial_elements.hpp"

#define EPS   1
#define SIG   1e-2
#define RCUT  (2.5*SIG)

template<typename RealType>
RealType compute_LJ_scalar(RealType r2, RealType eps, RealType sig2) {
    if (r2 < 6.25 * sig2) { /* r_cutoff = 2.5 sigma */
        RealType z = sig2 / r2;
        RealType u = z * z*z;
        return 24 * eps / r2 * u * (1 - 2 * u);
    }
    return 0;
}

template<typename RealType>
void leapfrog1(int n, RealType dt, RealType* x, RealType* v, RealType* a) {
    for (int i = 0; i < n; ++i, x += 2, v += 2, a += 2) {
        v[0] += a[0] * dt / 2;
        v[1] += a[1] * dt / 2;
        x[0] += v[0] * dt;
        x[1] += v[1] * dt;
    }
}

template<typename RealType>
void leapfrog2(int n, RealType dt, RealType* v, RealType* a) {
    for (int i = 0; i < n; ++i, v += 2, a += 2) {
        v[0] += a[0] * dt / 2;
        v[1] += a[1] * dt / 2;
    }
}

/**
 * Reflection at the boundary
 * @param wall
 * @param x
 * @param v
 * @param a
 */
template<typename RealType>
static void reflect(RealType wall, RealType* x, RealType* v, RealType* a) {
    *x = 2 * wall - (*x);
    *v = -(*v);
    *a = -(*a);
}

/**
 * Apply the reflection on all the particles at the border of the simulation
 * @param n Number of particles
 * @param x Position of particles
 * @param v Velocity of particles
 * @param a Acceleration of particles
 * @param borders Border of the simulation (XMIN, XMAX, YMIN, YMAX)
 */
template<typename RealType>
void apply_reflect(unsigned int n, RealType* x, RealType* v, RealType* a, RealType simsize) {
    unsigned int i = 0;
    while(i < n) {

        if (x[0] < 0.0) reflect((RealType) 0.0, x + 0, v + 0, a + 0);

        if (x[0] >= simsize) reflect(simsize-std::numeric_limits<RealType>::epsilon(), x + 0, v + 0, a + 0);

        if (x[1] < 0.0) reflect((RealType) 0.0, x + 1, v + 1, a + 1);

        if (x[1] >= simsize) reflect(simsize-std::numeric_limits<RealType>::epsilon(), x + 1, v + 1, a + 1);

        if(x[0] < 0.0 || x[1] < 0.0 || x[0] >= simsize || x[1] >= simsize) continue;

        i++; x+=2; v+=2; a+=2;
    }
}

template<int N>
void apply_reflect(std::vector<elements::Element<N>> &elements, const double simsize) {
    for(auto &element: elements){
        size_t dim = 0;
        while(dim < N){
            if(element.position.at(dim) < 0.0)
                reflect(0.0, &element.position[dim], &element.velocity[dim], &element.acceleration[dim]);
            if(element.position.at(dim) >= simsize)
                reflect(simsize-std::numeric_limits<double>::epsilon(), &element.position[dim], &element.velocity[dim], &element.acceleration[dim]);
            if(element.position.at(dim) < 0.0 || element.position.at(dim) >= simsize) continue;
            dim++;
        }
    }
}


/**
 * Reflection at the boundary
 * @param wall
 * @param x
 * @param v
 * @param a

template<class ContainerType, typename RealType, int N>
static void reflect(RealType wall, ContainerType &elements) {
    using Elements = typename std::enable_if<std::is_same<ContainerType::value_type, elements::Element<N>>::value>::type

    *x = 2 * wall - (*x);
    *v = -(*v);
    *a = -(*a);
}
*/

#endif //NBMPI_PHYSICS_HPP
