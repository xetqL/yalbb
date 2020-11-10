//
// Created by xetql on 11/9/20.
//

#pragma once
#include "math.hpp"

void reflect(Real wall, Real bf, Real* x, Real* v) {
    constexpr  Real two  = 2.0;
    const auto shock_abs = 1.0 - bf;
    const auto new_pos   = two * wall - (*x);
    const auto wall_dist = wall - new_pos;
    *x = new_pos + shock_abs * wall_dist;
    *v = (-(*v)) * bf;
}

template<size_t N>
struct SphericalBoundary {
    const std::array<Real, N> center {};
    const Real radius {};
    const Real r2 = radius*radius;
    void collide(std::array<Real, N>* pos, std::array<Real, N>* vel) const {
        using namespace vec;
        const auto& p = *pos;
        const auto& v = *vel;
        const auto CP = p - center;
        const auto norm2_CP = norm2<N>(CP);
        if(norm2_CP > r2) {
            const auto v_norm = normalize<N>(v);
            const auto ds = opt::solve_quadratic(1, 2.0 * dot<N>(CP, v_norm), norm2_CP - (r2));
            if(!ds.empty()) {
                const auto d = ds.at(opt::argmin(ds.cbegin(), ds.cend(), [](const auto& x){ return std::abs(x); }));
                const auto intersect_pt = p + v_norm * d;
                const auto n_norm = normalize<N>(center - intersect_pt);
                *pos = p - 2.0 * (dot<N>((p - intersect_pt), n_norm)) * n_norm;
                *vel = apply(v, ( apply(normalize<N>((*pos)-intersect_pt), v_norm, std::divides{}), std::multiplies{} );
            }
        }
    }
};

template<int N>
struct CubicalBoundary {
    const std::array<Real, 2*N> box;
    const Real bf;
    const std::array<Real, N>   box_size;

    explicit CubicalBoundary(std::array<Real, 2*N> box, Real bf) : box(box), bf(bf) {
        for(int i = 0; i < N; ++i) box_size[i] = box[2*i+1] - box[2*i];
    }

    void collide(std::array<Real, N>* pos, std::array<Real, N>* vel) const {
        for(auto dim=0; dim < N; ++dim){
            if(pos->at(dim) < box.at(2*dim))
                reflect(box.at(2*dim), bf, &pos->at(dim),   &vel->at(dim));
            if(pos->at(dim) > box.at(2*dim+1))
                reflect(box.at(2*dim+1), bf, &pos->at(dim), &vel->at(dim));
            dim++;
        }
    }
};

template<int N>
using Boundary = std::variant<CubicalBoundary<N>, SphericalBoundary<N>>;

