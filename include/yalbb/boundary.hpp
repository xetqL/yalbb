//
// Created by xetql on 11/9/20.
//

#pragma once
#include "math.hpp"
template<unsigned N>
constexpr unsigned compute_number_of_neighbors(){
    if constexpr(N == 3) // 3x3x3
        return 27;

    if constexpr(N == 2) // 3x3
        return 9;
    else
        return 0;
}

template<unsigned N>
struct IBoundary {
    static const unsigned n_neighbors = compute_number_of_neighbors<N>();
    virtual void apply(std::array<Real, N>* pos, std::array<Real, N>* vel) = 0;
    virtual std::array<Integer, n_neighbors> neighbors(Integer ibox) = 0;
};

void reflect(Real wall, Real bf, Real* x, Real* v) {
    constexpr  Real two  = 2.0;
    const auto shock_abs = 1.0 - bf;
    const auto new_pos   = two * wall - (*x);
    const auto wall_dist = wall - new_pos;
    *x = new_pos + shock_abs * wall_dist;
    *v = (-(*v)) * bf;
}

template<unsigned N>
struct SphericalBoundary {
    const std::array<Real, N> center {};
    const Real radius {};
    const Real r2 = radius*radius;

    void _vectorized_collide(Real* x, Real* y, Real* z, Real* vx, Real* vy, Real* vz) const {
        std::array<Real, 3> pos =  {*x,  *y,  *z};
        std::array<Real, 3> vel = {*vx, *vy, *vz};
        collide(&pos, &vel);
        *x  = pos[0]; *y  = pos[1]; *z  = pos[2];
        *vx = vel[0]; *vy = vel[1]; *vz = vel[2];
    }

    void collide(std::array<Real, N>* pos, std::array<Real, N>* vel) const {
        using namespace vec::generic;

        const auto& p = *pos;
        const auto& v = *vel;
        const auto CP = p - center;
        const auto norm2_CP = norm2(CP);

        if (norm2_CP > r2) {
            const auto v_norm = normalize(v);
            const auto ds = opt::solve_quadratic(static_cast<Real>(1.0), static_cast<Real>(2.0) * dot(CP, v_norm), norm2_CP - (r2));
            const auto d  = ds.at(opt::argmin(ds.cbegin(), ds.cend(), [](const auto& x){ return std::abs(x); }));
            const auto intersect_pt = p + (v_norm * d);
            const auto n_norm = normalize(center - intersect_pt);
            *pos = p - (static_cast<Real>(2.0) * (dot((p - intersect_pt), n_norm)) * n_norm);
            *vel = v - (static_cast<Real>(2.0) * (dot(v, n_norm)) * n_norm);
        }
    }
};

template<int N>
struct CubicalBoundary {
    const std::array<Real, 2*N> box {};
    const Real bf {};
    std::array<Real, N>   box_size {};

    explicit CubicalBoundary(std::array<Real, 2*N> box, Real bf) : box(box), bf(bf) {
        for(int i = 0; i < N; ++i) box_size[i] = box[2*i+1] - box[2*i];
    }

    void collide(std::array<Real, N>* pos, std::array<Real, N>* vel) const {
        for(auto dim=0; dim < N; ++dim) {
            if(pos->at(dim) < box.at(2*dim))   { reflect(box.at(2 * dim), bf, &pos->at(dim), &vel->at(dim)); }
            if(pos->at(dim) > box.at(2*dim+1)) { reflect(box.at(2 * dim + 1), bf, &pos->at(dim), &vel->at(dim)); }
        }
    }
};

template<int N>
using Boundary = std::variant<CubicalBoundary<N>, SphericalBoundary<N>>;

