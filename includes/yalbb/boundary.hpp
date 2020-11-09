//
// Created by xetql on 11/9/20.
//

#pragma once
#include "math.hpp"

template<int N>
struct SphericalBoundary {
    const std::array<Real, N> center;
    const Real radius;
    void collide(std::array<Real, N>* pos, std::array<Real, N>* vel) const {
        using namespace vec;
        const auto& p = *pos;
        const auto& v = *vel;
        // check if point is outside sphere
        const auto CP = p - center;
        const auto v_normalized = normalize<N>(v);
        if(norm(CP) > radius) {
            const auto ds = solve_quadratic(norm2(v), 2.0 * dot<N>(p, v_normalized), norm2<N>(p) - (radius*radius + norm<N>(center)));
            const auto d = *std::min_element(ds.cbegin(), ds.cend());
            const auto psphere = p + v_normalized * d;
            const auto n_normalized = normalize<N>(center - psphere);
            *pos = p * n_normalized;
            *vel = v * n_normalized;
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
                reflect(0.0, bf, &pos->at(dim), &vel->at(dim));
            if(pos->at(dim) > box.at(2*dim+1))
                reflect(box_size.at(dim), bf, &pos->at(dim), &vel->at(dim));
            dim++;
        }
    }
};

template<int N>
using Boundary = std::variant<CubicalBoundary<N>, SphericalBoundary<N>>;

