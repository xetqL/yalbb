//
// Created by xetql on 4/29/20.
//

#include "physics.hpp"

void reflect(Real wall, Real bf, Real* x, Real* v) {
    constexpr Real two = 2.0;
    const auto shock_abs = 1.0 - bf;
    const auto new_pos   = two * wall - (*x);
    const auto wall_dist = wall - new_pos;
    *x = new_pos + shock_abs * wall_dist;
    *v = (-(*v)) * bf;
}