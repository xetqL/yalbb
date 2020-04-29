//
// Created by xetql on 4/29/20.
//

#include "physics.hpp"

void reflect(Real wall, Real* x, Real* v) {
    constexpr Real two = 2.0;
    *x = two * wall - (*x);
    *v = -(*v);
}