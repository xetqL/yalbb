//
// Created by xetql on 11/9/20.
//

#pragma once
#include <array>
#include <vector>
#include "utils.hpp"

namespace vec{
    template<int N>
    std::array<Real, N> operator-(const std::array<Real, N> &lhs, const std::array<Real, N>& rhs){
        std::array<Real, N> res {};
        for(auto i = 0; i < N; ++i) res[i] = lhs[i] - rhs[i];
        return res;
    }

    template<int N>
    std::array<Real, N> operator*(const std::array<Real, N> &lhs, Real s){
        std::array<Real, N> res {};
        for(auto i = 0; i < N; ++i) res[i] = lhs[i] * s;
        return res;
    }
    template<int N>
    std::array<Real, N> operator*(Real s, const std::array<Real, N> &rhs){
        return rhs*s;
    }
    template<int N>
    std::array<Real, N> operator+(const std::array<Real, N> &lhs, const std::array<Real, N>& rhs){
        std::array<Real, N> res {};
        for(auto i = 0; i < N; ++i) res[i] = lhs[i] + rhs[i];
        return res;
    }
    template<int N>
    std::array<Real, N> operator/(const std::array<Real, N> &lhs, Real s) {
        return lhs*(1.0/s);
    }
    template<int N>
    std::array<Real, N> operator/(Real s, const std::array<Real, N> &rhs) {
        return rhs*(1.0/s);
    }

    template<int N>
    std::array<Real, N> operator-(const std::array<Real, N> &lhs){
        std::array<Real, N> res {};
        for(auto i = 0; i < N; ++i) res[i] = -lhs[i];
        return res;
    }

    template<int N>
    Real norm2(const std::array<Real, N> &lhs){
        Real norm = 0.0;
        for(auto i = 0; i < N; ++i) norm += lhs[i]*lhs[i];
        return norm;
    }
    template<int N>
    Real norm(const std::array<Real, N> &lhs){
        return std::sqrt(norm2(lhs));
    }

    template<int N>
    std::array<Real, N> normalize(const std::array<Real, N> &lhs){
        std::array<Real, N> res {};
        const auto vnorm = norm<N>(lhs);
        for(auto i = 0; i < N; ++i) res[i] = lhs / vnorm;
        return res;
    }
    template<int N>
    Real dot(const std::array<Real, N>& lhs, const std::array<Real, N>& rhs){
        return std::inner_product(std::begin(lhs), std::end(lhs), std::begin(rhs), (Real) 0.0);
    }

}

/* solve quadratic equation of the form ax^2 + bx + c = 0*/
std::vector<Real> solve_quadratic(Real a, Real b, Real c){
    Real delta = (b * b) - (4.0*a*c);
    if (delta < 0) { return {}; }
    std::vector<Real> solutions{};
    solutions.reserve(2);
    Real sqrt_delta = std::sqrt(delta);
    solutions.push_back( (-b + sqrt_delta) / (2.0 * a) );
    if (delta > 0.0) {
        solutions.push_back( (-b - sqrt_delta) / (2.0 * a) );
    }
    return solutions;
}

