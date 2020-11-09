//
// Created by xetql on 11/9/20.
//

#pragma once
#include <array>
#include <vector>
#include "utils.hpp"

namespace vec {
    template<size_t N>
    std::array<Real, N> operator - (const std::array<Real, N> &lhs, const std::array<Real, N>& rhs){
        std::array<Real, N> res {};
        for(size_t i = 0; i < N; ++i) res[i] = lhs[i] - rhs[i];
        return res;
    }

    template<size_t N>
    std::array<Real, N> operator * (const std::array<Real, N> &lhs, Real s){
        std::array<Real, N> res {};
        for(size_t i = 0; i < N; ++i) res[i] = lhs[i] * s;
        return res;
    }
    template<size_t N>
    std::array<Real, N> operator*(Real s, const std::array<Real, N> &rhs){
        return rhs*s;
    }
    template<size_t N>
    std::array<Real, N> operator+(const std::array<Real, N> &lhs, const std::array<Real, N>& rhs){
        std::array<Real, N> res {};
        for(size_t i = 0; i < N; ++i) res[i] = lhs[i] + rhs[i];
        return res;
    }
    template<size_t N>
    std::array<Real, N> operator/(const std::array<Real, N> &lhs, Real s) {
        return lhs*(1.0/s);
    }
    template<size_t N>
    std::array<Real, N> operator/(Real s, const std::array<Real, N> &rhs) {
        return rhs*(1.0/s);
    }

    template<size_t N>
    std::array<Real, N> operator-(const std::array<Real, N> &lhs){
        std::array<Real, N> res {};
        for(size_t i = 0; i < N; ++i) res[i] = -lhs[i];
        return res;
    }

    template<size_t N>
    Real norm2(const std::array<Real, N> &lhs){
        Real norm = 0.0;
        for(size_t i = 0; i < N; ++i) norm += lhs[i]*lhs[i];
        return norm;
    }
    template<size_t N>
    Real norm(const std::array<Real, N> &lhs){
        return std::sqrt(norm2(lhs));
    }

    template<size_t N>
    std::array<Real, N> normalize(const std::array<Real, N> &lhs){
        std::array<Real, N> res {};
        const auto vnorm = norm<N>(lhs);
        for(size_t i = 0; i < N; ++i) res[i] = lhs[i] / vnorm;
        return res;
    }
    template<size_t N>
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

