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
        std::array<Real, N> res {};
        for(size_t i = 0; i < N; ++i) res[i] = s / rhs[i];
        return res;
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
        return std::inner_product(std::cbegin(lhs), std::cend(lhs), std::begin(rhs), (Real) 0.0);
    }

    template<size_t N, class BinaryOp>
    std::array<Real, N> apply(const std::array<Real, N> &lhs, const std::array<Real, N> &rhs, BinaryOp op) {
        std::array<Real, N> res {};
        for(size_t i = 0; i < N; ++i) res[i] = op(lhs[i], rhs[i]);
        return res;
    }

}
namespace opt {
    template<class InputIt>
    auto argmin(InputIt beg, InputIt end) {
        return std::distance(beg, std::min_element(beg, end));
    }

    template<class InputIt, class UnaryOp>
    auto argmin(InputIt beg, InputIt end, UnaryOp op) {
        std::vector<typename InputIt::value_type> v(beg, end);
        std::transform(v.begin(), v.end(), v.begin(), op);
        auto result = std::min_element(v.begin(), v.end());
        return std::distance(v.begin(), result);
    }

    /* solve quadratic equation of the form ax^2 + bx + c = 0*/
    std::vector<Real> solve_quadratic(Real a, Real b, Real c){
        const Real delta = (b * b) - (4.0*a*c);
        if (delta < 0) { return {}; }
        const Real two_a = 2.0 * a;
        const Real sqrt_delta = std::sqrt(delta);
        return {(-b + sqrt_delta) / (two_a), (-b - sqrt_delta) / (two_a)};
    }
}


