//
// Created by xetql on 11/9/20.
//

#pragma once
#include <array>
#include <vector>
#include "utils.hpp"

namespace vec::generic{
    namespace
    {
        template<class InputIt, class OutputIt, class BinaryOp>
        OutputIt apply(InputIt beg1, InputIt end1, InputIt beg2, OutputIt out, BinaryOp op) {
            const auto beg_out = out;
            for(;beg1 != end1; beg1++, beg2++){
                (*out) = op( (*beg1), (*beg2) );
                out++;
            }
            return beg_out;
        }
        template<class InputIt, class OutputIt, class Scalar, class BinaryOp>
        OutputIt apply(InputIt beg1, InputIt end1, Scalar v, OutputIt out, BinaryOp op) {
            const auto beg_out = out;
            for(;beg1 != end1; beg1++){
                *(out++) = op( (*beg1), v );
            }
            return beg_out;
        }
        template<class InputIt, class OutputIt, class UnaryOp>
        OutputIt apply(InputIt beg1, InputIt end1, OutputIt out, UnaryOp op) {
        const auto beg_out = out;
        for(;beg1 != end1; beg1++, out++){
            *out = op( (*beg1) );
        }
        return beg_out;
    }
    }
    template<class T> T operator - (const T& lhs, const T& rhs){
        T ret = lhs;
        apply(std::begin(lhs), std::end(lhs), std::begin(rhs), std::begin(ret), std::minus{});
        return ret;
    }
    template<class T> T operator + (const T& lhs, const T& rhs){
        T ret = lhs;
        apply(std::begin(lhs), std::end(lhs), std::begin(rhs), std::begin(ret), std::plus{});
        return ret;
    }
    template<class T> T operator * (const T& lhs, const T& rhs){
        T ret = lhs;
        apply(std::begin(lhs), std::end(lhs), std::begin(rhs), std::begin(ret), std::multiplies{});
        return ret;
    }
    template<class T> T operator * (const typename T::value_type lhs, const T& rhs){
        return rhs * lhs;
    }
    template<class T> T operator * (const T& lhs, const typename T::value_type rhs){
        T ret = lhs;
        apply(std::begin(lhs), std::end(lhs), rhs, std::begin(ret), std::multiplies{});
        return ret;
    }
    template<class T> T abs(const T& lhs){
        T ret = lhs;
        apply(std::begin(lhs), std::end(lhs), std::begin(ret), [](auto& v){return std::abs(v);});
        return ret;
    }

    template<class T> bool almost_equal(const T& lhs, const T& rhs){
        T epsilon = lhs;
        std::fill(epsilon.begin(), epsilon.end(), std::numeric_limits<typename T::value_type>::epsilon());
        return abs(lhs - rhs) <= epsilon;
    }

    template<class T> bool operator == (const T& lhs, const T& rhs){
        return std::equal(std::begin(lhs), std::end(lhs), std::begin(rhs));
    }

    template<class T> bool operator <= (const T& lhs, const T& rhs){
        return std::equal(std::begin(lhs), std::end(lhs), std::begin(rhs), std::less_equal<typename T::value_type>{});
    }
    template<class T> T operator / (const T& lhs, const typename T::value_type rhs){
        T ret = lhs;
        apply(std::begin(lhs), std::end(lhs), rhs, std::begin(ret), std::divides{});
        return ret;
    }

    template<class T> auto norm2(const T& lhs) {
        const auto X = lhs*lhs;
        return std::accumulate(std::begin(X), std::end(X), (typename T::value_type) 0.0);
    }
    template<class T> auto norm(const T& lhs)  {
        return std::sqrt(norm2(lhs));
    }
    template<class T> auto normalize(const T& lhs) {
        return lhs * (static_cast<typename T::value_type>(1.0) / norm(lhs));
    }
    template<class T> auto dot(const T& lhs, const T& rhs) {
        return std::inner_product(std::cbegin(lhs), std::cend(lhs), std::begin(rhs), (typename T::value_type) 0.0);
    }
}

namespace opt {
    template<class InputIt> auto argmin(InputIt beg, InputIt end) noexcept {
        return std::distance(beg, std::min_element(beg, end));
    }
    template<class InputIt, class UnaryOp> auto argmin(InputIt beg, InputIt end, UnaryOp op) noexcept {
        std::vector<typename InputIt::value_type> v(beg, end);
        std::transform(v.begin(), v.end(), v.begin(), op);
        auto result = std::min_element(v.begin(), v.end());
        return std::distance(v.begin(), result);
    }
    inline std::vector<Real> solve_quadratic(Real a, Real b, Real c) noexcept {
        const Real delta = (b * b) - (static_cast<Real>(4.0)*a*c);
        const Real two_a = static_cast<Real>(2.0) * a;
        return delta >= 0 ?
            std::vector<Real>({(-b + std::sqrt(delta)) / (two_a), (-b - std::sqrt(delta)) / (two_a)}) :
            std::vector<Real>();
    }
}


