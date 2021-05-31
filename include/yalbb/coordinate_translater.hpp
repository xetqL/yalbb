//
// Created by xetql on 4/29/20.
//

#ifndef NBMPI_COORDINATE_TRANSLATER_HPP
#define NBMPI_COORDINATE_TRANSLATER_HPP

#include "utils.hpp"

class CoordinateTranslater {
public:
    static std::tuple<Integer, Integer, Integer>
    inline __translate_linear_index_into_xyz(const Integer index, const Integer ncols, const Integer nrows) {
        return {(index % ncols), static_cast<Integer>(std::floor(index / ncols)) % nrows, std::floor(index / (ncols * nrows))};    // depth
    };

    static auto translate_linear_index_into_xyz (const Integer i, const Integer ncols, const Integer nrows) {
        auto iZ = i / (ncols * nrows);
        auto remainder = i % (ncols * nrows);
        auto iY = remainder / ncols;
        auto iX = remainder % ncols;
        return std::make_tuple(iX, iY, iZ);
    };


    static std::array<Real, 3>
    inline translate_xyz_into_position(std::tuple<Integer, Integer, Integer>&& xyz, const Real rc) {
        return {std::get<0>(xyz) * rc, std::get<1>(xyz) * rc, std::get<2>(xyz) * rc};
    };

    static std::array<Real, 3>
    inline translate_xyz_into_position(std::array<Integer,3>& xyz, const Real rc) {
        return {xyz[0] * rc, xyz[1] * rc, xyz[2] * rc};
    };

    template<int N>
    static inline Integer translate_xyz_into_linear_index(const std::tuple<Integer, Integer, Integer> xyz, const BoundingBox<N>& bbox, const Real rc) {
        auto lc = get_cell_number_by_dimension<N>(bbox, rc);
        return std::get<0>(xyz) + std::get<1>(xyz) * lc[0] + lc[0] * lc[1] * std::get<2>(xyz);
    };

    template<int N> static Integer
    inline translate_position_into_local_index(const std::array<Real, N> &position, Real rc, const BoundingBox<N>& bbox, const Integer c, const Integer r){
        auto xyz = CoordinateTranslater::translate_position_into_local_xyz<N>(position, bbox, rc);
        int cell = xyz[0];
        cell += c * xyz[1];
        if constexpr(N==3) {
            cell += c * r *xyz[2];
        }
        return cell;
    }

    template<int N>
    static std::array<Real, 3>
    translate_local_xyz_into_position(std::tuple<Integer, Integer, Integer> local_index, const BoundingBox<N>& bbox, const Real rc){
        auto global_position = CoordinateTranslater::translate_xyz_into_position(std::forward<std::tuple<Integer, Integer, Integer>>(local_index), rc);
        for(auto i = 0; i < N; ++i)
            global_position[i] += bbox[2*i];
        return global_position;
    }

    template<int N>
    static std::array<Real, 3>
    translate_local_xyz_into_position(std::array<Integer, 3>& local_index, const BoundingBox<N>& bbox, const Real rc){
        auto local_position = CoordinateTranslater::translate_xyz_into_position(local_index, rc);
        for(auto i = 0; i < N; ++i) local_position[i] += bbox[2*i];
        return local_position;
    }


    template<int N> static std::array<Integer, N>
    translate_position_into_xyz(const std::array<Real, N>& position, Real rc){
        std::array<Integer, N> ret;
        std::transform(position.cbegin(), position.cend(), std::begin(ret), [rc](Real v){ return (Integer) ( (double) v / (double) rc ); });        return ret;
    }
    template<int N> static std::array<Integer, 3>
    position_into_xyz(const std::array<Real, N>& position, Real rc){
        std::array<Integer, 3> ret = {0,0, 0};
        std::transform(position.cbegin(), position.cend(), std::begin(ret), [rc](Real v){ return (Integer) ( (double) v / (double) rc ); });
        return ret;
    }

    template<int N> static std::array<Integer, N>
    translate_position_into_local_xyz(const std::array<Real, N>& position, const BoundingBox<N>& bbox, Real rc){
        std::array<Integer, N> ret;
        for(int i = 0; i<N; ++i) {
            ret[i] = std::floor(((double) position[i] - (double) bbox[2*i]) / (double) rc);
        }
        return ret;
    }
    template<int N> static std::array<Real, N>
    translate_global_index_into_position(Integer global_index){}
};


#endif //NBMPI_COORDINATE_TRANSLATER_HPP
