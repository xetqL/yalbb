//
// Created by xetql on 4/29/20.
//

#ifndef NBMPI_COORDINATE_TRANSLATER_HPP
#define NBMPI_COORDINATE_TRANSLATER_HPP

#include "utils.hpp"

namespace CoordinateTranslater {
    using XYZ               = std::tuple<Integer,Integer,Integer>;
    using XYZPositionArray  = std::array<Real, 3>;

    XYZ translate_linear_index_into_xyz (Integer i, Integer ncols, Integer nrows);

    XYZPositionArray
    translate_xyz_into_position(XYZ&& xyz, Real rc);

    template<int N> std::array<Integer, N>
    translate_position_into_local_xyz(const std::array<Real, N>& position, const BoundingBox<N>& bbox, Real rc){
        std::array<Integer, N> ret;
        for(int i = 0; i<N; ++i) {
            ret[i] = std::floor(((double) position[i] - (double) bbox[2*i]) / (double) rc);
        }
        return ret;
    }

    template<int N> Integer
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
    XYZPositionArray
    translate_local_xyz_into_position(XYZ local_index, const BoundingBox<N>& bbox, const Real rc){
        auto global_position = CoordinateTranslater::translate_xyz_into_position(std::forward<XYZ>(local_index), rc);
        for(auto i = 0; i < N; ++i)
            global_position[i] += bbox[2*i];
        return global_position;
    }


/* DEPRECATED
    static std::array<Real, 3>
    inline translate_xyz_into_position(std::array<Integer,3>& xyz, const Real rc);
    static std::tuple<Integer, Integer, Integer>
    inline _translate_linear_index_into_xyz(const Integer index, const Integer ncols, const Integer nrows);
    template<int N>
    static inline Integer translate_xyz_into_linear_index(const std::tuple<Integer, Integer, Integer> xyz, const BoundingBox<N>& bbox, const Real rc) {
        auto lc = get_cell_number_by_dimension<N>(bbox, rc);
        return std::get<0>(xyz) + std::get<1>(xyz) * lc[0] + lc[0] * lc[1] * std::get<2>(xyz);
    };
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
    template<int N> static std::array<Real, N>
    translate_global_index_into_position(Integer global_index){}
*/
};


#endif //NBMPI_COORDINATE_TRANSLATER_HPP
