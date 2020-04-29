//
// Created by xetql on 4/29/20.
//

#ifndef NBMPI_COORDINATE_TRANSLATER_HPP
#define NBMPI_COORDINATE_TRANSLATER_HPP

#include "utils.hpp"

class CoordinateTranslater {
public:
    static std::tuple<Integer, Integer, Integer>
    inline translate_linear_index_into_xyz(const Integer index, const Integer ncols, const Integer nrows) {
        return {(index % ncols), std::floor(index % (ncols * nrows) / nrows), std::floor(index / (ncols * nrows))};    // depth
    };
    static std::tuple<Real, Real, Real>
    inline translate_xyz_into_position(const std::tuple<Integer, Integer, Integer>&& xyz, const Real rc) {
        return {std::get<0>(xyz) * rc, std::get<1>(xyz) * rc, std::get<2>(xyz) * rc};
    };

    template<int N>
    static inline Integer translate_xyz_into_linear_index(const std::tuple<Integer, Integer, Integer>&& xyz, const BoundingBox<N>& bbox, const Real rc) {
        auto lc = get_cell_number_by_dimension<N>(bbox, rc);
        return std::get<0>(xyz) + std::get<1>(xyz) * lc[0] + lc[0]*lc[1]*std::get<2>(xyz);
    };
    template<int N> static Integer
    translate_position_into_local_index(const std::array<Real, N>& position){}
    template<int N> static std::array<Real, N>
    translate_local_index_into_position(const Integer local_index, const BoundingBox<N>& bbox, const Real rc){
        auto lc = get_cell_number_by_dimension<N>(bbox, rc);
        auto[local_pos_x,local_pos_y,local_pos_z] = translate_xyz_into_position(translate_linear_index_into_xyz(local_index, lc[0], lc[1]), rc);
        std::array<Real, N> position;
        position[0] = local_pos_x+bbox[0];
        position[1] = local_pos_x+bbox[2];
        if constexpr(N==3)
            position[2] = local_pos_x+bbox[4];
        return position;
    }
    template<int N> static Integer
    translate_position_into_global_index(const std::array<Real, N>& position){}
    template<int N> static std::array<Real, N>
    translate_global_index_into_position(Integer global_index){}
};


#endif //NBMPI_COORDINATE_TRANSLATER_HPP