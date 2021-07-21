//
// Created by xetql on 7/21/21.
//

#include <coordinate_translater.hpp>

CoordinateTranslater::XYZ CoordinateTranslater::translate_linear_index_into_xyz(const Integer i, const Integer ncols, const Integer nrows)  {
    auto iZ = i / (ncols * nrows);
    auto remainder = i % (ncols * nrows);
    auto iY = remainder / ncols;
    auto iX = remainder % ncols;
    return std::make_tuple(iX, iY, iZ);
}

CoordinateTranslater::XYZPositionArray
inline CoordinateTranslater::translate_xyz_into_position(std::tuple<Integer, Integer, Integer> &&xyz, const Real rc) {
    return {std::get<0>(xyz) * rc, std::get<1>(xyz) * rc, std::get<2>(xyz) * rc};
}