//
// Created by xetql on 2/5/18.
//

#ifndef NBMPI_OUTPUT_FORMATTER_HPP
#define NBMPI_OUTPUT_FORMATTER_HPP

#include <cstdio>
#include <cstdint>
#include <vector>
#include <iostream>     // std::cout, std::fixed, std::scientific
#include <fstream>

#include "params.hpp"
#include "utils.hpp"
struct SimpleCSVFormatter {
    const char separator;
    SimpleCSVFormatter(char separator) : separator(separator){}
    template<int N> inline void write_data(std::ostream &stream, const std::array<Real, N>& pos){
        stream << pos.at(0) << separator << pos.at(1);
        if constexpr (N == 3) stream << separator <<  pos.at(2);
        stream << std::endl;
    }
    inline void write_header(std::ostream &stream, const int n, float simsize){
        configure_stream(stream);
    }
    template<int N> inline void write_frame_header(std::ostream &stream){
        stream << "x coord" << separator << "y coord";
        if constexpr (N == 3) stream << separator << "z coord";
        stream << std::endl;
    }
private:
    inline void configure_stream(std::ostream &stream, int precision = 6){
        stream << std::fixed << std::setprecision(6);
    }
};

template<int N, class T, class GetDataFunc>
void write_frame_data(std::ostream &stream, std::vector<T>& els, GetDataFunc getDataFunc, SimpleCSVFormatter& formatter) {
    formatter.write_frame_header<N>(stream);
    for(const T &el : els ) {
        formatter.write_data<N>(stream, getDataFunc(el));
    }
}

#endif //NBMPI_OUTPUT_FORMATTER_HPP
