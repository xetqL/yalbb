//
// Created by xetql on 7/21/21.
//

#include <output_formatter.hpp>

void SimpleCSVFormatter::configure_stream(std::ostream &stream, int precision) {
    stream << std::fixed << std::setprecision(precision);
}

void SimpleCSVFormatter::write_header(std::ostream &stream, int n, float simsize) {
    SimpleCSVFormatter::configure_stream(stream);
}

