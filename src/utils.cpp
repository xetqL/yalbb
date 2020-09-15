//
// Created by xetql on 4/29/20.
//
#include "utils.hpp"
Real io::str_to_real(const std::string& str){
    if constexpr (std::is_same<Real, double>::value){
        return std::stod(str);
    } else {
        return std::stof(str);
    }
}
inline std::string get_date_as_string() {
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%d-%m-%Y-%H-%M-%S");
    std::string date = oss.str();
    return date;
}

bool file_exists(const std::string fileName) {
    std::ifstream infile(fileName);
    return infile.good();
}

std::vector<std::string> split(const std::string &s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}
