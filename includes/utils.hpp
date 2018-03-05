//
// Created by xetql on 13.12.17.
//

#ifndef NBMPI_UTILS_HPP
#define NBMPI_UTILS_HPP

#include <ctime>
#include <vector>
#include <stdexcept>

#define TIME_IT(a){\
 double start = MPI_Wtime();\
 a;\
 double end = MPI_Wtime();\
 auto diff = (end - start) / MPI_Wtick();\
 std::cout << std::scientific << diff << std::endl;\
};\

inline std::string get_date_as_string() {
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%d-%m-%Y-%H-%M-%S");
    std::string date = oss.str();
    return date;
}

namespace partitioning { namespace utils {
        template<typename A, typename B>
        const std::vector<std::pair<A, B>> zip(const std::vector<A>& a, const std::vector<B>& b){
            std::vector<std::pair<A,B>> zipAB;
            int sizeAB = a.size();
            for(int i = 0; i < sizeAB; ++i)
                zipAB.push_back(std::make_pair(a.at(i), b.at(i)));
            return zipAB;
        }

        template<typename A, typename B>
        const std::pair<std::vector<A>, std::vector<B>> unzip(const std::vector<std::pair<A,B>>& ab){
            std::vector<A> left;
            std::vector<B> right;
            int sizeAB = ab.size();
            for(int i = 0; i < sizeAB; ++i){
                auto pair = ab.at(i);
                left.push_back(pair.first);
                right.push_back(pair.second);
            }
            return std::make_pair(left, right);
        }

        /**
         * Copied from https://stackoverflow.com/questions/17294629/merging-flattening-sub-vectors-into-a-single-vector-c-converting-2d-to-1d
         * @tparam R Return Container class
         * @tparam Top Top container class from the container
         * @tparam Sub Sub class deduced from the original container
         * @param all Container that contains the sub containers
         * @return flattened container
         */

        template <template<typename...> class R=std::vector,
                typename Top,
                typename Sub = typename Top::value_type>
        R<typename Sub::value_type> flatten(Top const& all)
        {
            using std::begin;
            using std::end;
            R<typename Sub::value_type> accum;
            for(auto& sub : all)
                std::copy(begin(sub), end(sub), std::inserter(accum, end(accum)));
            return accum;
        }
}   }

#endif //NBMPI_UTILS_HPP
