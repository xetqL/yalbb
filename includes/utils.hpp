//
// Created by xetql on 13.12.17.
//

#ifndef NBMPI_UTILS_HPP
#define NBMPI_UTILS_HPP

#include <vector>
#include <stdexcept>

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

        template<typename A>
        const std::vector<A> flatten(const std::vector<std::vector<A>>& to_flatten){
            int total_size = 0;
            for(auto const& col: to_flatten) total_size += col.size();
            std::vector<A> flattened(total_size);
            typename std::vector<A>::iterator it = flattened.begin();
            for(auto const& col: to_flatten) {
                //copy the
                std::copy(col.begin(), col.end(), it);
                std::advance(it, col.size());
            }
            return std::move(flattened);
        }
}   }

#endif //NBMPI_UTILS_HPP
