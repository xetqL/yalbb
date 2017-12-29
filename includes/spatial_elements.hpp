//
// Created by xetql on 12/28/17.
//

#ifndef NBMPI_GEOMETRIC_ELEMENT_HPP
#define NBMPI_GEOMETRIC_ELEMENT_HPP

#include <array>
#include <iostream>

namespace elements {

    template<int N>
    struct Element {
        std::array<double, N> position,  velocity;

        constexpr Element(std::array<double, N> p, std::array<double, N> v) : position(p), velocity(v){}
        constexpr Element() {
            std::fill(velocity.begin(), velocity.end(), 0.0);
            std::fill(position.begin(), position.end(), 0.0);
        }
        static constexpr int size() {
            return N * 2.0;
        }

        /**
         * Equality of two elements regarding the VALUE of their properties
         * @param rhs another element
         * @return true if the position and the velocity of the two elements are equals
         */
        bool operator==(const Element &rhs) const {
            return position == rhs.position && velocity == rhs.velocity;
        }

        bool operator!=(const Element &rhs) const {
            return !(rhs == *this);
        }

        friend std::ostream &operator<<(std::ostream &os, const Element &element) {
            std::string pos = "("+std::to_string(element.position.at(0));
            for(int i = 1; i < N; i++){
                pos += "," + std::to_string(element.position.at(i));
            }
            pos += ")";
            std::string vel = "("+std::to_string(element.velocity.at(0));
            for(int i = 1; i < N; i++){
                vel += "," + std::to_string(element.velocity.at(i));
            }
            vel += ")";
            os << "position: " << pos << " velocity: " << vel;
            return os;
        }

    };

}

#endif //NBMPI_GEOMETRIC_ELEMENT_HPP
