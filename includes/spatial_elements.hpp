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

        static const int number_of_dimensions = N;
        std::array<double, N> position,  velocity, acceleration;

        constexpr Element(std::array<double, N> p, std::array<double, N> v) : position(p), velocity(v){
            std::fill(acceleration.begin(), acceleration.end(), 0.0);
        }
        constexpr Element() {
            std::fill(velocity.begin(), velocity.end(), 0.0);
            std::fill(position.begin(), position.end(), 0.0);
            std::fill(acceleration.begin(), acceleration.end(), 0.0);
        }
        /**
         * Total size of the structure
         * @return The number of element per dimension times the number of characteristics (3)
         */
        static constexpr int size() {
            return N * 3;
        }

        static Element<N> create(std::array<double, N> &p, std::array<double, N> &v ){
            Element<N> e(p, v);
            return e;
        }

        template<class Distribution, class Generator>
        static Element<N> create_random( Distribution& dist, Generator &gen ){
            std::array<double, N> p, v;
            std::generate(p.begin(), p.end(), [&dist, &gen](){return dist(gen);});
            std::generate(v.begin(), v.end(), [&dist, &gen](){return dist(gen);});
            return Element::create(p, v);
        }

        template<class Distribution, class Generator, class RejectionPredicate>
        static Element<N> create_random( Distribution& dist, Generator &gen, RejectionPredicate pred){
            std::array<double, N> p, v;
            do{
                //generate a single point in N dimension
                std::generate(p.begin(), p.end(), [&dist, &gen](){return dist(gen);});
            } while(!pred(p));
            //generate a single velocity in N dimension
            std::generate(v.begin(), v.end(), [&dist, &gen](){return dist(gen);});
            return Element::create(p, v);
        }

        template<class Distribution, class Generator, int Cnt>
        static std::array<Element<N>, Cnt> create_random_n( Distribution& dist, Generator &gen ){
            std::array<Element<N>, Cnt> elements;
            std::generate(elements.begin(), elements.end(), [=]()mutable {
                return Element<N>::create_random(dist, gen);
            });
            return elements;
        }

        template<class Container, class Distribution, class Generator>
        static void create_random_n(Container &elements, Distribution& dist, Generator &gen ) {
            std::generate(elements.begin(), elements.end(), [=]()mutable {
                return Element<N>::create_random(dist, gen);
            });
        }

        template<class Container, class Distribution, class Generator, class RejectionPredicate>
        static void create_random_n(Container &elements, Distribution& dist, Generator &gen, RejectionPredicate pred ) {
            //apply rejection sampling using the predicate given in parameter
            //construct a new function that does the work
            std::generate(elements.begin(), elements.end(), [&]() mutable {
                return Element<N>::create_random(dist, gen, [&elements, pred](auto const el){
                       return std::all_of(elements.begin(), elements.end(), [&](auto p){return pred(p.position, el);});
                });
            });
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

    template<int N>
    double distance2(std::array<double, N> e1, std::array<double, N> e2) {
        double r2 = 0.0;
        for(int i = 0; i < N; ++i){
            r2 += std::pow(e1.at(i) - e2.at(i), 2);
        }
        return r2;
    }

    template<int N, typename T>
    std::vector<Element<N>> transform(const int length, const T* positions, const T* velocities) {
        std::vector<Element<N>> elements(length);
        for(int i=0; i < length; ++i){
            Element<N> e({positions[2*i], positions[2*i+1]}, {velocities[2*i],velocities[2*i+1]});
            elements[i] = e;
        }
        return elements;
    }

#include <string>
    template<int N, typename T>
    void transform(std::vector<Element<N>>& elements, const T* positions, const T* velocities) throw() {
        if(elements.empty()) {
            throw std::runtime_error("Can not transform data into an empty vector");
        }
        std::generate(elements.begin(), elements.end(), [i = 0, &positions, &velocities]() mutable {
            Element<N> e({positions[i], positions[i+1]}, {velocities[i], velocities[i+1]});
            i=i+N;
            return e;
        });
    }
}

#endif //NBMPI_GEOMETRIC_ELEMENT_HPP
