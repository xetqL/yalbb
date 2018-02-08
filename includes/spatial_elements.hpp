//
// Created by xetql on 12/28/17.
//

#ifndef NBMPI_GEOMETRIC_ELEMENT_HPP
#define NBMPI_GEOMETRIC_ELEMENT_HPP

#include <array>
#include <iostream>
#include <algorithm>

namespace elements {

    template<int N>
    struct Element {
        int identifier;
        std::array<double, N> position,  velocity, acceleration;

        static const int number_of_dimensions = N;
        constexpr Element(std::array<double, N> p, std::array<double, N> v, const int id) : identifier(id), position(p), velocity(v){
            std::fill(acceleration.begin(), acceleration.end(), 0.0);
        }
        constexpr Element(std::array<double, N> p, std::array<double, N> v, std::array<double,N> a, const int id) : identifier(id), position(p), velocity(v), acceleration(a){
            std::fill(acceleration.begin(), acceleration.end(), 0.0);
        }
        constexpr Element() : identifier(0){
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

        static Element<N> create(std::array<double, N> &p, std::array<double, N> &v, int id){
            Element<N> e(p, v, id);
            return e;
        }

        static Element<N> createc(std::array<double, N> p, std::array<double, N> v, int id){
            Element<N> e(p, v, id);
            return e;
        }

        template<class Distribution, class Generator>
        static Element<N> create_random( Distribution& dist, Generator &gen, int id){
            std::array<double, N> p, v;
            std::generate(p.begin(), p.end(), [&dist, &gen](){return dist(gen);});
            std::generate(v.begin(), v.end(), [&dist, &gen](){return dist(gen);});
            return Element::create(p, v, id);
        }

        template<class Distribution, class Generator, class RejectionPredicate>
        static Element<N> create_random( Distribution& dist, Generator &gen, int id, RejectionPredicate pred){
            std::array<double, N> p, v;
            //generate point in N dimension
            int trial = 0;
            do {
                if(trial >= 1000) throw std::runtime_error("Could not generate particle that satisfies the predicate. Try another distribution.");
                std::generate(p.begin(), p.end(), [&dist, &gen](){return dist(gen);});
                trial++;
            } while(!pred(p));
            //generate velocity in N dimension
            std::generate(v.begin(), v.end(), [&dist, &gen](){return dist(gen);});
            return Element::create(p, v, id);
        }

        template<class Distribution, class Generator, int Cnt>
        static std::array<Element<N>, Cnt> create_random_n( Distribution& dist, Generator &gen ){
            std::array<Element<N>, Cnt> elements;
            int id = 0;
            std::generate(elements.begin(), elements.end(), [&]()mutable {
                return Element<N>::create_random(dist, gen, id++);
            });
            return elements;
        }

        template<class Container, class Distribution, class Generator>
        static void create_random_n(Container &elements, Distribution& dist, Generator &gen ) {
            int id = 0;
            std::generate(elements.begin(), elements.end(), [&]()mutable {
                return Element<N>::create_random(dist, gen, id++);
            });
        }

        template<class Container, class Distribution, class Generator, class RejectionPredicate>
        static void create_random_n(Container &elements, Distribution& dist, Generator &gen, RejectionPredicate pred ) {
            //apply rejection sampling using the predicate given in parameter
            //construct a new function that does the work
            int id = 0;
            std::generate(elements.begin(), elements.end(), [&]() mutable {
                return Element<N>::create_random(dist, gen, id++, [&elements, pred](auto const el){
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
            return position == rhs.position && velocity == rhs.velocity && identifier == rhs.identifier;
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
            std::string acc = "("+std::to_string(element.acceleration.at(0));
            for(int i = 1; i < N; i++){
                acc += "," + std::to_string(element.acceleration.at(i));
            }
            acc += ")";
            os << "position: " << pos << " velocity: " << vel << " acceleration: " << acc << " id: " << element.identifier;
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

    template<int N>
    double distance2(const std::pair<std::pair<double, double>, std::pair<double, double>> &l, const Element<N> &el){
        return std::pow((l.second.second - l.first.second)*el.position.at(0) - (l.second.first - l.first.first)*el.position.at(1) + l.second.first*l.first.second - l.second.second*l.first.first, 2)/
               (std::pow((l.second.second - l.first.second),2) + std::pow((l.second.first - l.first.first),2));
    }

    template<int N>
    double distance2(const std::array<std::pair<double, double>, N> &domain, const Element<N> &e1){
        double width = std::abs(domain.at(0).second - domain.at(0).first);
        double height= std::abs(domain.at(1).second - domain.at(1).first);
        double x = (domain.at(0).second + domain.at(0).first)/2.0;
        double y = (domain.at(1).second + domain.at(1).first)/2.0;
        double dx = std::max(std::abs(e1.position.at(0) - x) - width / 2, 0.0);
        double dy = std::max(std::abs(e1.position.at(1) - y) - height / 2, 0.0);
        return dx * dx + dy * dy;
    }

    template<int N, typename T>
    std::vector<Element<N>> transform(const int length, const T* positions, const T* velocities) {
        std::vector<Element<N>> elements(length);
        for(int i=0; i < length; ++i){
            Element<N> e({positions[2*i], positions[2*i+1]}, {velocities[2*i],velocities[2*i+1]}, i);
            elements[i] = e;
        }
        return elements;
    }

    template<int N, typename T>
    std::vector<Element<N>> transform(const int length, const T* positions, const T* velocities, const T* acceleration) {
        std::vector<Element<N>> elements(length);
        for(int i=0; i < length; ++i){
            Element<N> e({positions[2*i], positions[2*i+1]}, {velocities[2*i],velocities[2*i+1]}, {acceleration[2*i], acceleration[2*i+1]}, i);
            elements[i] = e;
        }
        return elements;
    }

    template<int N, typename T>
    void transform(std::vector<Element<N>>& elements, const T* positions, const T* velocities) throw() {
        if(elements.empty()) {
            throw std::runtime_error("Can not transform data into an empty vector");
        }
        std::generate(elements.begin(), elements.end(), [i = 0, id=0, &positions, &velocities]() mutable {
            Element<N> e({positions[i], positions[i+1]}, {velocities[i], velocities[i+1]}, id);
            i=i+N;
            id++;
            return e;
        });
    }

    template<int N>
    void serialize_positions(const std::vector<Element<N>>& elements, double* positions){
        size_t element_id = 0;
        for (auto const& el : elements){
            for(size_t dim = 0; dim < N; ++dim)
                positions[element_id * N + dim] = el.position.at(dim);
            element_id++;
        }
    }

    template<int N, typename T>
    void serialize(const std::vector<Element<N>>& elements, T* positions, T* velocities, T* acceleration){
        size_t element_id = 0;
        for (auto const& el : elements){
            for(size_t dim = 0; dim < N; ++dim){
                positions[element_id * N + dim] = (double) el.position.at(dim);  positions[element_id * N + dim] = (double) el.position.at(dim);
                velocities[element_id * N + dim] = (double) el.velocity.at(dim); velocities[element_id * N + dim] = (double) el.velocity.at(dim);
                acceleration[element_id * N + dim] = (double) el.acceleration.at(dim);  acceleration[element_id * N + dim] = (double) el.acceleration.at(dim);
            }
            element_id++;
        }
    }

    template<int N>
    bool is_inside(const Element<N> &element, const std::array<std::pair<double, double>, N> domain){
        auto element_position = element.position;

        for(size_t dim = 0; dim < N; ++dim){
            if(element_position.at(dim) < domain.at(dim).first || domain.at(dim).second < element_position.at(dim)) return false;
        }

        return true;
    }

}

#endif //NBMPI_GEOMETRIC_ELEMENT_HPP
