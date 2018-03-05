//
// Created by xetql on 12/28/17.
//

#ifndef NBMPI_GEOMETRIC_ELEMENT_HPP
#define NBMPI_GEOMETRIC_ELEMENT_HPP

#include <array>
#include <iostream>
#include <algorithm>
#include <mpi.h>
#include "partitioner.hpp"

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

        static constexpr int byte_size() {
            return N * 3 * sizeof(double) + sizeof(int);
        }

        std::string to_communication_buffer(){
            std::ostringstream comm_buf;
            for(int i = 0; i < N-1; ++i)
                comm_buf << std::to_string(position[i]) << " ";
            comm_buf << std::to_string(position[N-1]);

            comm_buf << ";";

            for(int i = 0; i < N-1; ++i)
                comm_buf << std::to_string(velocity[i]) << " ";
            comm_buf << std::to_string(velocity[N-1]);

            comm_buf << ";";

            for(int i = 0; i < N-1; ++i)
                comm_buf << std::to_string(acceleration[i]) << " ";
            comm_buf << std::to_string(acceleration[N-1]);

            comm_buf << ";";

            comm_buf << std::to_string(identifier);

            comm_buf << "!";


            return comm_buf.str();

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

    template <int N, typename RealType>
    void init_particles_random_v(std::vector<elements::Element<N>> &elements, RealType T0, int seed = 0) {

        int n = elements.size();
        //std::random_device rd; //Will be used to obtain a seed for the random number engine
        std::mt19937 gen(seed); //Standard mersenne_twister_engine seeded with rd()
        std::uniform_real_distribution<double> udist(0.0, 1.0);
        for (int i = 0; i < n; ++i) {
            double R = T0 * std::sqrt(-2 * std::log(udist(gen)));
            double T = 2 * M_PI * udist(gen);
            elements[i].velocity[0] = (double) (R * std::cos(T));
            elements[i].velocity[1] = (double) (R * std::sin(T));
        }
    }

    template<int N>
    partitioning::CommunicationDatatype register_datatype() {

        MPI_Datatype element_datatype,
                vec_datatype,
                range_datatype,
                domain_datatype,
                oldtype_range[1],
                oldtype_element[2];

        MPI_Aint offset[2], intex;

        int blockcount_element[2], blockcount_range[1];

        // register particle element type
        int array_size = N;
        MPI_Type_contiguous(array_size, MPI_DOUBLE, &vec_datatype);
        MPI_Type_commit(&vec_datatype);

        blockcount_element[0] = 1;
        blockcount_element[1] = 3; //position, velocity, acceleration

        oldtype_element[0] = MPI_INT;
        oldtype_element[1] = vec_datatype;

        MPI_Type_extent(MPI_INT, &intex);
        offset[0] = static_cast<MPI_Aint>(0);
        offset[1] = intex;

        MPI_Type_struct(2, blockcount_element, offset, oldtype_element, &element_datatype);
        MPI_Type_commit(&element_datatype);

        blockcount_range[0] = N;
        oldtype_range[0] = MPI_DOUBLE;
        MPI_Type_struct(1, blockcount_range, offset, oldtype_range, &range_datatype);
        MPI_Type_commit(&range_datatype);

        MPI_Type_contiguous(N, range_datatype, &domain_datatype);
        MPI_Type_commit(&domain_datatype);

        return partitioning::CommunicationDatatype(vec_datatype, element_datatype, range_datatype, domain_datatype);
    }
}

#endif //NBMPI_GEOMETRIC_ELEMENT_HPP
