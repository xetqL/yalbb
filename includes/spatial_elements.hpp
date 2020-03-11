//
// Created by xetql on 12/28/17.
//

#ifndef NBMPI_GEOMETRIC_ELEMENT_HPP
#define NBMPI_GEOMETRIC_ELEMENT_HPP

#include <array>
#include <iostream>
#include <algorithm>
#include <mpi.h>
#include <type_traits>

#include "utils.hpp"
#include "communication_datatype.hpp"

namespace elements {

    using ElementRealType = Real;

    template<int N>
    struct Element {
        int gid;
        int lid;
        std::array<ElementRealType, N> position,  velocity, acceleration;

        static const int number_of_dimensions = N;

        constexpr Element(std::array<ElementRealType, N> p, std::array<ElementRealType, N> v, const int gid, const int lid) : gid(gid), lid(lid), position(p), velocity(v), acceleration(){
            //std::fill(acceleration.begin(), acceleration.end(), 0.0);
        }

        constexpr Element(std::array<ElementRealType, N> p, std::array<ElementRealType, N> v, std::array<ElementRealType,N> a, const int gid, const int lid) : gid(gid), lid(lid), position(p), velocity(v), acceleration(a){
            //std::fill(acceleration.begin(), acceleration.end(), 0.0);
        }

        constexpr Element() : gid(0), lid(0), position(), velocity(), acceleration() {
            //std::fill(velocity.begin(), velocity.end(), 0.0);
            //std::fill(position.begin(), position.end(), 0.0);
            //std::fill(acceleration.begin(), acceleration.end(), 0.0);
        }

        /**
         * Total size of the structure
         * @return The number of element per dimension times the number of characteristics (3)
         */
        static constexpr int size() {
            return N * 3;
        }

        static constexpr int byte_size() {
            return N * 3 * sizeof(ElementRealType) + 2 * sizeof(int);
        }

        static Element<N> create(std::array<ElementRealType, N> &p, std::array<ElementRealType, N> &v, int gid, int lid){
            Element<N> e(p, v, gid, lid);
            return e;
        }

        static Element<N> createc(std::array<ElementRealType, N> p, std::array<ElementRealType, N> v, int gid, int lid){
            Element<N> e(p, v, gid, lid);
            return e;
        }

        template<class Distribution, class Generator>
        static Element<N> create_random( Distribution& dist, Generator &gen, int gid, int lid){
            std::array<ElementRealType, N> p, v;
            std::generate(p.begin(), p.end(), [&dist, &gen](){return dist(gen);});
            std::generate(v.begin(), v.end(), [&dist, &gen](){return dist(gen);});
            return Element::create(p, v, gid, lid);
        }

        template<class Distribution, class Generator>
        static Element<N> create_random( Distribution& distx, Distribution& disty, Distribution& distz, Generator &gen, int gid, int lid){
            std::array<ElementRealType, N> p, v;
            p[0] = distx(gen);
            p[1] = disty(gen);
            if(N > 2) p[2] = distz(gen);

            return Element::create(p, v, gid, lid);
        }

        template<class Distribution, class Generator, class RejectionPredicate>
        static Element<N> create_random( Distribution& dist, Generator &gen, int gid, int lid, RejectionPredicate pred){
            std::array<ElementRealType, N> p, v;
            //generate point in N dimension
            int trial = 0;
            do {
                if(trial >= 1000) throw std::runtime_error("Could not generate particles that satisfy the predicate. Try another distribution.");
                std::generate(p.begin(), p.end(), [&dist, &gen](){return dist(gen);});
                trial++;
            } while(!pred(p));
            //generate velocity in N dimension
            std::generate(v.begin(), v.end(), [&dist, &gen](){return dist(gen);});
            return Element::create(p, v, gid, lid);
        }

        template<class Distribution, class Generator, int Cnt>
        static std::array<Element<N>, Cnt> create_random_n( Distribution& dist, Generator &gen ){
            std::array<Element<N>, Cnt> elements;
            int id = 0;
            std::generate(elements.begin(), elements.end(), [&]()mutable {
                return Element<N>::create_random(dist, gen, id, id++);
            });
            return elements;
        }

        template<class Container, class Distribution, class Generator>
        static void create_random_n(Container &elements, Distribution& dist, Generator &gen ) {
            int id = 0;
            std::generate(elements.begin(), elements.end(), [&]()mutable {
                return Element<N>::create_random(dist, gen, id, id++);
            });
        }

        template<class Container, class Distribution, class Generator, class RejectionPredicate>
        static void create_random_n(Container &elements, Distribution& dist, Generator &gen, RejectionPredicate pred ) {
            //apply rejection sampling using the predicate given in parameter
            //construct a new function that does the work
            int id = 0;
            std::generate(elements.begin(), elements.end(), [&]() mutable {
                return Element<N>::create_random(dist, gen, id, id++, [&elements, pred](auto const el) {
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
            return position == rhs.position && velocity == rhs.velocity && gid == rhs.gid;
        }

        bool operator!=(const Element &rhs) const {
            return !(rhs == *this);
        }

        friend std::ostream &operator<<(std::ostream &os, const Element &element) {
            std::string pos = std::to_string(element.position.at(0));
            for(int i = 1; i < N; i++){
                pos += " " + std::to_string(element.position.at(i));
            }

            std::string vel = std::to_string(element.velocity.at(0));
            for(int i = 1; i < N; i++){
                vel += " " + std::to_string(element.velocity.at(i));
            }

            std::string acc = std::to_string(element.acceleration.at(0));
            for(int i = 1; i < N; i++){
                acc += " " + std::to_string(element.acceleration.at(i));
            }

            os << pos << ";" << vel << ";" << acc << ";" << element.gid << ";" << element.lid;
            return os;
        }

        std::string to_string(Real lsub) {
            std::string pos = std::to_string(this->position.at(0));
            for(int i = 1; i < N; i++){
                pos += " " + std::to_string(this->position.at(i));
            }

            std::string idx = std::to_string(std::floor(this->position.at(0) / lsub));
            for(int i = 1; i < N; i++){
                idx += " " + std::to_string(std::floor(this->position.at(i) / lsub));
            }

            std::string vel = std::to_string(this->velocity.at(0));
            for(int i = 1; i < N; i++){
                vel += " " + std::to_string(this->velocity.at(i));
            }

            std::string acc = std::to_string(this->acceleration.at(0));
            for(int i = 1; i < N; i++){
                acc += " " + std::to_string(this->acceleration.at(i));
            }



            return "(" + pos + ") " + "(" + idx + ") " + "(" + vel + ") " + "(" + acc + ") " + std::to_string(this->gid) + ";";
        }

    };

    template<int N>
    using BoundingBox = std::array<Real, 2*N>;

    template<int dim, int N>
    constexpr Real get_size(const BoundingBox<N>& bbox) {
        return bbox[2*dim+1] - bbox[2*dim];
    }

    template<int N>
    BoundingBox<N> get_bounding_box(const std::vector<elements::Element<N>>& elements, Real rc){
        BoundingBox<N> new_bbox = {std::numeric_limits<Real>::max(), std::numeric_limits<Real>::lowest(),
                                   std::numeric_limits<Real>::max(), std::numeric_limits<Real>::lowest(),
                                   std::numeric_limits<Real>::max(), std::numeric_limits<Real>::lowest()};
        for(const auto& el : elements) {
            new_bbox[0] = std::min(new_bbox[0], el.position[0]);
            new_bbox[1] = std::max(new_bbox[1], el.position[0]);
            new_bbox[2] = std::min(new_bbox[2], el.position[1]);
            new_bbox[3] = std::max(new_bbox[3], el.position[1]);
            new_bbox[4] = std::min(new_bbox[4], el.position[2]);
            new_bbox[5] = std::max(new_bbox[5], el.position[2]);
        }

        /* hook to grid, resulting bbox is divisible by lc[i] forall i */
        for(int i = 0; i < N; ++i) {
            new_bbox[2*i]   = std::max((Real) 0.0, std::floor(new_bbox[2*i] / rc) * rc - 2*rc);
            new_bbox[2*i+1] = std::ceil(new_bbox[2*i+1] / rc) * rc + 2*rc;
        }

        return new_bbox;
    }

    template<int N>
    BoundingBox<N> get_bounding_box(const std::vector<elements::Element<N>>& elements1,
                                    const std::vector<elements::Element<N>>& elements2,Real rc){
        BoundingBox<N> new_bbox = {std::numeric_limits<Real>::max(), std::numeric_limits<Real>::lowest(),
                                   std::numeric_limits<Real>::max(), std::numeric_limits<Real>::lowest(),
                                   std::numeric_limits<Real>::max(), std::numeric_limits<Real>::lowest()};
        for(const auto& el : elements1) {
            new_bbox[0] = std::min(new_bbox[0], el.position[0]);
            new_bbox[1] = std::max(new_bbox[1], el.position[0]);
            new_bbox[2] = std::min(new_bbox[2], el.position[1]);
            new_bbox[3] = std::max(new_bbox[3], el.position[1]);
            new_bbox[4] = std::min(new_bbox[4], el.position[2]);
            new_bbox[5] = std::max(new_bbox[5], el.position[2]);
        }

        for(const auto& el : elements2) {
            new_bbox[0] = std::min(new_bbox[0], el.position[0]);
            new_bbox[1] = std::max(new_bbox[1], el.position[0]);
            new_bbox[2] = std::min(new_bbox[2], el.position[1]);
            new_bbox[3] = std::max(new_bbox[3], el.position[1]);
            new_bbox[4] = std::min(new_bbox[4], el.position[2]);
            new_bbox[5] = std::max(new_bbox[5], el.position[2]);
        }

        /* hook to grid, resulting bbox is divisible by lc[i] forall i */
        for(int i = 0; i < N; ++i) {
            new_bbox[2*i]   = std::max((Real) 0.0, std::floor(new_bbox[2*i] / rc) * rc - 2*rc);
            new_bbox[2*i+1] = std::ceil(new_bbox[2*i+1] / rc) * rc + 2*rc;
        }

        return new_bbox;
    }

    template<int N>
    std::array<Integer, N> get_cell_number_by_dimension(const BoundingBox<N>& bbox, Real rc) {
        std::array<Integer, N> lc;
        lc [0] = get_size<0, N>(bbox) / rc;
        lc [1] = get_size<1, N>(bbox) / rc;
        if constexpr(N==3) lc [2] = get_size<2, N>(bbox) / rc;
        return lc;
    }

    template<int N>
    Integer get_total_cell_number(const BoundingBox<N>& bbox, Real rc){
        auto lc = get_cell_number_by_dimension<N>(bbox, rc);
        return std::accumulate(lc.begin(), lc.end(), 1, [](auto prev, auto v){return prev*v;});
    }

    template<int N>
    void import_from_file_float(std::string filename, std::vector<Element<N>>& particles) {

        std::ifstream pfile;
        pfile.open(filename, std::ifstream::in);
        if(!pfile.good()) throw std::runtime_error("bad particle file");

        std::string line;
        while (std::getline(pfile, line)) {
            auto parameters = split(line, ';');
            auto str_pos = split(parameters[0], ' ');
            auto str_vel = split(parameters[1], ' ');
            auto str_acc = split(parameters[2], ' ');
            auto str_gid = parameters[3];
            auto str_lid = parameters[4];
            Element<N> e;

            for(int i = 0; i < N; ++i)
                e.position[i] = std::stof(str_pos[i], 0);
            for(int i = 0; i < N; ++i)
                e.velocity[i] = std::stof(str_vel[i], 0);
            for(int i = 0; i < N; ++i)
                e.acceleration[i] = std::stof(str_acc[i], 0);

            e.gid = std::stoi(str_gid);
            e.lid = std::stoi(str_lid);
            particles.push_back(e);
        }
    }

    template<int N>
    void import_from_file_double(std::string filename, std::vector<Element<N>>& particles) {

        std::ifstream pfile;
        pfile.open(filename, std::ifstream::in);
        if(!pfile.good()) throw std::runtime_error("bad particle file");

        std::string line;
        while (std::getline(pfile, line)) {
            auto parameters = split(line, ';');
            auto str_pos = split(parameters[0], ' ');
            auto str_vel = split(parameters[1], ' ');
            auto str_acc = split(parameters[2], ' ');
            auto str_gid = parameters[3];
            auto str_lid = parameters[4];
            Element<N> e;

            for(int i = 0; i < N; ++i)
                e.position[i] = std::stod(str_pos[i], 0);
            for(int i = 0; i < N; ++i)
                e.velocity[i] = std::stod(str_vel[i], 0);
            for(int i = 0; i < N; ++i)
                e.acceleration[i] = std::stod(str_acc[i], 0);

            e.gid = std::stoi(str_gid);
            e.lid = std::stoi(str_lid);
            particles.push_back(e);
        }
    }

    template<int N, class RealType, bool UseDoublePrecision = std::is_same<RealType, double>::value>
    void import_from_file(std::string filename, std::vector<Element<N>>& particles) {
        if(UseDoublePrecision) import_from_file_double<N>(filename, particles);
        else import_from_file_float<N>(filename, particles);
    }

    template<int N>
    void export_to_file(std::string filename, const std::vector<Element<N>> elements) {
        std::ofstream particles_data;
        if (file_exists(filename)) std::remove(filename.c_str());
        particles_data.open(filename, std::ofstream::out);
        for(auto const& e : elements) {
            particles_data << e << std::endl;
        }
        particles_data.close();
    }

    template<int N>
    const inline ElementRealType distance2(const std::array<ElementRealType, N>& e1, const std::array<ElementRealType, N>& e2)  {

        std::array<Real, N> e1e2;
        for(int i = 0; i < N; ++i) e1e2[i] = e1[0] - e2[0];
        return std::accumulate(e1e2.cbegin(), e1e2.cend(), 0.0, [](Real prev, Real v){ return prev + v*v; });
    }

    template<int N>
    const inline ElementRealType distance2(const elements::Element<N> &e1, const elements::Element<N> &e2)  {
        return elements::distance2<N>(e1.position, e2.position);
    }

    template<int N, typename T>
    std::vector<Element<N>> transform(const int length, const T* positions, const T* velocities) {
        std::vector<Element<N>> elements(length);
        for(int i=0; i < length; ++i){
            Element<N> e({positions[2*i], positions[2*i+1]}, {velocities[2*i],velocities[2*i+1]}, i, i);
            elements[i] = e;
        }
        return elements;
    }

    template<int N, typename T>
    std::vector<Element<N>> transform(const int length, const T* positions, const T* velocities, const T* acceleration) {
        std::vector<Element<N>> elements;
        elements.reserve(length);
        for(int i=0; i < length; ++i) {
            elements.emplace_back({positions[2*i], positions[2*i+1]}, {velocities[2*i],velocities[2*i+1]}, {acceleration[2*i], acceleration[2*i+1]}, i, i);
        }
        return elements;
    }

    template<int N, typename T>
    void transform(std::vector<Element<N>>& elements, const T* positions, const T* velocities) throw() {
        if(elements.empty()) {
            throw std::runtime_error("Can not transform data into an empty vector");
        }
        std::generate(elements.begin(), elements.end(), [i = 0, id=0, &positions, &velocities]() mutable {
            Element<N> e({positions[i], positions[i+1]}, {velocities[i], velocities[i+1]}, id, id);
            i=i+N;
            id++;
            return e;
        });
    }

    template<int N>
    void serialize_positions(const std::vector<Element<N>>& elements, ElementRealType* positions){
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
                positions[element_id * N + dim] = (ElementRealType) el.position.at(dim);  positions[element_id * N + dim] = (ElementRealType) el.position.at(dim);
                velocities[element_id * N + dim] = (ElementRealType) el.velocity.at(dim); velocities[element_id * N + dim] = (ElementRealType) el.velocity.at(dim);
                acceleration[element_id * N + dim] = (ElementRealType) el.acceleration.at(dim);  acceleration[element_id * N + dim] = (ElementRealType) el.acceleration.at(dim);
            }
            element_id++;
        }
    }

    template<int N>
    bool is_inside(const Element<N> &element, const std::array<std::pair<ElementRealType, ElementRealType>, N> domain){
        auto element_position = element.position;

        for(size_t dim = 0; dim < N; ++dim){
            if(element_position.at(dim) < domain.at(dim).first || domain.at(dim).second < element_position.at(dim)) return false;
        }

        return true;
    }

    template <int N, typename RealType>
    void init_particles_random_v(std::vector<elements::Element<N>> &elements, RealType T0, int seed = 0) {
        int n = elements.size();
        std::mt19937 gen(seed); //Standard mersenne_twister_engine seeded with rd()
        std::normal_distribution<ElementRealType> ndist(0.0, T0 * T0);
        for (int i = 0; i < n; ++i) {
            elements[i].velocity[0] = ndist(gen);
            elements[i].velocity[1] = ndist(gen);
            if (N == 3) elements[i].velocity[2] = ndist(gen);
        }
    }

    template<int N, bool UseDoublePrecision = std::is_same<ElementRealType, double>::value>
    CommunicationDatatype register_datatype() {
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
        auto mpi_raw_datatype = UseDoublePrecision ? MPI_DOUBLE : MPI_FLOAT;

        MPI_Type_contiguous(array_size, mpi_raw_datatype, &vec_datatype);

        MPI_Type_commit(&vec_datatype);

        blockcount_element[0] = 2; //gid, lid
        blockcount_element[1] = 3; //position, velocity, acceleration

        oldtype_element[0] = MPI_INT;
        oldtype_element[1] = vec_datatype;

        MPI_Type_extent(MPI_INT, &intex);
        offset[0] = static_cast<MPI_Aint>(0);
        offset[1] = 2 * intex;

        MPI_Type_struct(2, blockcount_element, offset, oldtype_element, &element_datatype);
        MPI_Type_commit(&element_datatype);

        blockcount_range[0] = N;
        oldtype_range[0] = mpi_raw_datatype;
        MPI_Type_struct(1, blockcount_range, offset, oldtype_range, &range_datatype);
        MPI_Type_commit(&range_datatype);

        MPI_Type_contiguous(N, range_datatype, &domain_datatype);
        MPI_Type_commit(&domain_datatype);

        return {vec_datatype, element_datatype, range_datatype, domain_datatype};
    }
}

template<int N>
struct MESH_DATA {
    std::vector<elements::Element<N>> els;
};

#endif //NBMPI_GEOMETRIC_ELEMENT_HPP
