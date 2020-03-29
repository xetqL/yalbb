//
// Created by xetql on 13.12.17.
//

#ifndef NBMPI_UTILS_HPP
#define NBMPI_UTILS_HPP

#include <ctime>
#include <vector>
#include <stdexcept>
#include <iomanip>
#include <functional>
#include <cassert>
#include <cmath>
#include <limits>
#include <iostream>
#include <fstream>

#define binary_node_max_id_for_level(x) (std::pow(2, (int) (std::log(x+1)/std::log(2))+1) - 2)

using Real = float;
using Integer = long long int;

template<class T>
inline void update_local_ids(std::vector<T>& els, std::function<void (T&, Integer)> setLidF) {
    Integer i = 0; for(auto& el : els) setLidF(els->at(i), i++);
}

template<int N, class T>
inline void gather_elements_on(const int world_size,
                               const int my_rank,
                               const int nb_elements,
                               const std::vector<T> &local_el,
                               const int dest_rank,
                               std::vector<T> &dest_el,
                               const MPI_Datatype &sendtype,
                               const MPI_Comm &comm) {
    int nlocal = local_el.size();
    std::vector<int> counts(world_size, 0), displs(world_size, 0);
    MPI_Gather(&nlocal, 1, MPI_INT, &counts.front(), 1, MPI_INT, dest_rank, comm);
    for (int cpt = 0; cpt < world_size; ++cpt) displs[cpt] = cpt == 0 ? 0 : displs[cpt - 1] + counts[cpt - 1];
    if (my_rank == dest_rank) dest_el.resize(nb_elements);
    MPI_Gatherv(&local_el.front(), nlocal, sendtype,
                &dest_el.front(), &counts.front(), &displs.front(), sendtype, dest_rank, comm);
}

#define TIME_IT(a, name){\
 double start = MPI_Wtime();\
 a;\
 double end = MPI_Wtime();\
 auto diff = (end - start) / 1e-3;\
 std::cout << name << " took " << diff << " milliseconds" << std::endl;\
};\

#define START_TIMER(var)\
double var = MPI_Wtime();

#define RESTART_TIMER(v) \
v = MPI_Wtime() - v;

#define END_TIMER(var)\
var = MPI_Wtime() - var;

#define PAR_START_TIMER(var, comm)\
MPI_Barrier(comm);\
double var = MPI_Wtime();\

#define PAR_END_TIMER(var, comm)\
MPI_Barrier(comm);\
var = MPI_Wtime() - var;

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

template<class IntegerType, typename = std::enable_if<std::numeric_limits<IntegerType>::is_integer>>
inline IntegerType bitselect(IntegerType condition, IntegerType truereturnvalue, IntegerType falsereturnvalue) {
    return (truereturnvalue & -condition) | (falsereturnvalue & ~(-condition)); //a when TRUE
}

template<typename T>
inline T dto(double v) {
    T ret = (T) v;

    if(std::isinf(ret)){
        if(ret == -INFINITY){
            ret = std::numeric_limits<T>::lowest();
        } else {
            ret = std::numeric_limits<T>::max();
        }
    }

    return ret;
}

template<int N>
using BoundingBox = std::array<Real, 2*N>;

template<int D, int N> constexpr Real get_size(const BoundingBox<N>& bbox) {    return bbox.at(2*D+1) - bbox.at(2*D); }
template<int D, int N> constexpr Real get_min_dim(const BoundingBox<N>& bbox) { return bbox.at(2*D); }
template<int D, int N> constexpr Real get_max_dim(const BoundingBox<N>& bbox) { return bbox.at(2*D+1); }

// C++ template to print vector container elements
template <typename T, size_t N>
std::ostream& operator<<(std::ostream& os, const std::array<T, N>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i) {
        os << v[i];
        if (i != v.size() - 1)
            os << ", ";
    }
    os << "]\n";
    return os;
}


template<int N>
void update_bbox_for_container(BoundingBox<N>& new_bbox) {}
template <int N, class First, class... Rest>
void update_bbox_for_container(BoundingBox<N>& new_bbox, First& first, Rest&... rest) {
    for (const auto &el : first) {
        new_bbox[0] = std::min(new_bbox[0], el.position[0]);
        new_bbox[1] = std::max(new_bbox[1], el.position[0]);
        new_bbox[2] = std::min(new_bbox[2], el.position[1]);
        new_bbox[3] = std::max(new_bbox[3], el.position[1]);
        if constexpr (N==3) {
            new_bbox[4] = std::min(new_bbox[4], el.position[2]);
            new_bbox[5] = std::max(new_bbox[5], el.position[2]);
        }
    }
    update_bbox_for_container<N>(new_bbox, rest...);
}

template<int N, class... T>
BoundingBox<N> get_bounding_box(Real rc, T&... elementContainers){
    BoundingBox<N> new_bbox;
    if constexpr (N==3)
        new_bbox = {std::numeric_limits<Real>::max(), std::numeric_limits<Real>::lowest(),
                    std::numeric_limits<Real>::max(), std::numeric_limits<Real>::lowest(),
                    std::numeric_limits<Real>::max(), std::numeric_limits<Real>::lowest()};
    else
        new_bbox = {std::numeric_limits<Real>::max(), std::numeric_limits<Real>::lowest(),
                    std::numeric_limits<Real>::max(), std::numeric_limits<Real>::lowest()};
    update_bbox_for_container<N>(new_bbox, elementContainers...);
    /* hook to grid, resulting bbox is divisible by lc[i] forall i */
    for(int i = 0; i < N; ++i) {
        new_bbox.at(2*i)   = std::max((Real) 0.0, std::floor(new_bbox.at(2*i) / rc) * rc - (Real) 2.0*rc);
        new_bbox.at(2*i+1) = std::ceil(new_bbox.at(2*i+1) / rc) * rc + (Real) 2.0 * rc;
    }
    return new_bbox;
}

template<int N, class... T>
void update_bounding_box(BoundingBox<N>& bbox, Real rc, T&... elementContainers){
    update_bbox_for_container<N>(bbox, elementContainers...);
    /* hook to grid, resulting bbox is divisible by lc[i] forall i */
    for(int i = 0; i < N; ++i) {
        bbox.at(2*i)   = std::max((Real) 0.0, std::floor(bbox.at(2*i) / rc) * rc - 2*rc);
        bbox.at(2*i+1) = std::ceil(bbox.at(2*i+1) / rc) * rc + 2*rc;
    }
}

template<int N, class T>
void add_to_bounding_box(BoundingBox<N>& bbox, Real rc, T begin, T end){
    while(begin != end){
        bbox[0] = std::min(bbox[0], (*begin).position.at(0));
        bbox[1] = std::max(bbox[1], (*begin).position.at(0));
        bbox[2] = std::min(bbox[2], (*begin).position.at(1));
        bbox[3] = std::max(bbox[3], (*begin).position.at(1));
        if constexpr (N==3) {
            bbox[4] = std::min(bbox[4], (*begin).position.at(2));
            bbox[5] = std::max(bbox[5], (*begin).position.at(2));
        }
        begin++;
    }

    /* hook to grid, resulting bbox is divisible by lc[i] forall i */
    for(int i = 0; i < N; ++i) {
        bbox[2*i]   = std::max((Real) 0.0, std::floor(bbox[2*i] / rc) * rc - 2*rc);
        bbox[2*i+1] = std::ceil(bbox[2*i+1] / rc) * rc + 2*rc;
    }
}

template<int N>
inline std::array<Integer, N> get_cell_number_by_dimension(const BoundingBox<N>& bbox, Real rc) {
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

class CoordinateTranslater {
public:
    static std::tuple<Integer, Integer, Integer>
    inline translate_linear_index_into_xyz(const Integer index, const Integer ncols, const Integer nrows) {
        return {(index % ncols), std::floor(index % (ncols * nrows) / nrows), std::floor(index / (ncols * nrows))};    // depth
    };
    static std::tuple<Real, Real, Real>
    inline translate_xyz_into_position(const std::tuple<Integer, Integer, Integer>&& xyz, const Real rc) {
        return {std::get<0>(xyz) * rc, std::get<1>(xyz) * rc, std::get<2>(xyz) * rc};
    };

    template<int N>
    static inline Integer translate_xyz_into_linear_index(const std::tuple<Integer, Integer, Integer>&& xyz, const BoundingBox<N>& bbox, const Real rc) {
        auto lc = get_cell_number_by_dimension<N>(bbox, rc);
        return std::get<0>(xyz) + std::get<1>(xyz) * lc[0] + lc[0]*lc[1]*std::get<2>(xyz);
    };

    template<int N> static Integer
    translate_position_into_local_index(const std::array<Real, N>& position){}

    template<int N> static std::array<Real, N>
    translate_local_index_into_position(const Integer local_index, const BoundingBox<N>& bbox, const Real rc){
        auto lc = get_cell_number_by_dimension<N>(bbox, rc);
        auto[local_pos_x,local_pos_y,local_pos_z] = translate_xyz_into_position(translate_linear_index_into_xyz(local_index, lc[0], lc[1]), rc);
        std::array<Real, N> position;
        position[0] = local_pos_x+bbox[0];
        position[1] = local_pos_x+bbox[2];
        if constexpr(N==3)
            position[2] = local_pos_x+bbox[4];
        return position;
    }

    template<int N> static Integer
    translate_position_into_global_index(const std::array<Real, N>& position){}
    template<int N> static std::array<Real, N>
    translate_global_index_into_position(Integer global_index){}
};

template<int N>
Integer position_to_local_cell_index(std::array<Real, N> const &position, Real rc, const BoundingBox<N>& bbox, const Integer c, const Integer r){
    Integer lidx = (Integer) std::floor((position.at(0) - bbox[0]) / rc),
            lidy = (Integer) std::floor((position.at(1) - bbox[2]) / rc),
            lidz = 0;
    if constexpr(N==3)
        lidz = (Integer) std::floor((position.at(2) - bbox[4]) / rc);
    return lidx + c*lidy + c*r*lidz;
}



template<int N>
inline Integer
position_to_cell(std::array<Real, N> const &position, const Real lsub, const Integer c, const Integer r = 0) {
    Integer idx = (Integer) std::floor(position.at(0) / lsub);
    idx += c *    (Integer) std::floor(position.at(1) / lsub);
    if constexpr(N==3)
        idx += c * r * (Integer) std::floor(position.at(2) / lsub);
    return idx;
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

inline void
linear_to_grid(const long long index, const long long c, const long long r, int &x_idx, int &y_idx, int &z_idx) {
    x_idx = (int) (index % (c * r) % c);           // col
    y_idx = (int) std::floor(index % (c * r) / c); // row
    z_idx = (int) std::floor(index / (c * r));     // depth
};

namespace functional {
    template<typename R>
    inline R slice(R const &v, size_t slice_start, size_t slice_size) {
        size_t slice_max_size = v.size();
        slice_size = slice_size > slice_max_size ? slice_max_size : slice_size + 1;
        R s(v.begin() + slice_start, v.begin() + slice_size);
        return s;
    }

    template<typename To,
            template<typename...> class R=std::vector,
            typename StlFrom,
            typename F>
    R<To> map(StlFrom const &all, F const &map_func) {
        using std::begin;
        using std::end;
        R<To> accum;
        for (typename StlFrom::const_iterator it = begin(all); it != end(all); ++it) {
            std::back_insert_iterator<R<To> > back_it(accum);
            back_it = map_func(*it);
        }
        return accum;
    }

    template<typename To,
            template<typename...> class R=std::vector,
            typename StlFrom,
            typename ScanRightFunc>
    R<To> scan_left(StlFrom const &all, ScanRightFunc const &scan_func, To init_value) {
        using std::begin;
        using std::end;
        R<To> accum;
        std::back_insert_iterator<R<To> > back_it(accum);
        back_it = init_value;
        for (typename StlFrom::const_iterator it = begin(all); it != end(all); ++it) {
            std::back_insert_iterator<R<To> > back_it(accum);
            back_it = scan_func(*(end(accum) - 1), *it);
        }
        return accum;
    }

    template<typename StlFrom,
            typename To = typename StlFrom::value_type,
            typename ReduceFunc>
    To reduce(StlFrom const &all, ReduceFunc const &reduce_func, To const &init_value) {
        using std::begin;
        using std::end;
        To accum = init_value;
        for (typename StlFrom::const_iterator it = begin(all); it != end(all); ++it) {
            accum = reduce_func(accum, *it);
        }
        return accum;
    }

    template<typename A, typename B>
    const std::vector<std::pair<A, B>> zip(const std::vector<A> &a, const std::vector<B> &b) {
        std::vector<std::pair<A, B>> zipAB;
        int sizeAB = a.size();
        for (int i = 0; i < sizeAB; ++i)
            zipAB.push_back(std::make_pair(a.at(i), b.at(i)));
        return zipAB;
    }

    template<typename A, typename B,
            template<typename...> class I1=std::vector,
            template<typename...> class R1=std::vector,
            template<typename...> class R2=std::vector>
    const std::pair<R1<A>, R2<B>> unzip(const I1<std::pair<A, B>> &ab) {
        R1<A> left;
        R2<B> right;
        int sizeAB = ab.size();
        for (int i = 0; i < sizeAB; ++i) {
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
    template<template<typename...> class R=std::vector,
            typename Top,
            typename Sub = typename Top::value_type>
    R<typename Sub::value_type> flatten(Top const &all) {
        using std::begin;
        using std::end;
        R<typename Sub::value_type> accum;
        for (auto &sub : all)
            std::copy(begin(sub), end(sub), std::inserter(accum, end(accum)));
        return accum;
    }

}
namespace statistic {
    template<class RealType>
    std::tuple<RealType, RealType, RealType> sph2cart(RealType azimuth, RealType elevation, RealType r) {
        RealType x = r * std::cos(elevation) * std::cos(azimuth);
        RealType y = r * std::cos(elevation) * std::sin(azimuth);
        RealType z = r * std::sin(elevation);
        return std::make_tuple(x, y, z);
    }

    template<int N, class RealType>
    class UniformSphericalDistribution {
        const RealType sphere_radius, spherex, spherey, spherez;
    public:
        UniformSphericalDistribution(RealType sphere_radius, RealType spherex, RealType spherey, RealType spherez) :
                sphere_radius(sphere_radius), spherex(spherex), spherey(spherey), spherez(spherez) {}

        std::array<RealType, N> operator()(std::mt19937 &gen) {
            /*
            r1 = (np.random.uniform(0, 1 , n)*(b**3-a**3)+a**3)**(1/3);
            phi1 = np.arccos(-1 + 2*np.random.uniform(0, 1, n));
            th1 = 2*pi*np.random.uniform(0, 1, n);
            x = r1*np.sin(phi1)*np.sin(th1) + X;
            y = r1*np.sin(phi1)*np.cos(th1) + Y;
            z = r1*np.cos(phi1) + Z;
            */
            RealType a = sphere_radius, b = 0.0;
            std::uniform_real_distribution<RealType> udist(0.0, 1.0);

            RealType r1 = std::pow((udist(gen) * (std::pow(b, 3) - std::pow(a, 3)) + std::pow(a, 3)), 1.0 / 3.0);
            RealType ph1 = std::acos(-1.0 + 2.0 * udist(gen));
            RealType th1 = 2.0 * M_PI * udist(gen);

            auto p = std::make_tuple<RealType, RealType, RealType>(
                    r1 * std::sin(ph1) * std::sin(th1),
                    r1 * std::sin(ph1) * std::cos(th1),
                    r1 * std::cos(ph1)
            );
            if (N > 2) return {(std::get<0>(p)) + spherex, std::get<1>(p) + spherey, std::get<2>(p) + spherez};
            else return {std::get<0>(p) + spherex, std::get<1>(p) + spherey};
        }
    };

    template<int N, class RealType>
    class NormalSphericalDistribution {
        const RealType sphere_size, spherex, spherey, spherez;
    public:
        NormalSphericalDistribution(RealType sphere_size, RealType spherex, RealType spherey, RealType spherez) :
                sphere_size(sphere_size), spherex(spherex), spherey(spherey), spherez(spherez) {}

        std::array<RealType, N> operator()(std::mt19937 &gen) {
            std::array<RealType, N> res;
            std::normal_distribution<RealType> ndistx(spherex, sphere_size / 2.0); // could do better
            std::normal_distribution<RealType> ndisty(spherey, sphere_size / 2.0); // could do better
            if (N == 3) {
                RealType x, y, z;
                do {
                    std::normal_distribution<RealType> ndistz(spherez, sphere_size / 2.0); // could do better
                    x = ndistx(gen);
                    y = ndisty(gen);
                    z = ndistz(gen);
                    res[0] = x;
                    res[1] = y;
                    res[2] = z;
                } while (
                        (spherex - x) * (spherex - x) + (spherey - y) * (spherey - y) + (spherez - z) * (spherez - z) <=
                        (sphere_size * sphere_size / 4.0));
            } else {
                RealType x, y;
                do {
                    x = ndistx(gen);
                    y = ndisty(gen);
                    res[0] = x;
                    res[1] = y;
                } while ((spherex - x) * (spherex - x) + (spherey - y) * (spherey - y) <=
                         (sphere_size * sphere_size / 4.0));
            }
            return res;
        }
    };


/**
 * From http://www.tangentex.com/RegLin.htm
 * @tparam ContainerA
 * @tparam ContainerB
 * @param x x data
 * @param y y data
 * @return (a,b) of ax+b
 */
    template<typename Realtype, typename ContainerA, typename ContainerB>
    std::pair<Realtype, Realtype> linear_regression(const ContainerA &x, const ContainerB &y) {
        int i;
        Realtype xsomme, ysomme, xysomme, xxsomme;

        Realtype ai, bi;

        xsomme = 0.0;
        ysomme = 0.0;
        xysomme = 0.0;
        xxsomme = 0.0;
        const int n = x.size();
        for (i = 0; i < n; i++) {
            xsomme = xsomme + x[i];
            ysomme = ysomme + y[i];
            xysomme = xysomme + x[i] * y[i];
            xxsomme = xxsomme + x[i] * x[i];
        }
        ai = (n * xysomme - xsomme * ysomme) / (n * xxsomme - xsomme * xsomme);
        bi = (ysomme - ai * xsomme) / n;

        return std::make_pair(ai, bi);
    }

} // end of namespace statistic

namespace partitioning {
    namespace utils {

        template<typename A, typename B>
        const std::vector<std::pair<A, B>> zip(const std::vector<A> &a, const std::vector<B> &b) {
            std::vector<std::pair<A, B>> zipAB;
            int sizeAB = a.size();
            for (int i = 0; i < sizeAB; ++i)
                zipAB.push_back(std::make_pair(a.at(i), b.at(i)));
            return zipAB;
        }

        template<typename A, typename B>
        const std::pair<std::vector<A>, std::vector<B>> unzip(const std::vector<std::pair<A, B>> &ab) {
            std::vector<A> left;
            std::vector<B> right;
            int sizeAB = ab.size();
            for (int i = 0; i < sizeAB; ++i) {
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
        template<template<typename...> class R=std::vector,
                typename Top,
                typename Sub = typename Top::value_type>
        R<typename Sub::value_type> flatten(Top const &all) {
            using std::begin;
            using std::end;
            R<typename Sub::value_type> accum;
            for (auto &sub : all)
                std::copy(begin(sub), end(sub), std::inserter(accum, end(accum)));
            return accum;
        }

    }
}

#endif //NBMPI_UTILS_HPP
