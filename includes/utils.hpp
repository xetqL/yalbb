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
#include <random>
#include <cstring>

inline std::string get_date_as_string();
bool file_exists(const std::string fileName);
std::vector<std::string> split(const std::string &s, char delimiter);

template<class T>
struct MESH_DATA {
    std::vector<T> els;
};

using Real       = float;
using Time       = double;
using Rank       = int;
using Integer    = long long int;
using Complexity = Integer;
using Index      = Integer;

template<class GetPosPtrFunc, class GetVelPtrFunc, class GetForceFunc, class BoxIntersectionFunc, class PointAssignationFunc, class LoadBalancingFunc>
class FunctionWrapper {
    GetPosPtrFunc posPtrFunc;
    GetVelPtrFunc velPtrFunc;
    GetForceFunc forceFunc;
    BoxIntersectionFunc boxIntersectionFunc;
    PointAssignationFunc pointAssignationFunc;
    LoadBalancingFunc loadBalancingFunc;
public:
    FunctionWrapper(GetPosPtrFunc posPtrFunc, GetVelPtrFunc velPtrFunc, GetForceFunc forceFunc,
                                  BoxIntersectionFunc boxIntersectionFunc, PointAssignationFunc pointAssignationFunc,
                                  LoadBalancingFunc loadBalancingFunc) : posPtrFunc(posPtrFunc), velPtrFunc(velPtrFunc),
                                                                         forceFunc(forceFunc),
                                                                         boxIntersectionFunc(boxIntersectionFunc),
                                                                         pointAssignationFunc(pointAssignationFunc),
                                                                         loadBalancingFunc(loadBalancingFunc) {}

    const GetPosPtrFunc &getPosPtrFunc() const {
        return posPtrFunc;
    }

    void setPosPtrFunc(const GetPosPtrFunc &posPtrFunc) {
        FunctionWrapper::posPtrFunc = posPtrFunc;
    }

    const GetVelPtrFunc &getVelPtrFunc() const {
        return velPtrFunc;
    }

    void setVelPtrFunc(const GetVelPtrFunc &velPtrFunc) {
        FunctionWrapper::velPtrFunc = velPtrFunc;
    }

    GetForceFunc getForceFunc() const {
        return forceFunc;
    }

    void setGetForceFunc(GetForceFunc forceFunc) {
        FunctionWrapper::forceFunc = forceFunc;
    }

    BoxIntersectionFunc getBoxIntersectionFunc() const {
        return boxIntersectionFunc;
    }

    void setBoxIntersectionFunc(BoxIntersectionFunc boxIntersectionFunc) {
        FunctionWrapper::boxIntersectionFunc = boxIntersectionFunc;
    }

    PointAssignationFunc getPointAssignationFunc() const {
        return pointAssignationFunc;
    }

    void setPointAssignationFunc(PointAssignationFunc pointAssignationFunc) {
        FunctionWrapper::pointAssignationFunc = pointAssignationFunc;
    }

    LoadBalancingFunc getLoadBalancingFunc() const {
        return loadBalancingFunc;
    }

    void setLoadBalancingFunc(LoadBalancingFunc loadBalancingFunc) {
        FunctionWrapper::loadBalancingFunc = loadBalancingFunc;
    }
};

template<int N>
std::array<double, N> get_as_double_array(const std::array<Real, N>& real_array){
    if constexpr(N==2)
        return {(double) real_array[0], (double) real_array[1]};
    else
        return {(double) real_array[0], (double) real_array[1], (double) real_array[2]};
}

template<int N>
inline void put_in_double_array(std::array<double, N>& double_array, const std::array<Real, N>& real_array){
    double_array[0] = real_array[0];
    double_array[1] = real_array[1];
    if constexpr (N==3)
        double_array[2] = real_array[2];
}

template<int N, class F>
inline void map(std::array<double, N>& double_array, F f){
    double_array[0] = f(double_array[0]);
    double_array[1] = f(double_array[1]);
    if constexpr (N==3)
        double_array[2] = f(double_array[2]);
}

template<class T>
inline void update_local_ids(std::vector<T>& els, std::function<void (T&, Integer)> setLidF) {
    Integer i = 0; for(auto& el : els) setLidF(els->at(i), i++);
}

template<class IntegerType, typename = std::enable_if<std::numeric_limits<IntegerType>::is_integer>>
inline IntegerType bitselect(IntegerType condition, IntegerType truereturnvalue, IntegerType falsereturnvalue) {
    return (truereturnvalue & -condition) | (falsereturnvalue & ~(-condition)); //a when TRUE
}

// C++ template to print vector container elements
template <typename T, size_t N> std::ostream& operator<<(std::ostream& os, const std::array<T, N>& v)
{
    for (int i = 0; i < N; ++i) {
        os << v[i];
        if (i != v.size() - 1)
            os << " ";
    }
    return os;
}
template <typename T> std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    const auto s = v.size();
    for (int i = 0; i < s; ++i) {
        os << v[i];
        if (i != s - 1) os << "\n";
    }
    return os;
}
template<typename T> T dto(double v) {
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

template<int N> using BoundingBox = std::array<Real, 2*N>;
template<int D, int N> constexpr Real get_size(const BoundingBox<N>& bbox) { return bbox.at(2*D+1) - bbox.at(2*D); }
template<int D, int N> constexpr Real get_min_dim(const BoundingBox<N>& bbox) { return bbox.at(2*D); }
template<int D, int N> constexpr Real get_max_dim(const BoundingBox<N>& bbox) { return bbox.at(2*D+1); }

template<int N>
bool is_within(const BoundingBox<N>& bbox, std::array<Real, N>& xyz){
    bool within = true;
    for(int i = 0; i < N; i++){
        within = within && (bbox[2*i] <= xyz[i]) && (xyz[i] <= bbox[2*i+1]);
    }
    return within;
}

template<int N, class GetPosFunc>
void update_bbox_for_container(BoundingBox<N>& new_bbox, GetPosFunc getPosFunc) {}

template <int N, class GetPosFunc, class First, class... Rest>
void update_bbox_for_container(BoundingBox<N>& new_bbox, GetPosFunc getPosFunc, First& first, Rest&... rest) {
    for(int i = 0; i < N; ++i) {
        for (auto &el : first) {
            const auto& pos    = *getPosFunc(&el);
            new_bbox.at(2*i)   = std::min(new_bbox.at(2*i),   pos.at(i));
            new_bbox.at(2*i+1) = std::max(new_bbox.at(2*i+1), pos.at(i));
        }
    }
    update_bbox_for_container<N>(new_bbox, getPosFunc, rest...);


}

template<class T> void apply_resize_strategy(std::vector<T>* vec, size_t minimum_size){
    size_t actual_size = vec->size();
    if(actual_size < minimum_size) {
        vec->resize(minimum_size);
    } else if(actual_size >= 2.0 * minimum_size) {
        vec->resize(actual_size / 2.0 + minimum_size / 2.0);
    }
}

template<int N, class GetPosFunc, class... T>
BoundingBox<N> get_bounding_box(Real rc, GetPosFunc getPosFunc, T&... elementContainers){
    BoundingBox<N> new_bbox;

    if constexpr (N==3) {
        new_bbox = {std::numeric_limits<Real>::max(), std::numeric_limits<Real>::lowest(),
                    std::numeric_limits<Real>::max(), std::numeric_limits<Real>::lowest(),
                    std::numeric_limits<Real>::max(), std::numeric_limits<Real>::lowest()};
    }else {
        new_bbox = {std::numeric_limits<Real>::max(), std::numeric_limits<Real>::lowest(),
                    std::numeric_limits<Real>::max(), std::numeric_limits<Real>::lowest()};
    }

    update_bbox_for_container<N>(new_bbox, getPosFunc, elementContainers...);
    /* hook to grid, resulting bbox is divisible by lc[i] forall i */
    Real radius = 2.0*rc;
    for(int i = 0; i < N; ++i) {
        new_bbox.at(2*i)   = std::max((Real)0.0, std::floor(new_bbox.at(2*i) / rc)  * rc - radius);
        new_bbox.at(2*i+1) = std::ceil(new_bbox.at(2*i+1) / rc) * rc + radius;
    }

    return new_bbox;
}

template<int N, class GetPosFunc, class... T>
void update_bounding_box(BoundingBox<N>& bbox, Real rc, GetPosFunc getPosFunc, T&... elementContainers){
    update_bbox_for_container<N>(bbox, getPosFunc, elementContainers...);
    /* hook to grid, resulting bbox is divisible by lc[i] forall i */
    Real radius = 4*rc;
    for(int i = 0; i < N; ++i) {
        bbox.at(2*i)   = std::max((Real)0.0, std::floor(bbox.at(2*i) / rc)  * rc - radius);
        bbox.at(2*i+1) = std::ceil(bbox.at(2*i+1) / rc) + rc + radius;
    }
}
template<int N, class T>
void add_to_bounding_box(BoundingBox<N>& bbox, Real rc, T begin, T end){
    while(begin != end){
        bbox[0] = std::min(bbox.at(0), (*begin).position.at(0));
        bbox[1] = std::max(bbox.at(1), (*begin).position.at(0));
        bbox[2] = std::min(bbox.at(2), (*begin).position.at(1));
        bbox[3] = std::max(bbox.at(3), (*begin).position.at(1));
        if constexpr (N==3) {
            bbox[4] = std::min(bbox.at(4), (*begin).position.at(2));
            bbox[5] = std::max(bbox.at(5), (*begin).position.at(2));
        }
        begin++;
    }

    /* hook to grid, resulting bbox is divisible by lc[i] forall i */
    for(int i = 0; i < N; ++i) {
        bbox.at(2*i)   = std::max((Real)0.0, std::floor(bbox.at(2*i) / rc)  * rc - rc);
        bbox.at(2*i+1) =  std::ceil(bbox.at(2*i+1) / rc)* rc + rc;
    }
}

template<int N>
inline std::array<Integer, N> get_cell_number_by_dimension(const BoundingBox<N>& bbox, Real rc) {
    std::array<Integer, N> lc;
    lc [0] = std::round(get_size<0, N>(bbox) / rc);
    lc [1] = std::round(get_size<1, N>(bbox) / rc);
    if constexpr(N==3)
        lc [2] = std::round(get_size<2, N>(bbox) / rc);
    return lc;
}

template<int N>
Integer get_total_cell_number(const BoundingBox<N>& bbox, Real rc){
    auto lc = get_cell_number_by_dimension<N>(bbox, rc);
    return std::accumulate(lc.begin(), lc.end(), 1, [](auto prev, auto v){return prev * v;});
}

template<int N>
inline Integer position_to_local_cell_index(std::array<Real, N> const &position, Real rc, const BoundingBox<N>& bbox, const Integer c, const Integer r){
    if constexpr(N==3) {
        return ((position.at(0) - bbox[0]) / rc) + c * ((Integer) ((position.at(1) - bbox[2]) / rc)) + c * r * ((Integer) std::floor((position.at(2) - bbox[4]) / rc));
    } else {
        return ((position.at(0) - bbox[0]) / rc) + c * ((Integer) ((position.at(1) - bbox[2]) / rc));
    }
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
    std::vector<std::pair<A, B>> zip(const std::vector<A> &a, const std::vector<B> &b) {
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
    std::pair<R1<A>, R2<B>> unzip(const I1<std::pair<A, B>> &ab) {
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


#endif //NBMPI_UTILS_HPP