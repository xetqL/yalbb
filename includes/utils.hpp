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

#define TIME_IT(a, name){\
 double start = MPI_Wtime();\
 a;\
 double end = MPI_Wtime();\
 auto diff = (end - start) / 1e-3;\
 std::cout << name << " took " << diff << " milliseconds" << std::endl;\
};\

inline std::string get_date_as_string() {
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%d-%m-%Y-%H-%M-%S");
    std::string date = oss.str();
    return date;
}

template<int N>
inline int position_to_cell(std::array<double, N> const& position, const double lsub, const int c, const int r = 0) {
    std::vector<int> weight = {1, c, c*r};
    int idx = 0;
    for(int i = 0; i < N; ++i)
        idx += weight[i] * std::floor(position.at(i) / lsub);
    return idx;
}

inline void linear_to_grid(const int index, const int c, const int r, int& x_idx, int& y_idx, int& z_idx){
     x_idx = (index % (c*r) % c);           // col
     y_idx = std::floor(index % (c*r) / r); // row
     z_idx = std::floor(index / (c*r));     // depth
    assert(c==r);
    assert(x_idx < c);
    assert(y_idx < r);
    assert(z_idx < r);

};

namespace functional {
template<typename R>
inline R slice(R const& v, size_t slice_start, size_t slice_size) {
    size_t slice_max_size = v.size();
    slice_size = slice_size > slice_max_size ? slice_max_size : slice_size+1;
    R s(v.begin()+slice_start, v.begin() + slice_size);
    return s;
}

template <typename To,
          template<typename...> class R=std::vector,
          typename StlFrom,
          typename F>
R<To> map(StlFrom const& all, F const& map_func) {
    using std::begin;
    using std::end;
    R<To> accum;
    for(typename StlFrom::const_iterator it = begin(all); it != end(all); ++it) {
        std::back_insert_iterator< R<To> > back_it (accum);
        back_it = map_func(*it);
    }
    return accum;
}

template <typename To,
        template<typename...> class R=std::vector,
        typename StlFrom,
        typename ScanRightFunc>
R<To> scan_left(StlFrom const& all, ScanRightFunc const& scan_func, To init_value) {
    using std::begin;
    using std::end;
    R<To> accum;
    std::back_insert_iterator< R<To> > back_it (accum);
    back_it = init_value;
    for(typename StlFrom::const_iterator it = begin(all); it != end(all); ++it) {
        std::back_insert_iterator< R<To> > back_it (accum);
        back_it = scan_func(*(end(accum)-1), *it);
    }
    return accum;
}

template <typename StlFrom,
        typename To = typename StlFrom::value_type,
        typename ReduceFunc>
To reduce(StlFrom const& all, ReduceFunc const& reduce_func, To const& init_value) {
    using std::begin;
    using std::end;
    To accum = init_value;
    for(typename StlFrom::const_iterator it = begin(all); it != end(all); ++it){
        accum = reduce_func(accum, *it);
    }
    return accum;
}

template<typename A, typename B>
const std::vector<std::pair<A, B>> zip(const std::vector<A>& a, const std::vector<B>& b) {
    std::vector<std::pair<A,B>> zipAB;
    int sizeAB = a.size();
    for(int i = 0; i < sizeAB; ++i)
        zipAB.push_back(std::make_pair(a.at(i), b.at(i)));
    return zipAB;
}

template<typename A, typename B,
         template<typename...> class I1=std::vector,
         template<typename...> class R1=std::vector,
         template<typename...> class R2=std::vector>
const std::pair<R1<A>, R2<B>> unzip(const I1<std::pair<A, B>>& ab) {
    R1<A> left;
    R2<B> right;
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
}
namespace statistic {
/**
 * From http://www.tangentex.com/RegLin.htm
 * @tparam ContainerA
 * @tparam ContainerB
 * @param x x data
 * @param y y data
 * @return (a,b) of ax+b
 */
template<typename Realtype, typename ContainerA, typename ContainerB>
std::pair<Realtype, Realtype> linear_regression(const ContainerA& x, const ContainerB& y) {
    int i; Realtype xsomme, ysomme, xysomme, xxsomme;

    Realtype ai, bi;

    xsomme = 0.0; ysomme = 0.0;
    xysomme = 0.0; xxsomme = 0.0;
    const int n = x.size();
    for (i=0;i<n;i++) {
        xsomme = xsomme + x[i]; ysomme = ysomme + y[i];
        xysomme = xysomme + x[i]*y[i];
        xxsomme = xxsomme + x[i]*x[i];
    }
    ai = (n*xysomme - xsomme*ysomme)/(n*xxsomme - xsomme*xsomme);
    bi = (ysomme - ai*xsomme)/n;

    return std::make_pair(ai, bi);
}

} // end of namespace statistic

namespace partitioning {
namespace utils {

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

}
}

#endif //NBMPI_UTILS_HPP
