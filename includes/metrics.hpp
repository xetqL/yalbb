//
// Created by xetql on 21.03.18.
//

#ifndef NBMPI_METRICS_H
#define NBMPI_METRICS_H

#include <utility>
#include <vector>
#include <cmath>
#include <limits>
#include <deque>
#include <numeric>

#include <gsl/gsl_histogram.h>
#include <gsl/gsl_statistics.h>


namespace metric {
namespace topology {

/**
* Compute the weighted load histogram of a particle PE.
* This histogram tells whether the particle distribution is heterogeneous.
* In particular, it gives an idea on the complexity a particle can expect in a given cell from the cell-linked-list
* algorithm.
* Moreover, the weighted load tells if the complexity impacts few or many particles. Of course, one prefer
* impacting few particles by an high complexity than the reverse case.
* If the histogram has a positive skewness the distribution is heterogeneous, because a lot of cell have a small
* complexity and only few cells have a high complexity.
* @tparam N Particle Dimension
* @tparam MapType Particle Linked List structure type
* @param number_of_cell_per_row Number of CLL per row
* @param npart Number of particle attribute to the PE
* @param plist The CLL structure
*/
template <class FloatingPointPrecision, int N, class MapType>
std::vector<FloatingPointPrecision> compute_cells_loads(int number_of_cell_per_row, int npart, MapType &plist){
    int number_of_cell_per_col = number_of_cell_per_row,
        total_cells = number_of_cell_per_col*number_of_cell_per_row;

    FloatingPointPrecision avg = (FloatingPointPrecision) npart / (number_of_cell_per_row * number_of_cell_per_col);
    int diff = total_cells - plist.size();
    int xcellidx;
    int ycellidx;
    int zcellidx;
    std::vector<FloatingPointPrecision> loads(diff, 0.0);
    for(auto const& cell : plist) {
        int cellidx  = cell.first;
        linear_to_grid(cellidx, number_of_cell_per_row, number_of_cell_per_row, xcellidx, ycellidx, zcellidx);
        int particle_complexity = 0;
        FloatingPointPrecision weight = plist[cellidx].get()->size(); // the weight of this cell

        // Explore neighboring cells
        constexpr int zstart= N == 3 ? -1 : 0; // if in 2D, there is only 1 depth
        constexpr int zstop = N == 3 ?  2 : 1; // so, iterate only through [0,1)
        for (int neighborx = -1; neighborx < 2; neighborx++) {
            for (int neighbory = -1; neighbory < 2; neighbory++) {
                for (int neighborz = zstart; neighborz < zstop; neighborz++) {
                    // Check boundary conditions
                    if (xcellidx + neighborx < 0 || xcellidx + neighborx >= number_of_cell_per_row) continue;
                    if (ycellidx + neighbory < 0 || ycellidx + neighbory >= number_of_cell_per_row) continue;
                    if (zcellidx + neighborz < 0 || zcellidx + neighborz >= number_of_cell_per_row) continue;
                    int ncellidx = (xcellidx + neighborx) +
                                   number_of_cell_per_row * (ycellidx + neighbory) +
                                   number_of_cell_per_row * number_of_cell_per_row * (zcellidx + neighborz);
                    if (plist.find(ncellidx) != plist.end())
                        particle_complexity += plist[ncellidx].get()->size();
                }
            }
        }
        FloatingPointPrecision cell_weighted_load = weight * particle_complexity; //between 0 and npart.
        loads.push_back(cell_weighted_load);
    }
    return loads;
};

}// end namespace topology

namespace load_balancing {

template<typename RealType>
RealType compute_gini_index(std::vector<RealType> const& revenues){
    const int pop_size = revenues.size();
    std::vector<int> world(pop_size);
    std::iota(world.begin(), world.end(), 1);
    double h = 1.0 / (RealType) pop_size;
    RealType total_workload_sec = std::accumulate(revenues.begin(), revenues.end(), 0.0); //summed time + communications
    std::vector<RealType> workload_ratios = functional::map<RealType>(revenues, [&total_workload_sec](auto v){return v / total_workload_sec;});
    std::sort(workload_ratios.begin(), workload_ratios.end());
    std::vector<RealType> cumulative_workload_ratios = functional::scan_left(workload_ratios, [](auto acc, auto v){ return acc + v;}, (RealType) 0.0);
    const unsigned int nb_ratios = cumulative_workload_ratios.size();
    double gini_area = 0.0;
    for(size_t i = 1; i < nb_ratios; ++i)
        gini_area += h * (cumulative_workload_ratios[i-1] + cumulative_workload_ratios[i]) / 2.0;
    return (0.5 - gini_area);
}

template<typename Container>
double load_imbalance_standard_metric(Container const& timings) {
    return 100.0 * (std::max(timings.begin(), timings.end()) / gsl_stats_mean(&timings.front(), timings.size(), 1) - 1.0);
}

}// end namespace load_balancing

namespace load_dynamic {

template<typename Container>
inline double compute_ema(int size, double alpha, const Container& Y){
    const int position = Y.size() - size;
    const int starting_el = position >= 0 ? position : 0;
    double acc = Y[starting_el];
    for (int t = starting_el+1; t < Y.size(); ++t) acc = alpha*Y[t] + (1-alpha)*acc;
    return acc;
}

template<typename Container>
inline double compute_ma(double alpha, const Container& Y){
    return (std::accumulate(Y.begin(), Y.end(), 0.0)) / Y.size();
}

/**
* Moving Average Convergence Divergence, will tell us if the "local" trend follow the "long term" trend.
* + : Tell us that it increase compared to long term
* - : Tell us that it decrease ""
* @param Y the window
* @return MACD indicator
*/
template<typename Container>
inline double compute_macd(const Container& Y, const int sz_small_ema = 12, const int sz_big_ema = 26, const double alpha = 0.95) {
    return compute_ema(sz_small_ema, alpha, Y) - compute_ema(sz_big_ema, alpha, Y);
}

}// end namespace load_dynamic
}// end namespace metric

template<class T>
struct SlidingWindow {
    std::deque<T> data_container;

    int window_max_size;
    SlidingWindow(int window_max_size): window_max_size(window_max_size) {};
    inline void add(const T& data){

        if(data_container.size() < window_max_size)
            data_container.push_back(data); // add while not full
        else {                      // when full
            data_container.pop_front();     // delete oldest data
            data_container.push_back(data); // push new data
        }
    }
};

#endif //NBMPI_METRICS_H
