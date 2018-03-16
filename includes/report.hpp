//
// Created by xetql on 05.03.18.
//

#ifndef NBMPI_REPORT_HPP
#define NBMPI_REPORT_HPP

#include <vector>
#include <fstream>
#include <iterator>
inline void write_report_header(std::ofstream &stream, const sim_param_t* params, const int caller_rank, const int worker_id=0, const char* delimiter=";"){
    if(caller_rank == worker_id){
        stream << params->world_size << delimiter
               << params->npart << delimiter
               << (params->nframes * params->npframe) << delimiter
               << params->lb_interval << delimiter
               << params->simsize << delimiter
               << params->G << delimiter
               << params->seed;

        stream << std::endl;
    }
}

template<typename RealType>
inline void write_report_data(std::ofstream &stream, const int ts_idx, const std::vector<RealType> &timings, const int caller_rank, const int worker_id=0, const char* delimiter=";"){
    if(caller_rank == worker_id) {
        stream << std::to_string(ts_idx) << delimiter;
        std::copy(timings.begin(), timings.end(), std::ostream_iterator<RealType>(stream, delimiter));
        stream << std::endl;
    }
}

#endif //NBMPI_REPORT_HPP
