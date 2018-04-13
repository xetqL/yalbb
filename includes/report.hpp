//
// Created by xetql on 05.03.18.
//

#ifndef NBMPI_REPORT_HPP
#define NBMPI_REPORT_HPP

#include <vector>
#include <fstream>
#include <iterator>
#include "params.hpp"

inline void write_report_header(std::ofstream &stream, const sim_param_t* params, const int caller_rank, const int worker_id=0, const char* delimiter=";"){
    if(caller_rank == worker_id){
        stream << params->world_size << delimiter
               << params->npart << delimiter
               << (params->nframes * params->npframe) << delimiter
               << params->lb_interval << delimiter
               << params->simsize << delimiter
               << params->G << delimiter
               << params->seed << 0;
        stream << std::endl;
    }
}

inline void write_report_header_bin(std::ofstream &stream, const sim_param_t* params, const int caller_rank, const int worker_id=0){
    if(caller_rank == worker_id){
        unsigned int timesteps = params->nframes * params->npframe;
        stream.write(reinterpret_cast<const char*>(&params->world_size),       sizeof(unsigned int));
        stream.write(reinterpret_cast<const char*>(&params->npart),            sizeof(int));
        stream.write(reinterpret_cast<const char*>(&timesteps),                sizeof(unsigned int));
        stream.write(reinterpret_cast<const char*>(&params->lb_interval),      sizeof(unsigned int));
        stream.write(reinterpret_cast<const char*>(&params->one_shot_lb_call), sizeof(int));
        stream.write(reinterpret_cast<const char*>(&params->simsize),          sizeof(float));
        stream.write(reinterpret_cast<const char*>(&params->G),                sizeof(float));
        stream.write(reinterpret_cast<const char*>(&params->seed),             sizeof(int));
        stream.write(params->uuid.c_str(),                                     16);
    }
}

/**
 * Write one line in the report (i.e., one time step). String format
 * @tparam RealType
 * @param stream the stream to write in
 * @param ts_idx Id of the time step
 * @param timings Vector containings the time taken by each PE
 * @param caller_rank The rank of the PE that is calling this function
 * @param worker_id The rank of the PE dedicated to write the data
 * @param delimiter The char separating elements
 */
template<typename RealType>
inline void write_report_data(std::ofstream &stream, const int ts_idx, const std::vector<RealType> &data,
                              const int caller_rank, const int worker_id=0, const char* delimiter=";"){
    if(caller_rank == worker_id) {
        stream << std::to_string(ts_idx) << delimiter;
        std::move(data.begin(), data.end(), std::ostream_iterator<RealType>(stream, delimiter));
        stream << std::endl;
    }
}

/**
 * Write one line in the report (i.e., one time step). Binary mode.
 * @tparam RealType
 * @param stream the stream to write in
 * @param ts_idx Id of the time step
 * @param timings Vector containings the time taken by each PE
 * @param caller_rank The rank of the PE that is calling this function
 * @param worker_id The rank of the PE dedicated to write the data
 * @param delimiter The char separating elements
 */
template<typename RealType=float>
inline void write_report_data_bin(std::ofstream &stream, const int ts_idx, const std::vector<RealType> &data,
                                  const int caller_rank, const int worker_id=0){
    if(caller_rank == worker_id) {
        stream.write(reinterpret_cast<const char*>(&ts_idx), sizeof(int));
        for (const RealType& value: data) {
            stream.write(reinterpret_cast<const char*>(&value), sizeof(RealType));
        }
    }
}

template<typename RealType=float>
inline void write_metric_data_bin(std::ofstream &stream, const int ts_idx, const std::vector<RealType> &data,
                                  const int caller_rank, const int worker_id=0){
    if(caller_rank == worker_id) {
        int N = data.size();
        stream.write(reinterpret_cast<const char*>(&ts_idx), sizeof(int));
        stream.write(reinterpret_cast<const char*>(&N),      sizeof(int));
        for (const RealType& value: data) {
            stream.write(reinterpret_cast<const char*>(&value), sizeof(RealType));
        }
    }
}

template<typename RealType=float>
inline void write_report_total_time_bin(std::ofstream &stream, RealType simtime,
                                  const int caller_rank, const int worker_id=0){
    if(caller_rank == worker_id) {
        stream.write(reinterpret_cast<const char*>(&simtime), sizeof(RealType));
    }
}


#endif //NBMPI_REPORT_HPP
