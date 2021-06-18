//
// Created by xetql on 11/20/20.
//
#pragma once

#include <fstream>
#include <utility>
#include <filesystem>
#include <vector>
#include <ostream>

#define show(x) #x << "=" << x

namespace io {

    struct ParallelOutput {
        int rank = -1;
        std::ostream& out;

        explicit ParallelOutput(std::ostream& _out) : out(_out) {
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        }
        const ParallelOutput& operator<<(std::ostream& (*F)(std::ostream&)) const {
            if(!rank) F(out);
            return *this;
        }
        template<class T> ParallelOutput& operator<<(const T& data) {
            if(!rank) std::cout << data;
            return *this;
        }

    };


}

namespace simulation {
    namespace {
        std::ostream null{nullptr};
    }
    enum ReportData {
        Imbalance,
        CumulativeImbalance,
        CumulativeVanillaImbalance,
        Time,
        CumulativeTime,
        Efficiency,
        LoadBalancingIteration,
        LoadBalancingCost,
        Interactions,
        NumOfNeighbors,
        LbPerf,
        SequentialTime,
    };

    template<unsigned N>
    constexpr inline auto frame_header(){
        if constexpr (N==3) return "x coord,y coord,z coord";
        if constexpr (N==2) return "x coord,y coord";
        if constexpr (N==1) return "x coord";
        if constexpr (N==0) return "";
    }

    // manage report files
class MonitoringSession {
        std::ofstream fparticle,
                        fimbalance,
                        fcumimbalance,
                        fvanimbalance,
                        ftime,
                        fcumtime,
                        fefficiency,
                        flbit,
                        flbcost,
                        finteractions,
                        fstdout,
                        fnumofneighbors,
                        flbperf,
                        fseqtime;

        const bool is_managing   = false,
                   is_logging_particles = false,
                   monitoring = true;
        std::string frame_files_folder {};
    public:
        MonitoringSession(bool is_managing, bool log_particles, const std::string& folder_prefix, const std::string& file_prefix, bool monitoring = true):
        is_managing(is_managing), is_logging_particles(log_particles), monitoring(monitoring) {
            if(is_managing && monitoring) {
                const std::string monitoring_files_folder = folder_prefix+"/monitoring";
                std::filesystem::create_directories(monitoring_files_folder);
                fstdout.open(monitoring_files_folder+"/"+"stdout.txt");
                fcumimbalance.open(monitoring_files_folder + "/" + file_prefix + "cum_imbalance.txt");
                fimbalance.open(monitoring_files_folder + "/" + file_prefix + "imbalance.txt");
                fvanimbalance.open(monitoring_files_folder + "/" + file_prefix + "van_cum_imbalance.txt");
                fcumtime.open(monitoring_files_folder + "/" + file_prefix + "cum_time.txt");
                ftime.open(monitoring_files_folder + "/" + file_prefix + "time.txt");
                fefficiency.open(monitoring_files_folder + "/" + file_prefix + "efficiency.txt");
                flbit.open(monitoring_files_folder + "/" + file_prefix + "lb_it.txt");
                flbcost.open(monitoring_files_folder + "/" + file_prefix + "lb_cost.txt");
                finteractions.open(monitoring_files_folder+"/"+ file_prefix +"interactions.txt");
                fnumofneighbors.open(monitoring_files_folder+"/"+ file_prefix +"numofneighbors.txt");
                flbperf.open(monitoring_files_folder+"/"+ file_prefix +"flbperf.txt");
                fseqtime.open(monitoring_files_folder+"/"+ file_prefix +"fseqtime.txt");

                if(log_particles) {
                    frame_files_folder = folder_prefix+"/frames";
                    std::filesystem::create_directories(frame_files_folder);
                }
            }
        }

        /* close report files */
        ~MonitoringSession() {
            if(is_managing && monitoring) {
                fstdout.close();
                fcumimbalance.close();
                fimbalance.close();
                fvanimbalance.close();
                fcumtime.close();
                ftime.close();
                fefficiency.close();
                flbit.close();
                flbcost.close();
                finteractions.close();
                fnumofneighbors.close();
                flbperf.close();
                fseqtime.close();
            }
        }

        auto& get_stream(ReportData type) {
            switch(type) {
                case Imbalance:
                    return fimbalance;
                case CumulativeImbalance:
                    return fcumimbalance;
                case CumulativeVanillaImbalance:
                    return fvanimbalance;
                case Time:
                    return ftime;
                case CumulativeTime:
                    return fcumtime;
                case Efficiency:
                    return fefficiency;
                case LoadBalancingIteration:
                    return flbit;
                case LoadBalancingCost:
                    return flbcost;
                case Interactions:
                    return finteractions;
                case NumOfNeighbors:
                    return fnumofneighbors;
                case LbPerf:
                    return flbperf;
                case SequentialTime:
                    return fseqtime;
                }
        }

        template<class T=void>
        void report(ReportData type, const T& report_value, const std::string sep = "\n") {
            using namespace std;
            if(is_managing && monitoring) {
                auto& target_stream = get_stream(type);
                target_stream << report_value << sep;
                target_stream.flush();
            }

        }

        template<unsigned N, class E, class GetDataFunc>
        void report_particle(std::vector<E>& recv_buf, const std::vector<int>& ranks, GetDataFunc f, unsigned id){
            if(is_managing && is_logging_particles && monitoring) {
                std::stringstream str;
                for(int i = 0; i < recv_buf.size(); i++){
                    str << *f(&recv_buf.at(i)) <<","<< (ranks.at(i)) <<std::endl;
                }
                fparticle.open(frame_files_folder+"/particle.csv."+ std::to_string(id));
                fparticle << frame_header<N>() << ",rank" << std::endl;
                fparticle << str.str();
                fparticle.close();
            }
        }

        template<class T>
        friend std::ostream& operator<<(MonitoringSession &session, const T os) {
            if(session.is_managing) {
                session.fstdout << os;
                return session.fstdout;
            } else {
                return null;
            }
        }
    };
}
