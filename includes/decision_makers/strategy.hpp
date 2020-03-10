//
// Created by xetql on 02.07.18.
//

#ifndef NBMPI_STRATEGY_HPP
#define NBMPI_STRATEGY_HPP


#include <random>
#include <queue>
#include "../utils.hpp"

namespace decision_making {
    struct LBMetrics{

        const std::vector<Real> metrics;

        LBMetrics(const std::vector<Real> metrics) : metrics(metrics) {};

        Real get_gini_times(){
            return metrics.at(0);
        }
        Real get_gini_complexities(){
            return metrics.at(1);
        }
        Real get_gini_communications(){
            return metrics.at(2);
        }
        Real get_last_time_per_iteration(){
            return metrics.at(3);
        }
        Real get_variance_avg_time_per_iteration(){
            return metrics.at(4);
        }
        Real get_macd_gini_times(){
            return metrics.at(5);
        }
        Real get_macd_gini_complexities(){
            return metrics.at(6);
        }
        Real get_macd_times(){
            return metrics.at(7);
        }
        Real get_macd_gini_communications(){
            return metrics.at(8);
        }
    };

    template<class Policy>
    class PolicyRunner {
        std::unique_ptr<Policy> p;
    public:
        template<class... Args>
        PolicyRunner(Args... args){
           p = std::make_unique<Policy>(args...);
        }

        template<class DataHolder>
        bool should_load_balance(int it, DataHolder* holder){
            return p->apply(it, holder);
        };

        template<class DataHolder = std::nullptr_t>
        bool should_load_balance(int it) {
            return should_load_balance<DataHolder>(it, nullptr);
        };
    };

    class RandomPolicy {
        const Real lb_probability;
        std::mt19937 gen = std::mt19937(0);
        std::uniform_real_distribution<Real> dist = std::uniform_real_distribution<Real>(0.0, 1.0);
    public:
        RandomPolicy(Real lb_probability, int seed = 0 /* seed MUST be the same on all MPI ranks */) :
            lb_probability(lb_probability) {
            gen.seed(seed);
        }
        template<class DataHolder>
        bool apply(int it, DataHolder *holder) {
            return dist(gen) < lb_probability;
        }

    };

    class ThresholdHeuristicPolicy{
        const float threshold;
    public:
        ThresholdHeuristicPolicy(float threshold) : threshold(threshold){};
        template<class DataHolder>
        bool apply(int it, DataHolder *holder) {
            using ok = std::enable_if<std::is_same<DataHolder, LBMetrics>::value >;
            if(holder)
                return holder->get_gini_times() > threshold;
            else
                return false;
        }
    };

    class InFilePolicy{
    public:
        std::queue<bool> decisions;
        int period;
        InFilePolicy(std::string filename, int nframes, int npframe) {
            period = npframe;
            /* Read the targets of dataset files and apply decision at each frame */
            decisions = std::queue<bool>();
            std::ifstream dataset;
            dataset.open(filename, std::ofstream::in);
            if(!dataset.good())
                throw std::runtime_error("bad repr. file");
            std::string line;
            std::string buf;
            int decision_cnt = 0;
            bool clear = false;
            while (std::getline(dataset, line)) {
                if(clear) decisions = std::queue<bool>();
                std::stringstream ss(line);       // Insert the string into a stream
                std::vector<float> tokens; // Create vector to hold our words
                while (ss >> buf) tokens.push_back(std::stof(buf.c_str()));

                if(tokens.size() == 1) {
                    clear = true;
                    continue;
                } else {
                    clear = false;
                }

                decisions.push( *(tokens.end() - 1) > 0 );

            }
            dataset.close();
        }

        template<class DataHolder>
        bool apply(int it, DataHolder *holder) {
            if(it % period == 0) {
                auto decision = decisions.front();
                decisions.pop();
                return decision;
            } else return false;
        }
    };

    class PeriodicPolicy{
        const int period;
    public:
        PeriodicPolicy(int period) : period(period) {}
        template<class DataHolder>
        bool apply(int it, DataHolder *holder) {
            return (it % period) == 0;
        }

    };

    class NoLBPolicy{
    public:
        NoLBPolicy() {}
        template<class DataHolder>
        bool apply(int it, DataHolder *holder) {
            return false;
        }
    };
} // end of namespace decision_making

#endif //NBMPI_STRATEGY_HPP
