//
// Created by xetql on 02.07.18.
//

#ifndef NBMPI_STRATEGY_HPP
#define NBMPI_STRATEGY_HPP


#include <random>
#include <queue>
#include "../utils.hpp"

namespace decision_making {

    template<class Policy>
    class PolicyRunner {
        std::unique_ptr<Policy> p;
    public:
        template<class... Args> PolicyRunner(Args... args) : p(std::make_unique<Policy>(args...)) {}
        bool should_load_balance(int it) { return p->apply(it); };
    };

    class RandomPolicy {
        const Real lb_probability;
        std::mt19937 gen;
        mutable std::uniform_real_distribution<Real> dist = std::uniform_real_distribution<Real>(0.0, 1.0);
    public:
        RandomPolicy(Real lb_probability, int seed = 0) : lb_probability(lb_probability), gen(std::mt19937(seed)) {}
        bool apply(int it){ return dist(gen) < lb_probability; }
    };

    class IterationStatistics {
        std::array<double,  3> data = {0.0, 0.0, 0.0};
        std::vector<double> lb_times;
        int i = 0;
        int nproc;
    public:
        IterationStatistics(int nproc) : nproc(nproc) {}
        double  compute_avg_lb_time() { return std::accumulate(lb_times.cbegin(), lb_times.cend(), 0.0) / lb_times.size(); }
        double  get_cumulative_load_imbalance_slowdown() {return data[2]; }
        void    update_cumulative_load_imbalance_slowdown() { data[2] += data[1] - data[0]/nproc; }
        void    reset_load_imbalance_slowdown() { data[2] = 0.0; }
        double* max_it_time() { return &data[1]; }
        double* sum_it_time() { return &data[0]; }
        double* get_lb_time_ptr() {
            lb_times.push_back(std::numeric_limits<double>::lowest());
            return &lb_times[i++];
        }
    };

    class ThresholdPolicy {
        IterationStatistics* dataHolder;
        const std::function<Real (IterationStatistics*)> getDataF;
        const std::function<Real (IterationStatistics*)> getThresholdF;
    public:
        ThresholdPolicy(
                IterationStatistics* dataHolder,
                const std::function<Real (IterationStatistics*)> getDataF,
                const std::function<Real (IterationStatistics*)> getThresholdF) :
                dataHolder(dataHolder),
                getDataF(getDataF),
                getThresholdF(getThresholdF) {};
        bool apply(int it) { return dataHolder != nullptr ? getDataF(dataHolder) >= getThresholdF(dataHolder) : false; }
    };

    class InFilePolicy{
    public:
        mutable std::queue<bool> decisions;
        int period;
        InFilePolicy(std::string filename, int nframes, int npframe) {
            period = npframe;
            /* Read the targets of dataset files and apply decision at each frame */
            decisions = std::queue<bool>();
            std::ifstream dataset;
            dataset.open(filename, std::ofstream::in);
            if(!dataset.good()) throw std::runtime_error("bad repr. file");
            std::string line, buf;
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

        bool apply(int it) {
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
        bool apply(int it) { return (it % period) == 0; }
    };

    class NoLBPolicy{
    public:
        NoLBPolicy() {}
        bool apply(int it) { return false; }
    };
} // end of namespace decision_making

#endif //NBMPI_STRATEGY_HPP
