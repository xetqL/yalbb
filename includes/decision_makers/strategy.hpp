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
        bool apply(int it) { return dist(gen) < lb_probability; }
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
                dataHolder(dataHolder), getDataF(getDataF), getThresholdF(getThresholdF) {};
        bool apply(int it) {
            std::cout << std::fixed << std::setprecision(6) << getDataF(dataHolder) << " >= " << getThresholdF(dataHolder) <<  std::endl;
            return dataHolder != nullptr ? getDataF(dataHolder) >= getThresholdF(dataHolder) : false;
        }
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
