//
// Created by xetql on 02.07.18.
//

#ifndef NBMPI_STRATEGY_HPP
#define NBMPI_STRATEGY_HPP

#include <random>
#include <queue>
#include "../metrics.hpp"
namespace decision_making {
    class Policy {
    public:
        virtual bool should_load_balance(int it, metric::LBMetrics<double>* mc) = 0;
    };

    class RandomPolicy : public Policy {
        const float lb_probability;
        std::mt19937 gen = std::mt19937(0);
        std::uniform_real_distribution<float> dist = std::uniform_real_distribution<float>(0.0, 1.0);
    public:
        RandomPolicy(float lb_probability, int seed = 0 /* seed MUST be the same on all MPI ranks */) :
            lb_probability(lb_probability) {
            gen.seed(seed);
        }

        virtual bool should_load_balance(int it, metric::LBMetrics<double>* mc) override {
            return dist(gen) < lb_probability;
        }
    };

    class ThresholdHeuristicPolicy : public Policy{
        const float threshold;
        inline bool is_greater_than_threshold(metric::LBMetrics<double>* mc) {
            return mc->get_gini_times() > threshold;
        }

    public:
        ThresholdHeuristicPolicy(float threshold) : threshold(threshold){};
        virtual bool should_load_balance(int it, metric::LBMetrics<double>* mc) override {
            return is_greater_than_threshold(mc);
        }
    };
    class InFilePolicy : public Policy {

    public:
        std::queue<bool> decisions;
        int period;
        InFilePolicy(std::string filename, int nframes, int npframe) {
            period = npframe;
            /* Read the targets of dataset files and apply decision at each frame */
            decisions = std::queue<bool>();
            std::ifstream dataset;
            dataset.open(filename, std::ofstream::in);
            std::string line;
            std::string buf;
            int decision_cnt = 0;
            while (std::getline(dataset, line) && decision_cnt < nframes) {
                std::stringstream ss(line);       // Insert the string into a stream
                std::vector<float> tokens; // Create vector to hold our words
                while (ss >> buf) tokens.push_back(std::stof(buf.c_str()));
                if(std::any_of(tokens.begin(), tokens.end(), [](auto token){ return token > 0.0;})){
                    decisions.push( *(tokens.end() - 1) > 0 );

                }
                decision_cnt++;
            }
            dataset.close();
        }

        virtual bool should_load_balance(int it, metric::LBMetrics<double>* mc) override {
            if(it % period == 0 && it > 0) {
                std::cout << it << " " << decisions.size() << std::endl;
                auto decision = decisions.front();
                decisions.pop();
                return decision;
            } else return false;
        }
    };

    class PeriodicPolicy : public Policy{
        const int period;
    public:
        PeriodicPolicy(int period) : period(period){}
        virtual bool should_load_balance(int it, metric::LBMetrics<double>* mc) override {
            return (it % period) == 0;
        }
    };

    class NoLBPolicy : public Policy{
    public:
        NoLBPolicy() {}
        virtual bool should_load_balance(int it, metric::LBMetrics<double>* mc) override {
            return false;
        }
    };

} // end of namespace decision_making

#endif //NBMPI_STRATEGY_HPP
