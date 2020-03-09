//
// Created by xetql on 02.07.18.
//

#ifndef NBMPI_STRATEGY_HPP
#define NBMPI_STRATEGY_HPP


#include <random>
#include <queue>
#include "../utils.hpp"

namespace decision_making {
    class Policy {
    public:
        virtual inline bool should_load_balance(int it, Real* metrics, int n_metrics) = 0;

        virtual void print(std::string prefix) {
            std::cout << prefix << " ";
        }
    };

    class RandomPolicy : public Policy {
        const Real lb_probability;
        std::mt19937 gen = std::mt19937(0);
        std::uniform_real_distribution<Real> dist = std::uniform_real_distribution<Real>(0.0, 1.0);
    public:
        RandomPolicy(Real lb_probability, int seed = 0 /* seed MUST be the same on all MPI ranks */) :
            lb_probability(lb_probability) {
            gen.seed(seed);
        }

        virtual inline bool should_load_balance(int it, Real* metrics, int n_metrics) override {
            return dist(gen) < lb_probability;
        }

        void print(std::string prefix) override {
            Policy::print(prefix);
            std::cout << "Random Policy:" << std::endl;
        }

    };

    class ThresholdHeuristicPolicy : public Policy{
        const float threshold;
        bool is_greater_than_threshold(Real metric) {
            return metric > threshold;
        }
    public:
        ThresholdHeuristicPolicy(float threshold) : threshold(threshold){};
        virtual inline bool should_load_balance(int it, Real* metrics, int n_metrics) override {
            return is_greater_than_threshold(metrics[n_metrics-1]);
        }
        void print(std::string prefix) override {
            Policy::print(prefix);
            std::cout << "Threshold Policy:" << std::endl;
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
            if(!dataset.good()) throw std::runtime_error("bad repr. file");
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

        virtual inline bool should_load_balance(int it, Real* metrics, int n_metrics) override {
            if(it % period == 0) {
                auto decision = decisions.front();
                decisions.pop();
                return decision;
            } else return false;
        }

        void print(std::string prefix) override {
            Policy::print(prefix);
            std::cout << "InFile Policy:" << std::endl;
        }
    };

    class PeriodicPolicy : public Policy{
        const int period;
    public:
        PeriodicPolicy(int period) : period(period) {}
        virtual inline bool should_load_balance(int it, Real* metrics, int n_metrics) override {
            return (it % period) == 0;
        }
        void print(std::string prefix) override {
            Policy::print(prefix);
            std::cout << "Periodic Policy ("<< std::to_string(period) << "):" << std::endl;
        }
    };

    class NoLBPolicy : public Policy{
    public:
        NoLBPolicy() {}
        virtual inline bool should_load_balance(int it, Real* metrics, int n_metrics) override {
            return false;
        }
    };
} // end of namespace decision_making

#endif //NBMPI_STRATEGY_HPP
