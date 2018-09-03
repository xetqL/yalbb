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
        virtual inline bool should_load_balance(int it, metric::LBMetrics<double>* mc) = 0;
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

        virtual inline bool should_load_balance(int it, metric::LBMetrics<double>* mc) override {
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
        virtual inline bool should_load_balance(int it, metric::LBMetrics<double>* mc) override {
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
            std::cout << decisions.size() << std::endl;
        }

        virtual inline bool should_load_balance(int it, metric::LBMetrics<double>* mc) override {
            if(it % period == 0) {
                auto decision = decisions.front();
                decisions.pop();
                return decision;
            } else return false;
        }
    };

    class PeriodicPolicy : public Policy{
        const int period;
    public:
        PeriodicPolicy(int period) : period(period) {}
        virtual inline bool should_load_balance(int it, metric::LBMetrics<double>* mc) override {
            return (it % period) == 0;
        }
    };

    class NoLBPolicy : public Policy{
    public:
        NoLBPolicy() {}
        virtual inline bool should_load_balance(int it, metric::LBMetrics<double>* mc) override {
            return false;
        }
    };

    class NeuralNetworkPolicy : public Policy {
        mlpack::ann::FFN<mlpack::ann::MeanSquaredError<>> model;
        mlpack::optimization::RMSProp optimizer;
        std::shared_ptr<arma::mat> ds;
    public:

        NeuralNetworkPolicy(mlpack::optimization::RMSProp& opt) :
                optimizer(opt) {
        }

        void train(const arma::mat& ds) {
            arma::mat inputs = ds.head_cols(ds.n_cols - 1);
            arma::mat targets = ds.tail_cols(1);
            model.Train(inputs, targets, optimizer);
        }

        /**
         * Ask model what to do *without fitting value*
         */
        virtual inline bool should_load_balance(int it, metric::LBMetrics<double>* mc) override {
            arma::mat features(mc->metrics);
            arma::mat responses(1, 1);
            model.Predict(features, responses);
            return responses(0,0) >= 0.5;
        }
    };

} // end of namespace decision_making

#endif //NBMPI_STRATEGY_HPP
