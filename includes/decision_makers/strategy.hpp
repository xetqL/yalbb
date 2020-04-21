//
// Created by xetql on 02.07.18.
//

#ifndef NBMPI_STRATEGY_HPP
#define NBMPI_STRATEGY_HPP


#include <random>
#include <queue>
#include "../utils.hpp"

namespace decision_making {

    template<class P>
    class LBPolicy {
    public:
        virtual bool should_load_balance() = 0;
    };

    template<class Policy>
    class PolicyRunner : public LBPolicy<Policy> {
        std::unique_ptr<Policy> p;
    public:
        template<class... Args> PolicyRunner(Args... args) : p(std::make_unique<Policy>(args...)) {}
        bool should_load_balance() { return p->apply(); };
    };

    template<class Policy>
    class PolicyExecutor : public LBPolicy<Policy>{
        Probe* probe;
        Policy p;
    public:
        PolicyExecutor(Probe* probe, Policy p) : probe(probe), p(p) {}
        bool should_load_balance() { return p(*probe); };
    };

    class RandomPolicy {
        const Real lb_probability;
        std::mt19937 gen;
        mutable std::uniform_real_distribution<Real> dist = std::uniform_real_distribution<Real>(0.0, 1.0);
    public:
        RandomPolicy(Real lb_probability, int seed = 0) : lb_probability(lb_probability), gen(std::mt19937(seed)) {}
        bool apply() { return dist(gen) < lb_probability; }
    };

    class ThresholdPolicy {
        Probe* dataHolder;
        const std::function<Real (Probe*)> getDataF;
        const std::function<Real (Probe*)> getThresholdF;
    public:
        ThresholdPolicy(
                Probe* dataHolder,
                const std::function<Real (Probe*)> getDataF,
                const std::function<Real (Probe*)> getThresholdF) :
                dataHolder(dataHolder), getDataF(getDataF), getThresholdF(getThresholdF) {};
        bool apply() {
            return dataHolder != nullptr ? getDataF(dataHolder) >= getThresholdF(dataHolder) : false;
        }
    };
    template<class Policy>
    class CustomPolicy {
        Probe* dataHolder;
        Policy p;
    public:
        CustomPolicy(Probe* dataHolder, Policy p) : dataHolder(dataHolder), p(p) {};
        bool apply() {
            return p(*dataHolder);
        }
    };

    class InFilePolicy{
    public:
        mutable std::queue<bool> decisions;
        int period;
        Probe* probe;
        InFilePolicy(Probe* probe, std::string filename, int nframes, int npframe) : probe(probe) {
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
                std::stringstream ss(line);// Insert the string into a stream
                std::vector<float> tokens; // Create vector to hold our words
                while (ss >> buf) tokens.push_back(std::stof(buf.c_str()));
                clear = tokens.size() == 1;
                if(!clear) decisions.push( *(tokens.end() - 1) > 0 );
            }
            dataset.close();
        }

        bool apply() {
            if(probe->get_current_iteration() % period == 0) {
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
