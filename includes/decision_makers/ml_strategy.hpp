//
// Created by xetql on 12/11/18.
//

#ifndef NBMPI_ML_STRATEGY_HPP
#define NBMPI_ML_STRATEGY_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/init_rules/const_init.hpp>
#include <mlpack/core/optimizers/rmsprop/rmsprop.hpp>
#include <mlpack/core/optimizers/sgd/update_policies/vanilla_update.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/prereqs.hpp>
#include "strategy.hpp"
class NeuralNetworkPolicy : public Policy {
    mlpack::ann::MeanSquaredError<> mse;
    mlpack::ann::RandomInitialization rand_init;
    mlpack::ann::FFN<mlpack::ann::MeanSquaredError<>, mlpack::ann::RandomInitialization> model;
    mlpack::optimization::RMSProp optimizer;
    std::shared_ptr<arma::mat> ds;
    arma::mat inputs, targets;
public:

    NeuralNetworkPolicy(const std::string& ds_filename, int idx) {
        inputs.load(ds_filename+"-features-"+std::to_string(idx)+".mat", arma::raw_ascii);
        targets.load(ds_filename+"-targets-"+std::to_string(idx)+".mat", arma::raw_ascii);
        auto tin = inputs.t().eval();
        auto ttar= targets.t().eval();
        model.Add<mlpack::ann::Linear<> >(tin.n_rows, 6);
        model.Add<mlpack::ann::LeakyReLU<> >();
        model.Add<mlpack::ann::Linear<> >(6, ttar.n_rows);
        model.Add<mlpack::ann::LeakyReLU<> >();
        model.Train(tin, ttar);
    }

    void train() {
        model.Train(inputs, targets, optimizer);
    }

    /**
     * Query the model
     */
    virtual inline bool should_load_balance(int it, std::unique_ptr<metric::LBMetrics<double>> mc) override {
        if(!mc) return false;
        arma::mat features(mc->metrics);
        arma::mat responses(1, 1);
        model.Predict(features, responses);
        return responses(0,0) >= 0;
    }

    void print(std::string prefix) override {
        Policy::print(prefix);
        std::cout << "NeuralNetwork Policy:" << std::endl;
    }
};
#endif //NBMPI_ML_STRATEGY_HPP
