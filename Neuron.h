//
// Created by Nyasha on 2018/09/16.
//

#ifndef BACK_PROPAGATION_NEURON_H
#define BACK_PROPAGATION_NEURON_H
#include <vector>

class Neuron {

double error;


public:
    std::vector<double> weights;
    std::vector<double> inputs;

    double output;
    double input;
    Neuron (std::vector<double> _weights):weights(std::move(_weights)){};

    double getError() const {
        return error;
    }

    void setError(double error) {
        Neuron::error = error;
    }

    const std::vector<double> &getWeights() const {
        return weights;
    }

    void setWeights(const std::vector<double> &weights) {
        Neuron::weights = weights;
    }

    double getOutput() const {
        return output;
    }

    void setOutput(double output) {
        Neuron::output = output;
    }

    double getInput() const {
        return input;
    }

    void setInput(double input) {
        Neuron::input = input;
    }
};


#endif //BACK_PROPAGATION_NEURON_H
