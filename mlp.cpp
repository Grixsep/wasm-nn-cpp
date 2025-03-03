#include <vector>
#include <cmath>
#include <algorithm>
#include <emscripten/bind.h>

enum ActivationFunction { SIGMOID, RELU, TANH };

class MLP {
private:
    // weights[l] connects layer l to l+1 (dimensions: layerSizes[l] x layerSizes[l+1])
    std::vector<std::vector<std::vector<double>>> weights;
    // biases[l] for layer l+1 (dimensions: layerSizes[l+1])
    std::vector<std::vector<double>> biases;
    // velocities for momentum (same dimensions as weights and biases)
    std::vector<std::vector<std::vector<double>>> vWeights;
    std::vector<std::vector<double>> vBiases;
    
    std::vector<int> layerSizes;
    double learningRate;
    double momentumBeta;
    ActivationFunction activationFn;

    double applyActivation(double x) {
        switch (activationFn) {
            case SIGMOID: return 1.0 / (1.0 + exp(-x));
            case RELU:    return std::max(0.0, x);
            case TANH:    return tanh(x);
        }
        return x;
    }

    double activationDerivative(double activatedValue) {
        switch (activationFn) {
            case SIGMOID: return activatedValue * (1.0 - activatedValue);
            case RELU:    return activatedValue > 0 ? 1.0 : 0.0;
            case TANH:    return 1.0 - activatedValue * activatedValue;
        }
        return 1.0;
    }

public:
    // Constructor: accepts network layers (e.g., {2,4,1}), learning rate, momentum, and activation function.
    MLP(std::vector<int> layers, double lr, double momentum, ActivationFunction actFn)
        : layerSizes(layers), learningRate(lr), momentumBeta(momentum), activationFn(actFn)
    {
        weights.resize(layers.size() - 1);
        biases.resize(layers.size() - 1);
        vWeights.resize(layers.size() - 1);
        vBiases.resize(layers.size() - 1);
        for (size_t l = 0; l < layers.size() - 1; l++) {
            weights[l].resize(layers[l], std::vector<double>(layers[l + 1], 0.5));
            biases[l].resize(layers[l + 1], 0.5);
            vWeights[l].resize(layers[l], std::vector<double>(layers[l + 1], 0.0));
            vBiases[l].resize(layers[l + 1], 0.0);
        }
    }

    // Forward pass: returns the network's output for a given input vector.
    std::vector<double> predict(std::vector<double> input) {
        std::vector<double> activation = input;
        for (size_t l = 0; l < weights.size(); l++) {
            std::vector<double> nextActivation = biases[l];
            for (size_t j = 0; j < biases[l].size(); j++) {
                for (size_t i = 0; i < activation.size(); i++) {
                    nextActivation[j] += activation[i] * weights[l][i][j];
                }
                nextActivation[j] = applyActivation(nextActivation[j]);
            }
            activation = nextActivation;
        }
        return activation;
    }

    // Full backpropagation with mini-batch training and momentum.
    // If batchSize <= 0 or > number of samples, full batch is used.
    std::vector<double> trainAndReturnLoss(std::vector<std::vector<double>> X,
                                           std::vector<std::vector<double>> Y,
                                           int epochs, int batchSize) {
        std::vector<double> epochLosses;
        int nSamples = X.size();
        if (batchSize <= 0 || batchSize > nSamples)
            batchSize = nSamples;
        
        for (int e = 0; e < epochs; e++) {
            double epochLoss = 0.0;
            // Process training data in mini-batches.
            for (int batchStart = 0; batchStart < nSamples; batchStart += batchSize) {
                int batchEnd = std::min(batchStart + batchSize, nSamples);
                // Initialize gradient accumulators.
                std::vector<std::vector<std::vector<double>>> gradW(weights.size());
                std::vector<std::vector<double>> gradB(biases.size());
                for (size_t l = 0; l < weights.size(); l++) {
                    gradW[l].resize(weights[l].size(), std::vector<double>(weights[l][0].size(), 0.0));
                    gradB[l].resize(biases[l].size(), 0.0);
                }
                // For each sample in the batch:
                for (int sample = batchStart; sample < batchEnd; sample++) {
                    // Forward pass:
                    std::vector<std::vector<double>> activations;
                    activations.push_back(X[sample]);
                    for (size_t l = 0; l < weights.size(); l++) {
                        std::vector<double> layerOut = biases[l];
                        for (size_t j = 0; j < biases[l].size(); j++) {
                            for (size_t i = 0; i < activations[l].size(); i++) {
                                layerOut[j] += activations[l][i] * weights[l][i][j];
                            }
                            layerOut[j] = applyActivation(layerOut[j]);
                        }
                        activations.push_back(layerOut);
                    }
                    std::vector<double> output = activations.back();
                    double sampleLoss = 0.0;
                    std::vector<double> outputError(output.size());
                    for (size_t j = 0; j < output.size(); j++) {
                        double error = Y[sample][j] - output[j];
                        sampleLoss += error * error;
                        outputError[j] = error * activationDerivative(output[j]);
                    }
                    sampleLoss /= output.size();
                    epochLoss += sampleLoss;
                    
                    int L = activations.size();
                    // Backward pass: Compute deltas for each layer.
                    std::vector<std::vector<double>> deltas(L);
                    deltas[L - 1] = outputError;
                    for (int l = L - 2; l > 0; l--) {
                        deltas[l].resize(activations[l].size(), 0.0);
                        for (size_t i = 0; i < activations[l].size(); i++) {
                            double deltaSum = 0.0;
                            for (size_t j = 0; j < deltas[l + 1].size(); j++) {
                                deltaSum += weights[l][i][j] * deltas[l + 1][j];
                            }
                            deltas[l][i] = deltaSum * activationDerivative(activations[l][i]);
                        }
                    }
                    // Accumulate gradients.
                    for (size_t l = 0; l < weights.size(); l++) {
                        for (size_t i = 0; i < weights[l].size(); i++) {
                            for (size_t j = 0; j < weights[l][i].size(); j++) {
                                gradW[l][i][j] += activations[l][i] * deltas[l + 1][j];
                            }
                        }
                        for (size_t j = 0; j < biases[l].size(); j++) {
                            gradB[l][j] += deltas[l + 1][j];
                        }
                    }
                } // End of batch.
                int currentBatchSize = batchEnd - batchStart;
                // Update weights and biases using accumulated gradients and momentum.
                for (size_t l = 0; l < weights.size(); l++) {
                    for (size_t i = 0; i < weights[l].size(); i++) {
                        for (size_t j = 0; j < weights[l][i].size(); j++) {
                            double grad = gradW[l][i][j] / currentBatchSize;
                            vWeights[l][i][j] = momentumBeta * vWeights[l][i][j] - learningRate * grad;
                            weights[l][i][j] += vWeights[l][i][j];
                        }
                    }
                    for (size_t j = 0; j < biases[l].size(); j++) {
                        double grad = gradB[l][j] / currentBatchSize;
                        vBiases[l][j] = momentumBeta * vBiases[l][j] - learningRate * grad;
                        biases[l][j] += vBiases[l][j];
                    }
                }
            } // End of epoch.
            epochLosses.push_back(epochLoss / nSamples);
        }
        return epochLosses;
    }

    std::vector<std::vector<std::vector<double>>> getWeights() {
        return weights;
    }

    void setWeights(std::vector<std::vector<std::vector<double>>> newWeights) {
        weights = newWeights;
    }
};

EMSCRIPTEN_BINDINGS(mlp_module) {
    emscripten::class_<MLP>("MLP")
        .constructor<std::vector<int>, double, double, ActivationFunction>()
        .function("predict", &MLP::predict)
        .function("trainAndReturnLoss", &MLP::trainAndReturnLoss)
        .function("getWeights", &MLP::getWeights)
        .function("setWeights", &MLP::setWeights);

    emscripten::enum_<ActivationFunction>("ActivationFunction")
        .value("SIGMOID", SIGMOID)
        .value("RELU", RELU)
        .value("TANH", TANH);
}
