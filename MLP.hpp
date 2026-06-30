#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <random>
#include <span>
#include <stdfloat>
#include <vector>

// data structure for train test split
struct DatasetSplit {
    const std::vector<std::float32_t>& X_train_raw;
    const std::vector<uint8_t>& y_train_raw;
    uint32_t train_samples;
    uint32_t test_samples;
    uint32_t test_start_idx;
};

class MLP {

public:

    // type
    uint8_t type = 0; 

    // neurons
    std::array<std::float32_t, 30> input; // 30 input neurons (one for each feature)
    std::array<std::array<std::float32_t, 16>, 5> hidden; // 5 intermediate layers of 16 neurons
    std::float32_t output; // output neuron with binary value for classification

    // weights
    // input to first hidden layer
    std::array<std::array<std::float32_t, 16>, 30> weights_in;

    // hidden-to-hidden transitions
    std::array<std::array<std::array<std::float32_t, 16>, 16>, 4> weights_hidden;

    // fifth hidden layer to output
    std::array<std::float32_t, 16> weights_out;

    // Biases
    std::array<std::array<std::float32_t, 16>, 5> bias_hidden;
    std::float32_t bias_out;

    // deltas
    // hidden layers
    std::array<std::array<std::float32_t, 16>, 5> deltas_hidden;

    // fifth hidden layer to output
    std::float32_t delta_out;

    // constructor (activation function is chosen here)
    MLP(uint8_t t) : type(t) {
        // generating random seed
        std::random_device rd;
        std::mt19937 gen(rd());
        std::float32_t constant;
        (type == 0 || type == 4) ? constant = 2.0f : constant = 1.0f;
        // initial input weights
        std::normal_distribution<std::float32_t> distIn(0.0f, std::sqrt(constant / 30.0f));
        for (size_t i = 0; i < 30; ++i) {
            for (size_t j = 0; j < 16; ++j) {
                weights_in[i][j] = distIn(gen);
            }
        }
        // initial hidden weights
        std::normal_distribution<std::float32_t> dist_hidden(0.0f, std::sqrt(constant / 16.0f));
        for (size_t l = 0; l < 4; ++l) {
            for (size_t i = 0; i < 16; ++i) {
                for (size_t j = 0; j < 16; ++j) {
                    weights_hidden[l][i][j] = dist_hidden(gen);
                }
            }
        }
        // initial output weights
        for (size_t i = 0; i < 16; ++i) {
            weights_out[i] = dist_hidden(gen);
        }
        // starting biases
        for (size_t l = 0; l < 5; ++l) {
            for (size_t i = 0; i < 16; ++i) {
                bias_hidden[l][i] = 0.01f;
            }
        }
        bias_out = 0.01f;
    }

    // training step 
    void runTrainingStep(const std::span<const std::float32_t> features, const bool label, const std::float32_t alpha = 0.005f) {
        // set up initial weights as features
        for (uint8_t featureNumber = 0; featureNumber < 30; featureNumber++) {
            input[featureNumber] = features[featureNumber];
        }
        // run forward pass
        forward();
        // run backpropagation 
        backpropagation(label);
        // adjust weights and biases
        adjustWeightsAndBiases(alpha);
    }

    // single epoch
    void runEpoch(const std::vector<std::float32_t> &X, const std::vector<uint8_t> &y, 
    const uint32_t trainSamples, const std::float32_t alpha = 0.005f) {
        #pragma omp parallel for schedule(static)
        for (uint32_t currentSample = 0; currentSample < trainSamples; currentSample++) {
            const std::span<const std::float32_t> features(X.data() + currentSample * 30, 30);
            const bool label = static_cast<bool>(y[currentSample]);
            #pragma omp critical 
            {
                runTrainingStep(features, label, alpha);
            }
        }
    }

    // run multiple epochs
    void trainNetworkSplit(const std::vector<std::float32_t> &X, const std::vector<uint8_t> &y, 
    const uint32_t trainSamples, const uint32_t iterations = 1000, const std::float32_t alpha = 0.005f) {
        for (uint32_t iteration = 1; iteration <= iterations; iteration++) {
            runEpoch(X, y, trainSamples, alpha);
            std::cout << "Epoch " << iteration << " done\n";
        }
    }

    // save weights and biases on a binary file
    bool saveModel(const std::string &filename) {
        // open file
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to open " << filename << " for writing weights." << std::endl;
            return false;
        }
        // write weights
        file.write(reinterpret_cast<const char*>(weights_in.data()), sizeof(weights_in));
        file.write(reinterpret_cast<const char*>(weights_hidden.data()), sizeof(weights_hidden));
        file.write(reinterpret_cast<const char*>(weights_out.data()), sizeof(weights_out));
        // write biases
        file.write(reinterpret_cast<const char*>(bias_hidden.data()), sizeof(bias_hidden));
        file.write(reinterpret_cast<const char*>(&bias_out), sizeof(bias_out));
        // finalizing
        file.close();
        std::cout << "Model weights successfully stored in " << filename << "!" << std::endl;
        return true;
    }

    // and load models stored using the previous method
    bool loadModel(const std::string &filename) {
        // open file
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to open " << filename << " for reading weights." << std::endl;
            return false;
        }
        // read weights
        file.read(reinterpret_cast<char*>(weights_in.data()), sizeof(weights_in));
        file.read(reinterpret_cast<char*>(weights_hidden.data()), sizeof(weights_hidden));
        file.read(reinterpret_cast<char*>(weights_out.data()), sizeof(weights_out));
        // read biases
        file.read(reinterpret_cast<char*>(bias_hidden.data()), sizeof(bias_hidden));
        file.read(reinterpret_cast<char*>(&bias_out), sizeof(bias_out));
        // finalizing
        file.close();
        std::cout << "Model weights successfully loaded from " << filename << "!" << std::endl;
        return true;
    }

    // splitting the dataset
    DatasetSplit prepareTrainTestSplit(const std::vector<std::float32_t>& X, const std::vector<uint8_t>& y, float train_ratio = 0.7f) {
        uint32_t totalSamples = y.size();
        uint32_t trainSamples = static_cast<uint32_t>(totalSamples * train_ratio);
        uint32_t testSamples = totalSamples - trainSamples;
        uint32_t testStartIdx = trainSamples;
        std::cout << "Total Samples: " << totalSamples << '\n';
        std::cout << "Training Set:  " << trainSamples << " samples\n";
        std::cout << "Testing Set:   " << testSamples << " samples\n";
        return DatasetSplit{X, y, trainSamples, testSamples, testStartIdx};
    }

    // evaluate the model
    void evaluateNetwork(const std::vector<std::float32_t> &X, const std::vector<uint8_t> &y, 
    const uint32_t test_start_idx, const uint32_t test_samples) {
        uint32_t correct_predictions = 0;
        // multi-threading magic line
        #pragma omp parallel for reduction(+:correct_predictions)
        for (uint32_t currentIndex = 0; currentIndex < test_samples; currentIndex++) {
            uint32_t currentSample = test_start_idx + currentIndex;
            const std::span<const std::float32_t> features(X.data() + currentSample * 30, 30);
            const bool trueLabel = static_cast<bool>(y[currentSample]);
            // run inference
            for (uint8_t f = 0; f < 30; f++) input[f] = features[f];
            forward(); 
            // check prediction veracity
            bool predictedLabel = (output > 0.5f);
            if (predictedLabel == trueLabel) {
                correct_predictions++;
            }
        }
        // outputs model accuracy   
        const std::float32_t accuracy = (static_cast<std::float32_t>(correct_predictions) / test_samples) 
        * static_cast<std::float32_t>(100);
        std::cout << "Test Accuracy: " << accuracy << "% (" << correct_predictions << '/' << test_samples << ")\n";
    }

private:

    // activation functions
    void activate(std::float32_t &neuron) {
        switch (type) {
            // ReLU
            case 0: {
                neuron = (neuron > 0.0f) ? neuron : 0.0f;
                break;
            }
            // GeLU
            case 1: {
                std::float32_t inner = 0.79788456f * (neuron + 0.044715f * neuron * neuron * neuron);
                neuron = 0.5f * neuron * (1.0f + std::tanh(inner));
                break;
            }
            // sigmoid
            case 2: {
                neuron = 1.0f / (1.0f + std::exp(-neuron));
                break;
            }
            // tanh
            case 3: {
                neuron = std::tanh(neuron);
                break;
            }
            // leaky ReLU
            case 4: {
                neuron = (neuron > 0.0f) ? neuron : 0.01f * neuron;
                break;
            }
        }
    }

    // derivative of activation functions
    std::float32_t derive(const std::float32_t neuron) {
        switch (type) {
            // ReLU
            case 0: {
                return (neuron > 0) ? 1.0f : 0.0f;
            }
            // GeLU
            case 1: {
                std::float32_t inner = 0.79788456f * (neuron + 0.044715f * neuron * neuron * neuron);
                std::float32_t myTanh = std::tanh(inner);
                std::float32_t mySechSquared = 1.0f - (myTanh * myTanh);
                return 0.5f * (1.0f + myTanh) + 0.5f * neuron * mySechSquared * 0.79788456f * (1.0f + 0.134145f * neuron * neuron);
            }
            // sigmoid
            case 2: {
                std::float32_t sigmoid = 1.0f / (1.0f + std::exp(-neuron));
                return sigmoid - (1.0f - sigmoid);
            }
            // tanh
            case 3: {
                std::float32_t myTanh = std::tanh(neuron);
                return 1.0f - myTanh * myTanh;
            }
            // leaky ReLU
            case 4: {
                return (neuron > 0) ? 1.0f : 0.01f;
            }
        }
        return 0;
    }

    // forward pass
    void forward() {
        // input to first hidden layer
        for (uint8_t currentLayerNeuron = 0; currentLayerNeuron < 16; currentLayerNeuron++) {
            hidden[0][currentLayerNeuron] = bias_hidden[0][currentLayerNeuron];
            for (uint8_t previousLayerNeuron = 0; previousLayerNeuron < 30; previousLayerNeuron++) {
                hidden[0][currentLayerNeuron] += input[previousLayerNeuron] 
                * weights_in[previousLayerNeuron][currentLayerNeuron];
            }
            // this is where the activation function goes
            activate(hidden[0][currentLayerNeuron]);
        }
        // hidden layer to hidden layer
        for (uint8_t hiddenLayer = 1; hiddenLayer < 5; hiddenLayer++) {
            for (uint8_t currentLayerNeuron = 0; currentLayerNeuron < 16; currentLayerNeuron++) {
                const uint8_t previousHiddenLayer = hiddenLayer - 1;
                hidden[hiddenLayer][currentLayerNeuron] = bias_hidden[hiddenLayer][currentLayerNeuron];
                for (uint8_t previousLayerNeuron = 0; previousLayerNeuron < 16; previousLayerNeuron++) {
                    hidden[hiddenLayer][currentLayerNeuron] += hidden[previousHiddenLayer][previousLayerNeuron] 
                    * weights_hidden[previousHiddenLayer][previousLayerNeuron][currentLayerNeuron];
                }
                // this is where the activation function goes
                activate(hidden[hiddenLayer][currentLayerNeuron]);
            }
        }
        // fifth hidden layer to output
        output = bias_out;
        for (uint8_t previousLayerNeuron = 0; previousLayerNeuron < 16; previousLayerNeuron++) {
            output += hidden[4][previousLayerNeuron] * weights_out[previousLayerNeuron];
        }
        // output uses sigmoid every time for binary classification
        output = 1.0f / (1.0f + std::exp(-output));
    }

    // backpropagation
    void backpropagation(const bool y) {
        delta_out = output - y;
        // starting by fifth hidden layer 
        for (uint8_t currentLayerNeuron = 0; currentLayerNeuron < 16; currentLayerNeuron++) {
            // this is where the derivative goes
            const std::float32_t derivative = derive(hidden[4][currentLayerNeuron]);
            deltas_hidden[4][currentLayerNeuron] = delta_out * weights_out[currentLayerNeuron] * derivative;
        }
        // remaining hidden layers looped through by overflow-triggered break because I still wanted to use uin8_t :b
        for (uint8_t hiddenLayer = 3; hiddenLayer < 4; hiddenLayer--) {
            for (uint8_t currentLayerNeuron = 0; currentLayerNeuron < 16; currentLayerNeuron++) {
                const uint8_t nextHiddenLayer = hiddenLayer + 1;
                // this is where the derivative goes
                const std::float32_t derivative = derive(hidden[hiddenLayer][currentLayerNeuron]);
                std::float32_t sum = 0;
                for (uint8_t nextLayerNeuron = 0; nextLayerNeuron < 16; nextLayerNeuron++) {
                    sum += deltas_hidden[nextHiddenLayer][nextLayerNeuron] 
                    * weights_hidden[hiddenLayer][currentLayerNeuron][nextLayerNeuron];
                }
                deltas_hidden[hiddenLayer][currentLayerNeuron] = sum * derivative;
            }
        }
    }

    // adjusting weights and biases
    void adjustWeightsAndBiases(const std::float32_t alpha) {
        // adjusting input weights
        for (uint8_t currentLayerNeuron = 0; currentLayerNeuron < 16; currentLayerNeuron++) {
            bias_hidden[0][currentLayerNeuron] -= alpha * deltas_hidden[0][currentLayerNeuron];
            for (uint8_t previousLayerNeuron = 0; previousLayerNeuron < 30; previousLayerNeuron++) {
                std::float32_t gradient = input[previousLayerNeuron] * deltas_hidden[0][currentLayerNeuron];
                weights_in[previousLayerNeuron][currentLayerNeuron] -= alpha * gradient;
            }
        }
        // adjusting hidden weights
        for (uint8_t hiddenLayer = 0; hiddenLayer < 4; hiddenLayer++) {
            const uint8_t nextHiddenLayer = hiddenLayer + 1;
            for (uint8_t currentLayerNeuron = 0; currentLayerNeuron < 16; currentLayerNeuron++) {
                // Update bias for the next layer's neurons
                bias_hidden[nextHiddenLayer][currentLayerNeuron] -= alpha * deltas_hidden[nextHiddenLayer][currentLayerNeuron];
                for (uint8_t previousLayerNeuron = 0; previousLayerNeuron < 16; previousLayerNeuron++) {
                    const std::float32_t gradient = hidden[hiddenLayer][previousLayerNeuron] 
                    * deltas_hidden[nextHiddenLayer][currentLayerNeuron];
                    weights_hidden[hiddenLayer][previousLayerNeuron][currentLayerNeuron] -= alpha * gradient;
                }
            }
        }
        // new output bias
        bias_out -= alpha * delta_out;
        // adjusting output weights
        for (uint8_t currentLayerNeuron = 0; currentLayerNeuron < 16; currentLayerNeuron++) {
            weights_out[currentLayerNeuron] -= alpha * hidden[4][currentLayerNeuron] * delta_out;
        }
    }

};