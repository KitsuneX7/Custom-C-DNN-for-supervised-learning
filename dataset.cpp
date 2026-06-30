#include <MLP.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <random>
#include <span>
#include <sstream>
#include <stdfloat>
#include <string>
#include <vector>

// dataset sizes for flattened 1D vectors
const size_t X_size = 8544210;
const size_t y_size = 284807;

bool readBinaryFileAsFloats(std::vector<std::float32_t> &dataset, const std::string &filename, const size_t size) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << filename << std::endl;
        return false;
    }
    file.read(reinterpret_cast<char*>(dataset.data()), size * sizeof(std::float32_t));
    file.close();   
    return true;
}

bool readBinaryFileAsBools(std::vector<uint8_t> &dataset, const std::string &filename, const size_t size) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << filename << std::endl;
        return false;
    }
    file.read(reinterpret_cast<char*>(dataset.data()), size * sizeof(uint8_t));
    file.close();
    return true;
}

int main() {
    // activation function choice
    std::cout << "Enter the desired activation function\n0 - ReLU\n1 - GeLU\n2 - sigmoid\n3 - tanh\n4 - leaky ReLU\n";
    uint16_t function;
    std::cin >> function;
    if (function > 4) return 1;
    // memory buffers
    std::vector<std::float32_t> X(X_size);
    std::vector<uint8_t> y(y_size);
    // read features (X)
    if (!readBinaryFileAsFloats(X, "../creditcard_X.bin", X_size)) return 1;
    // read label (y)
    if (!readBinaryFileAsBools(y, "../creditcard_y.bin", y_size)) return 1;
    // create network with activation function ReLU
    MLP myNetwork(function);
    // split into train and test
    DatasetSplit split = myNetwork.prepareTrainTestSplit(X, y, 0.7);
    // train the train part of the dataset
    myNetwork.trainNetworkSplit(X, y, split.train_samples, 100, 0.005f);
    // evaluate the generated model on the test part of the dataset
    myNetwork.evaluateNetwork(X, y, split.test_start_idx, split.test_samples);
    // save weights and biases to a binary file
    switch (function) {
        case 0: {
            myNetwork.saveModel("creditcard_ReLU_stored.bin");
        }
        case 1: {
            myNetwork.saveModel("creditcard_GeLU_stored.bin");
        }
        case 2: {
            myNetwork.saveModel("creditcard_sigmoid_stored.bin");
        }
        case 3: {
            myNetwork.saveModel("creditcard_tanh_stored.bin");
        }
        case 4: {
            myNetwork.saveModel("creditcard_leakyReLU_stored.bin");
        }
    }
    return 0;
}
