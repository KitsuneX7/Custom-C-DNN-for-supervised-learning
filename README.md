# Custom High-Performance Deep Neural Network (C++20)

Welcome to a bare-metal, hyper-optimized Deep Learning implementation built completely from scratch in modern C++. This framework is engineered for raw mechanical sympathy—eschewing heavy, bloated machine learning libraries in favor of explicit memory layouts, flat stack arrays, and max-capacity CPU utilization.

The engine is currently benchmarked against the famous ULB Credit Card Fraud Detection dataset (using the PCA-transformed feature sets) to achieve exceptional binary classification accuracy right at the statistical limits of the dataset.

## Key Architectural Highlights

* **Cache-Locality First:** Built completely using contiguous `std::array` stacks over pointer-chasing heap allocations, minimizing L1/L2 cache misses.
* **Modern C++ Semantics:** Heavy utilization of `std::span` for zero-overhead, bounds-safe memory views and `<stdfloat>` for explicit hardware-width registration.
* **Bare-Metal Parallelism:** Multithreading powered via lightweight OpenMP scheduling, processing hundreds of thousands of samples across multi-core architectures in mere seconds.
* **Flexible Activation Matrix:** Supports dynamic switching between runtime profiles for traditional and modern activation functions, including ReLU, GeLU, Sigmoid, Tanh, and LeakyReLU.

---

## Getting Started & Usage

To get the framework up and running, follow these steps to prepare your dataset, compile the binary, and execute the interactive training suite.

### 1. Compilation
Ensure you have a modern C++ compiler supporting C++20 and OpenMP enabled. Compile the project with maximum optimization flags using the following command:

g++ -O3 -std=c++20 -fopenmp output/dataset.cpp -o nnet_benchmark

### 2. Execution
Run the compiled binary directly from your terminal with this command:

./nnet_benchmark

### 3. Interactive Configuration
Upon launching, the program will prompt you to enter a configuration integer corresponding to your desired activation function profile. Type your chosen integer into the console prompt and hit Enter:

* **0** : ReLU
* **1** : GeLU
* **2** : Sigmoid
* **3** : Tanh
* **4** : LeakyReLU

The network will automatically partition the dataset into training and testing slices, spin up its multi-threaded training loops, and output the tracking performance across your evaluation epochs!

---

## Repository Structure

* **output/dataset.cpp** - Core execution blueprint and interactive CLI loop handling.
* **MLP.hpp** - The central engine architecture housing forward propagation, backpropagation calculus, and memory-aligned layer matrices.

---

## Data Attribution
This project utilizes the Open Database License (ODbL) 1.0 Credit Card Fraud Detection dataset created by the Machine Learning Group at Université Libre de Bruxelles (ULB).
