# wasm-nn-cpp

**wasm-nn-cpp** is a neural network trainer and visualizer built from scratch in C++ and compiled to WebAssembly. This project showcases advanced C++ programming skills and a deep understanding of neural networks by combining high-performance native code with modern interactive web visualizations.

## Overview

The application lets users configure a neural network (e.g., architecture, learning rate, momentum, activation function, etc.), select a target function (such as XOR, Fancy Sine, Simple Sine, or Spiral), and then train the network. As training progresses, the app displays:
- A **logarithmic loss chart** (using Chart.js) to track training error.
- A **3D surface plot** (using ECharts) that shows both the target function and the model’s predictions—demonstrating how the network's output progressively approximates the target.

## Features

- **Neural Network from Scratch:**  
  Fully implemented in C++ with backpropagation, mini-batch training, and momentum.
  
- **WebAssembly Integration:**  
  Compiled with Emscripten, the network runs in the browser for high-performance computation.

- **Interactive Visualization:**  
  - **3D Surface Plot:** Displays the target function and model predictions over the input space.
  - **Logarithmic Loss Chart:** Shows the loss decreasing over epochs, capturing improvements on a log scale.
  
- **Target Function Selector:**  
  Choose between multiple functions (XOR, Fancy Sine, Simple Sine, Spiral) to see how the network learns different patterns.
  
- **Model Persistence:**  
  Save and load trained model weights to further experiment with training outcomes.

## Installation

1. **Install Emscripten:**  
   Follow the [Emscripten installation guide](https://emscripten.org/docs/getting_started/downloads.html).

2. **Compile the C++ Code:**  
   Open a terminal in your project folder and run:
   ```bash
   emcc mlp.cpp -o mlp.js -s MODULARIZE=1 -s EXPORT_ES6=1
