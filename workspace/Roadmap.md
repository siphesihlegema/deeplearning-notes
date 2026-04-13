# Deep Learning Learning Flow

## Phase 1: Foundations
* **Tensor Operations:** Mastery of multi-dimensional arrays, broadcasting, and memory-efficient slicing.
* **Gradient Calculation:** Understanding the chain rule and manual derivation of gradients for computational graphs.
* **[[Linear Regression]] from Scratch:** Implementation of the forward pass, MSE loss, and [[Stochastic Gradient Descent]] (SGD) without high-level libraries.
* **Softmax Classification:** Extending models to multi-class problems using cross-entropy loss and numerical stability techniques.

## Phase 2: Deep Networks & Optimization
* **Multilayer Perceptrons (MLP):** Implementing [[Hidden Layers]] and non-linear activation functions (ReLU, Sigmoid, Tanh).
* **Manual Backpropagation:** Coding the backward pass logic to update weights across multiple layers.
* **Regularization:** Implementing techniques to prevent overfitting, specifically Dropout and L2 Regularization (Weight Decay).
* **Advanced Optimizers:** Moving from basic SGD to adaptive methods like Adam, RMSProp, and Adagrad.

## Phase 3: Spatial & Sequential Models
* **Convolutional Neural Networks (CNN):** Implementing convolution kernels, pooling layers, padding, and stride for image processing.
* **Modern Vision Architectures:** Understanding residual connections (ResNet) and Batch Normalization for training stability.
* **Recurrent Neural Networks (RNN):** Modeling sequential data and managing the vanishing/exploding gradient problem.
* **Gated Units:** Implementing LSTMs and GRUs to handle long-term dependencies in sequences.

## Phase 4: Modern Deep Learning & Scaling
* **Attention Mechanisms:** Understanding how models dynamically weight different parts of an input sequence.
* **Transformers:** Implementing the Multi-Head Attention and Encoder-Decoder structure that powers modern LLMs.
* **Hardware & Performance:** Moving computations to the GPU and optimizing tensor operations for hardware acceleration.