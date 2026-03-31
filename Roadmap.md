**Goal**: Maximum depth of understanding + implementation mastery

**Books**: 
- D2L = Dive into Deep Learning (Zhang et al.) - 987 pages
- GBC = Deep Learning (Goodfellow, Bengio, Courville) - 801 pages

**Timeline**: 3-4 months

---

## Phase 0: Foundation Refresh (1-2 weeks)

### Math Brush-Up (DO NOT skip this)
Since you'll be implementing from scratch, you need these concepts sharp:

**Linear Algebra** (refresh as needed):
- **GBC Chapter 2** (Linear Algebra) - Skip to sections you need:
  - 2.1-2.6: Basic operations (if rusty)
  - **2.7 Eigendecomposition** (crucial for understanding PCA, transformations)
  - **2.8 SVD** (essential for modern architectures)
  - 2.12 PCA example (connects theory to practice)

**Probability** (critical for deep learning):
- **GBC Chapter 3** (Probability and Information Theory):
  - 3.1-3.9: Core probability (skim if comfortable)
  - **3.13 Information Theory** (cross-entropy, KL divergence - MUST KNOW)
  - **3.14 Structured Probabilistic Models** (graphs - foundation for later)

**Calculus** (for backprop understanding):
- **D2L Section 2.4** (Calculus) - Quick, practical review
- **D2L Section 2.5** (Automatic Differentiation) - **Code this yourself in numpy/python**

**Action Items**:
- [ ] Implement matrix operations from scratch (matmul, transpose, etc.)
- [ ] Code autodiff for simple functions (build a tiny computational graph)
- [ ] Implement common probability distributions (Normal, Bernoulli)

---

## Phase 1: Core Deep Learning (3-4 weeks)

### Week 1: Linear Models → First Neural Network

**Theory**:
- **D2L Chapter 3** (Linear Neural Networks) - Complete chapter
  - 3.1: Linear Regression (understand loss surfaces)
  - 3.2: **Implement from scratch** (not library code)
  - 3.4: Softmax Regression
  - 3.6: **Implement softmax from scratch**

**Deep Dive**:
- **GBC Section 5.5** (Maximum Likelihood Estimation) - Understanding what you're optimizing
- **GBC Section 5.9** (Stochastic Gradient Descent) - Core algorithm variants

**Implementation Projects**:
- [ ] Linear regression from scratch (numpy only)
- [ ] Softmax classifier on MNIST (numpy only, no PyTorch/TensorFlow)
- [ ] Implement SGD with momentum
- [ ] Visualize loss landscapes (2D parameter space)

---

### Week 2: Multilayer Networks & Backpropagation

**Theory**:
- **D2L Section 4.1** (Multilayer Perceptrons)
- **D2L Section 4.7** (Forward Prop, Backward Prop, Comp Graphs) - **CRITICAL**
- **GBC Section 6.5** (Back-Propagation) - Rigorous mathematical treatment

**Deep Dive**:
- **GBC Section 6.1** (Learning XOR) - Why we need depth
- **GBC Section 6.2** (Gradient-Based Learning)
  - 6.2.2 Output units (understand different output types)
  - 6.2.1 Cost functions (when to use which)

**Implementation Projects**:
- [ ] **Backprop from scratch** - Build your own autograd engine (like micrograd)
  - Support: add, mul, ReLU, matmul
  - Compute gradients via reverse-mode autodiff
- [ ] MLP on MNIST (using YOUR autograd)
- [ ] Implement different activation functions (ReLU, tanh, sigmoid)
- [ ] Verify gradients with numerical gradient checking

---

### Week 3: Regularization & Optimization

**Theory**:
- **D2L Sections 4.4-4.6** (Underfitting/Overfitting, Weight Decay, Dropout)
- **GBC Chapter 7** (Regularization) - Pick key sections:
  - 7.1: L2/L1 regularization
  - **7.12: Dropout** (theory + intuition)
  - 7.8: Early stopping
  - 7.4: Data augmentation

**Optimization**:
- **D2L Section 4.8** (Numerical Stability & Initialization)
- **GBC Chapter 8** (Optimization) - Focus on:
  - 8.1: How learning differs from pure optimization
  - **8.3: Basic Algorithms** (SGD, momentum, Nesterov)
  - **8.5: Adaptive Learning Rates** (Adam, RMSprop, AdaGrad)
  - 8.4: Initialization strategies (Xavier, He)

**Implementation Projects**:
- [ ] Implement optimizers: SGD+momentum, Adam, RMSprop
- [ ] Implement dropout (train vs eval mode)
- [ ] Implement batch normalization (optional but recommended)
- [ ] Weight initialization schemes (Xavier, He)
- [ ] Training curves visualization (loss, accuracy, gradient norms)

---

### Week 4: CNNs - Vision Architecture

**Theory**:
- **D2L Chapter 6** (Convolutional Neural Networks) - Complete
  - Skip the specific architecture sections initially (6.6-6.11)
  - Focus on: convolution operation, pooling, why CNNs work
- **GBC Chapter 9** (Convolutional Networks):
  - **9.1-9.3**: Convolution, motivation, pooling
  - 9.5: Convolution variants (stride, padding, dilation)

**Implementation Projects**:
- [ ] **Convolution from scratch** (im2col method in numpy)
- [ ] Pooling layers (max, average)
- [ ] Simple CNN on CIFAR-10 (using YOUR implementations)
- [ ] Visualize learned filters
- [ ] Implement data augmentation (flips, crops, rotations)

**Optional Deep Dive**:
- GBC 9.10: Neuroscientific basis (optional but fascinating)
- Read classic papers: LeNet, AlexNet, VGG (architecture evolution)

---

## Phase 2: Modern Deep Learning (3-4 weeks)

### Week 5: Sequence Models - RNNs & LSTMs

**Theory**:
- **D2L Chapter 8** (Recurrent Neural Networks) - Complete
  - Focus on: 8.4 (RNNs), 8.7 (LSTM), 8.8 (GRU)
- **GBC Chapter 10** (Sequence Modeling):
  - **10.1-10.2**: Unfolding graphs, RNN basics
  - **10.7**: Long-term dependencies (vanishing gradients)
  - **10.10**: LSTM & gated mechanisms

**Implementation Projects**:
- [ ] Vanilla RNN from scratch (character-level language model)
- [ ] **LSTM from scratch** (all gates, cell state)
- [ ] Backpropagation through time (BPTT)
- [ ] Text generation (Shakespeare, code, etc.)
- [ ] Gradient clipping implementation

---

### Week 6: Attention & Transformers

**Theory**:
- **D2L Chapter 10** (Attention Mechanisms) - Complete
  - 10.1-10.3: Attention basics
  - 10.4: Multi-head attention
  - 10.6: Self-attention
  - 10.7: **Transformer architecture**

**Modern Context** (supplement with papers):
- Read "Attention is All You Need" (Vaswani et al.)
- Understand: Q, K, V matrices, scaled dot-product attention, positional encoding

**Implementation Projects**:
- [ ] **Attention mechanism from scratch** (Bahdanau, Luong styles)
- [ ] **Transformer encoder from scratch**:
  - Multi-head self-attention
  - Position-wise FFN
  - Layer normalization
  - Positional encoding
- [ ] Simple translation task or sequence task
- [ ] Visualize attention weights

---

### Week 7: Modern Architectures & Training

**Theory**:
- **D2L Sections 7.6-7.8** (ResNet, DenseNet, etc.)
- **GBC Section 8.6** (Second-order methods - understand but don't implement)
- **GBC Section 7.13** (Adversarial Training)

**Modern Additions** (supplement):
- Batch Normalization (understand forward & backward)
- Layer Normalization, Group Normalization
- Residual connections (skip connections)

**Implementation Projects**:
- [ ] Residual blocks from scratch
- [ ] Batch Normalization (forward + backward pass)
- [ ] Layer Normalization
- [ ] Build a ResNet-like architecture
- [ ] Implement gradient accumulation
- [ ] Mixed precision training (understand concepts)

---

### Week 8: Generative Models

**Theory**:
- **D2L Chapter 17** (Generative Adversarial Networks)
- **GBC Chapter 20** (Deep Generative Models):
  - 20.10: Directed Generative Nets (VAEs, GANs conceptually)
  - 20.14: Evaluating generative models

**Implementation Projects**:
- [ ] **Vanilla GAN from scratch**:
  - Generator network
  - Discriminator network
  - Adversarial training loop
  - Mode collapse handling
- [ ] **VAE from scratch**:
  - Encoder (recognition model)
  - Reparameterization trick
  - Decoder (generative model)
  - ELBO loss
- [ ] Generate images (MNIST, simple datasets)

**Modern Extension** (optional):
- Diffusion models (conceptual understanding + paper reading)
- Read DDPM paper, understand forward/reverse process

---

## Phase 3: Advanced Topics & Research Preparation (2-3 weeks)

### Week 9: Training at Scale & Practical Methods

**Theory**:
- **GBC Chapter 11** (Practical Methodology) - Entire chapter
  - 11.4: Hyperparameter selection
  - 11.5: Debugging strategies
- **GBC Chapter 12** (Applications) - Skim sections of interest

**Implementation Focus**:
- [ ] Implement learning rate schedulers (step, cosine, warmup)
- [ ] Hyperparameter search (grid, random, Bayesian)
- [ ] Gradient accumulation for large batches
- [ ] Model checkpointing & resuming
- [ ] TensorBoard-style logging
- [ ] Distributed data parallel (conceptual + basic implementation)

---

### Week 10: Deep Dive into One Domain

Pick ONE area to go deep (aligned with your research interests):

**Option A: Computer Vision**
- D2L Chapter 13 (Computer Vision)
- GBC 12.2 (Computer Vision section)
- Implement: Object detection (YOLO-style), Semantic segmentation

**Option B: NLP & Language Models**
- D2L Chapter 14-15 (NLP chapters)
- GBC 12.4 (NLP section)
- Implement: BERT-style model, GPT-style decoder

**Option C: Reinforcement Learning Basics**
- D2L Chapter 16 (RL)
- Implement: DQN, Policy Gradients

**Option D: Graph Neural Networks**
- Supplement with papers (GNNs not in these books)
- Implement: GCN, GAT from scratch

---

### Week 11-12: Advanced Theory & Research Topics

**Deep Learning Theory**:
- **GBC Part III** (Deep Learning Research) - Selective reading:
  - Chapter 14: Autoencoders
  - Chapter 15: Representation Learning
  - Chapter 16: Structured Probabilistic Models (if interested in graphical models)

**Modern Research Topics** (papers):
- Self-supervised learning (SimCLR, MoCo)
- Meta-learning (MAML)
- Neural Architecture Search
- Model compression (pruning, quantization, distillation)

**Implementation**:
- [ ] Reproduce a recent paper's core algorithm
- [ ] Build a mini deep learning library (Python)
- [ ] Start C++ implementation (basic ops: matmul, conv, backprop)

---

## Phase 4: C++ Deep Learning Library (Ongoing)

### Foundational Work

**Core Components** (build incrementally):
1. **Tensor library** (multi-dimensional arrays, broadcasting)
2. **Autograd engine** (computational graph, reverse-mode AD)
3. **Basic operations** (matmul, conv2d, pooling, activations)
4. **Optimizers** (SGD, Adam)
5. **Loss functions**
6. **Data loading** (simple batching)
7. **GPU support** (CUDA kernels for key ops)

**References**:
- Study PyTorch C++ internals (ATen library)
- Eigen library for matrix operations
- ArrayFire for GPU operations
- Write CUDA kernels for critical operations

**Milestones**:
- [ ] Train linear regression in your C++ library
- [ ] Train MLP on MNIST
- [ ] Train CNN on CIFAR-10
- [ ] Implement automatic differentiation
- [ ] GPU acceleration for key operations

---

## Critical Implementation Tips

### 1. **Always Code from Scratch First**
- Use numpy/Python before any framework
- Understand every line you write
- Verify with numerical gradient checking

### 2. **Debugging Strategy**
- Start with tiny datasets (10 samples)
- Overfit a single batch (should reach ~100% accuracy)
- Check gradient flow (print norms)
- Visualize everything (weights, activations, gradients)

### 3. **Testing Regimen**
- Unit test each component
- Compare against PyTorch/TensorFlow implementations
- Check: forward pass, backward pass, parameter updates

### 4. **Math → Code Pipeline**
- Write equations on paper first
- Derive gradients manually
- Translate to code step-by-step
- Verify shapes at each step

---

## What to SKIP (Don't Read These)

**From D2L**:
- Chapter 9 (Recurrent Modern) - outdated, transformers dominate
- Appendix sections (reference only)

**From GBC**:
- Chapter 17 (Monte Carlo) - unless doing probabilistic ML
- Chapter 18 (Partition Functions) - deep generative model theory (very specialized)
- Chapter 19 (Approximate Inference) - unless doing Bayesian DL
- Chapters 13, 20 sections on RBMs/DBNs - largely historical

**General Rule**: 
- Skip anything marked "historical" unless you're curious
- Skim application sections, focus on fundamentals
- Skip proofs unless they give intuition

---

## Recommended Parallel Resources

### Papers (read after implementing basics):
1. "Gradient-Based Learning Applied to Document Recognition" (LeNet)
2. "ImageNet Classification with Deep CNNs" (AlexNet)
3. "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"
4. "Batch Normalization: Accelerating Deep Network Training"
5. "Attention is All You Need" (Transformers)
6. "Deep Residual Learning for Image Recognition" (ResNet)

### Code References:
- PyTorch source code (for design patterns)
- Karpathy's micrograd (minimal autograd)
- Karpathy's minGPT (minimal transformer)
- JAX documentation (pure functional approach)

### Testing Understanding:
- After each major component, write a blog post explaining it
- Implement the same concept 3 times (from memory each time)
- Teach it to someone (or rubber duck)

---

## Success Metrics

After completing this roadmap, you should be able to:

- [ ] Implement any architecture from a paper (from scratch)
- [ ] Debug training issues (gradient vanishing, mode collapse, etc.)
- [ ] Explain backpropagation to a 5-year-old and a PhD
- [ ] Write a research paper's core algorithm in C++
- [ ] Build a basic deep learning library with GPU support
- [ ] Read and critique recent papers
- [ ] Identify which components are critical vs. engineering choices

---

## Final Notes

**On Speed**:
- This is 10-12 weeks if full-time (8+ hours/day)
- Part-time: 20-24 weeks (4 hours/day)
- The implementation work is where true understanding happens

**On Depth vs Breadth**:
- Better to deeply understand 5 architectures than superficially know 50
- Implement everything at least once from scratch
- Use frameworks only after you understand what they're doing

**On Research Readiness**:
- By the end, you'll be able to read 95% of papers
- Missing pieces (probabilistic ML, RL) can be filled as needed
- C++ library work will solidify systems-level thinking

**Most Important**: 
**CODE FIRST, READ SECOND**. When you hit a chapter, try implementing before reading. Get stuck, THEN read the explanation. This builds intuition faster than passive reading.

Good luck! The fact that you're asking for a roadmap shows you're serious. Stick to the implementation-heavy approach and you'll be building state-of-the-art models in no time.