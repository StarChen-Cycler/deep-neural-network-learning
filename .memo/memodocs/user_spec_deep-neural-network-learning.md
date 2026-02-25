# Project Overview

## Project Name
Deep Neural Network Learning - Industrial Implementation Tutorial

## Project Type
Educational/Technical Tutorial Repository

## Core Feature Summary
A comprehensive hands-on tutorial project teaching industrial deep learning implementation through 30 practical tasks, covering neural network foundations, architectures, training techniques, and deployment optimization.

## Target Users
- Machine learning engineers transitioning to deep learning
- Graduate students learning DL implementation
- Engineers who want to understand "how it works under the hood"
- Developers preparing for ML engineer interviews

---

# User Stories

## Learning Path
1. As a learner, I want to implement neural network components from scratch to understand the mathematical foundations
2. As a learner, I want to verify my implementations against PyTorch autograd to ensure correctness
3. As a learner, I want to understand when to use different activation functions, optimizers, and normalization techniques
4. As a learner, I want to see real-world performance comparisons (speed, memory, accuracy)

## Implementation Focus
1. As a developer, I want numpy implementations to understand the math behind each component
2. As a developer, I want PyTorch implementations that match industrial standards
3. As a developer, I want unit tests with numerical gradient verification
4. As a developer, I want clear success criteria for each implementation

## Skill Progression
1. As a learner, I want to progress from basic (neuron, activation) to advanced (pruning, quantization, deployment)
2. As a learner, I want each task to build on previous tasks (clear dependencies)
3. As a learner, I want to see theory + code + experiments for each topic

---

# Core Features

## Phase 1: Neural Network Basics (7 tasks)
- Neuron model with 6 activation functions (Sigmoid, Tanh, ReLU, Leaky ReLU, GELU, Swish)
- Forward and backward propagation with gradient verification
- 5 common loss functions (MSE, CrossEntropy, Focal, Label Smoothing, Triplet)
- 6 optimizer variants (SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam)
- 5 weight initialization methods (Xavier, He, Kaiming, LSUV, Zero)

## Phase 2: Architectures (3 tasks)
- CNN layers with receptive field calculation
- RNN/LSTM/GRU with BPTT
- Self-Attention and Multi-Head Attention

## Phase 3: Training Techniques (4 tasks)
- 4 normalization techniques (BatchNorm, LayerNorm, InstanceNorm, GroupNorm)
- 4 dropout variants + L1/L2 regularization
- 5 learning rate schedulers
- Gradient stability (vanishing/explosion solutions)

## Phase 4: Advanced Training (4 tasks)
- Mixed precision training (FP16/BF16)
- Training debugging and monitoring
- Early stopping callback
- NaN loss debugging

## Phase 5: Optimization & Deployment (12 tasks)
- Data augmentation (image + text)
- Distributed Data Parallel (DDP)
- Model pruning (structured + unstructured)
- Model quantization (PTQ + QAT)
- ONNX export and inference
- Knowledge distillation
- Architecture comparison (CNN vs Transformer vs RNN)
- Edge deployment (TensorRT, NCNN)
- End-to-end pipeline
- Hyperparameter tuning
- Gradient accumulation
- Checkpoint save/resume
- Memory optimization

---

# Edge Cases

1. **Gradient edge cases**: ReLU dying, gradient explosion in deep networks
2. **Numerical stability**: NaN/Inf in mixed precision, loss scaling
3. **Hardware differences**: GPU vs CPU behavior, memory limits
4. **Framework compatibility**: PyTorch version differences
5. **Task dependencies**: Blocked tasks must wait for prerequisites

---

# Technical Constraints

1. **PyTorch version**: Latest stable (2.x)
2. **Python version**: 3.10+
3. **Hardware**: CUDA-capable GPU preferred, CPU fallback supported
4. **No external ML frameworks**: Implement from scratch with numpy where educational

---

# Success Criteria

Each task must have:
- Working code implementation
- Unit tests with numerical verification
- Theory documentation (formulas, derivations)
- PyTorch comparison/validation
- Clear success criteria and deliverables

---

# Project Structure

```
deep-neural-network-learning/
├── .memo/                  # Project documentation
├── .octie/                 # Task management
├── phase1_basics/          # Neural network fundamentals
├── phase2_architectures/   # CNN, RNN, Transformer
├── phase3_training/        # Training techniques
├── phase4_advanced/        # Advanced training
├── phase5_deployment/      # Optimization & deployment
└── README.md               # Project overview
```
