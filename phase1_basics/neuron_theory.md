# Neuron Model and Activation Functions

This document covers the mathematical foundations of artificial neurons and the 6 common activation functions implemented in this project.

## Table of Contents

1. [Single Neuron Model](#single-neuron-model)
2. [Activation Functions Overview](#activation-functions-overview)
3. [Sigmoid](#sigmoid)
4. [Tanh](#tanh)
5. [ReLU](#relu)
6. [Leaky ReLU](#leaky-relu)
7. [GELU](#gelu)
8. [Swish](#swish)
9. [Comparison and Use Cases](#comparison-and-use-cases)

---

## Single Neuron Model

### Mathematical Definition

A single artificial neuron computes:

```
y = f(Σ(wᵢxᵢ) + b) = f(wx + b)
```

Where:
- **x** = input vector (x₁, x₂, ..., xₙ)
- **w** = weight vector (w₁, w₂, ..., wₙ)
- **b** = bias (scalar)
- **f** = activation function (non-linear)

### Components

1. **Linear Transformation**: `z = wx + b`
   - Weights determine the importance of each input
   - Bias shifts the decision boundary

2. **Non-linear Activation**: `y = f(z)`
   - Enables learning complex patterns
   - Without non-linearity, stacked layers collapse to a single linear transform

### Vectorized Form (Batch Processing)

For a batch of m samples:

```
Z = XW + b
Y = f(Z)
```

- **X**: (m, n) input matrix
- **W**: (n, h) weight matrix
- **b**: (h,) bias vector
- **Y**: (m, h) output matrix

---

## Activation Functions Overview

| Function | Formula | Range | Gradient at 0 |
|----------|---------|-------|---------------|
| Sigmoid | σ(x) = 1/(1+e⁻ˣ) | (0, 1) | 0.25 |
| Tanh | tanh(x) | (-1, 1) | 1.0 |
| ReLU | max(0, x) | [0, +∞) | 0 (by convention) |
| Leaky ReLU | max(αx, x) | (-∞, +∞) | α or 1 |
| GELU | x·Φ(x) | ≈(-0.17, +∞) | 0.5 |
| Swish | x·σ(x) | ≈(-0.28, +∞) | 0.5 |

---

## Sigmoid

### Formula

```
σ(x) = 1 / (1 + e⁻ˣ)
```

### Derivative

```
σ'(x) = σ(x) · (1 - σ(x))
```

At x = 0:
- σ(0) = 0.5
- σ'(0) = 0.5 · 0.5 = **0.25**

### Properties

- **Output range**: (0, 1) - good for probabilities
- **Smooth**: Differentiable everywhere
- **Monotonic**: Always increasing

### Issues

- **Vanishing gradient**: For |x| > 5, gradient approaches 0
- **Not zero-centered**: Outputs always positive
- **Expensive**: Requires exponential computation

### When to Use

- Binary classification output layer (with cross-entropy loss)
- **Avoid** in hidden layers of deep networks

### PyTorch Example

```python
import torch
import torch.nn as nn

# Using nn.Sigmoid
sigmoid = nn.Sigmoid()
output = sigmoid(torch.tensor([0.0, 1.0, -1.0]))
# tensor([0.5000, 0.7311, 0.2689])

# Using torch.sigmoid
output = torch.sigmoid(torch.tensor([0.0, 1.0, -1.0]))
```

---

## Tanh

### Formula

```
tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)
```

### Derivative

```
tanh'(x) = 1 - tanh(x)²
```

At x = 0:
- tanh(0) = 0
- tanh'(0) = **1.0**

### Properties

- **Output range**: (-1, 1) - zero-centered
- **Smooth**: Differentiable everywhere
- **Monotonic**: Always increasing

### Issues

- **Vanishing gradient**: For |x| > 3, gradient approaches 0

### When to Use

- Hidden layers in shallow networks
- RNN/LSTM gates (built-in)
- When zero-centered outputs are important

### PyTorch Example

```python
import torch
import torch.nn as nn

# Using nn.Tanh
tanh = nn.Tanh()
output = tanh(torch.tensor([0.0, 1.0, -1.0]))
# tensor([0.0000, 0.7616, -0.7616])

# Using torch.tanh
output = torch.tanh(torch.tensor([0.0, 1.0, -1.0]))
```

---

## ReLU

### Formula

```
ReLU(x) = max(0, x)
```

### Derivative

```
ReLU'(x) = 1 if x > 0, else 0
```

Note: At x = 0, the gradient is technically undefined (subgradient). Convention: use 0.

### Properties

- **Output range**: [0, +∞)
- **Computationally efficient**: Simple comparison
- **Sparse activation**: Many neurons output 0

### Issues

- **Dying ReLU**: Neurons can become permanently inactive (always output 0)
- **Not zero-centered**: Outputs always non-negative

### When to Use

- Default choice for hidden layers in most networks
- CNNs
- Deep networks (avoids vanishing gradient for positive inputs)

### PyTorch Example

```python
import torch
import torch.nn as nn

# Using nn.ReLU
relu = nn.ReLU()
output = relu(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]))
# tensor([0., 0., 0., 1., 2.])

# Using torch.relu (functional)
output = torch.relu(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]))

# Using F.relu
import torch.nn.functional as F
output = F.relu(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]))
```

---

## Leaky ReLU

### Formula

```
LeakyReLU(x) = x if x > 0, else αx
```

where α is typically 0.01.

### Derivative

```
LeakyReLU'(x) = 1 if x > 0, else α
```

### Properties

- **Output range**: (-∞, +∞)
- **Addresses dying ReLU**: Small gradient for negative inputs
- **Simple**: Efficient computation

### When to Use

- When dying ReLU is a problem
- Deep networks with many layers
- As an alternative to standard ReLU

### PyTorch Example

```python
import torch
import torch.nn as nn

# Using nn.LeakyReLU
leaky_relu = nn.LeakyReLU(negative_slope=0.01)
output = leaky_relu(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]))
# tensor([-0.0200, -0.0100, 0.0000, 1.0000, 2.0000])

# Using F.leaky_relu
import torch.nn.functional as F
output = F.leaky_relu(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]), negative_slope=0.01)
```

---

## GELU

### Exact Formula

```
GELU(x) = x · Φ(x)
```

where Φ(x) is the CDF of the standard normal distribution:
```
Φ(x) = 0.5 · (1 + erf(x/√2))
```

### Approximation (Common)

```
GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
```

This approximation is accurate to within 0.01%.

### Derivative

```
GELU'(x) = Φ(x) + x · φ(x)
```

where φ(x) is the PDF of standard normal:
```
φ(x) = exp(-x²/2) / √(2π)
```

At x = 0:
- GELU(0) = 0 · 0.5 = **0**
- GELU'(0) = 0.5 + 0 = **0.5**

### Properties

- **Smooth**: Differentiable everywhere (unlike ReLU)
- **Non-monotonic**: Has a slight dip for negative values
- **Used in Transformers**: BERT, GPT, etc.

### When to Use

- Transformer architectures
- When smooth gradients are important
- State-of-the-art results in NLP

### PyTorch Example

```python
import torch
import torch.nn.functional as F

# Using F.gelu with tanh approximation
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
output = F.gelu(x, approximate='tanh')
# tensor([-0.0454, -0.1588, 0.0000, 0.8412, 1.9546])

# Using exact (no approximation)
output = F.gelu(x, approximate='none')
```

---

## Swish

### Formula

```
Swish(x) = x · σ(x) = x / (1 + e⁻ˣ)
```

Also known as **SiLU** (Sigmoid Linear Unit).

### Derivative

```
Swish'(x) = σ(x) + x · σ(x) · (1 - σ(x))
          = σ(x) · (1 + x - x · σ(x))
```

At x = 0:
- Swish(0) = 0 · 0.5 = **0**
- Swish'(0) = 0.5 + 0 = **0.5**

### Properties

- **Self-gated**: Uses sigmoid as a soft gate
- **Smooth**: Differentiable everywhere
- **Non-monotonic**: Has a slight dip for negative values
- **Minimum**: At x ≈ -1.278, Swish(x) ≈ -0.278

### When to Use

- Deep networks where ReLU struggles
- Computer vision tasks
- As an alternative to GELU

### PyTorch Example

```python
import torch
import torch.nn.functional as F

# Using F.silu (same as Swish)
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
output = F.silu(x)
# tensor([-0.2384, -0.2689, 0.0000, 0.7311, 1.7616])

# Using nn.SiLU
import torch.nn as nn
swish = nn.SiLU()
output = swish(x)
```

---

## Comparison and Use Cases

### Gradient Flow Analysis

| Function | Vanishing Gradient? | Dying Neurons? |
|----------|---------------------|----------------|
| Sigmoid | Yes (large \|x\|) | No |
| Tanh | Yes (large \|x\|) | No |
| ReLU | No (x > 0) | Yes (x < 0) |
| Leaky ReLU | No | No |
| GELU | No | No |
| Swish | No | No |

### Recommendations by Architecture

| Architecture | Recommended Activation |
|--------------|----------------------|
| CNNs | ReLU or Leaky ReLU |
| Transformers | GELU |
| RNNs/LSTMs | Tanh (built-in) |
| Deep MLPs | GELU or Swish |
| Binary Classification Output | Sigmoid |
| Multi-class Classification Output | Softmax |
| Regression Output | Linear (none) |

### Gradient at Key Points

```python
# Verification of key gradients
import numpy as np

# Sigmoid at x=0
x = 0
sigmoid_grad = 0.25  # σ'(0) = σ(0) * (1 - σ(0)) = 0.5 * 0.5

# ReLU
x_neg = -1  # grad = 0
x_pos = 1   # grad = 1

# GELU at x=0
x = 0
gelu_grad = 0.5  # Φ(0) + 0 * φ(0) = 0.5
```

### Visual Comparison

```
Values:     -2.0   -1.0    0.0    1.0    2.0
─────────────────────────────────────────────
Sigmoid:    0.12   0.27   0.50   0.73   0.88
Tanh:      -0.96  -0.76   0.00   0.76   0.96
ReLU:       0.00   0.00   0.00   1.00   2.00
Leaky:     -0.02  -0.01   0.00   1.00   2.00  (α=0.01)
GELU:      -0.05  -0.16   0.00   0.84   1.95
Swish:     -0.24  -0.27   0.00   0.73   1.76
```

---

## Running Tests

```bash
# Run all activation tests
pytest tests/test_activations.py -v

# Run specific test class
pytest tests/test_activations.py::TestSigmoid -v
pytest tests/test_activations.py::TestGELU -v

# Run with coverage
pytest tests/test_activations.py --cov=phase1_basics.activations --cov-report=html
```

## References

1. **GELU**: Hendrycks, D., & Gimpel, K. (2016). [Gaussian Error Linear Units (GELU)](https://arxiv.org/abs/1606.08415)
2. **Swish**: Ramachandran, P., et al. (2017). [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
3. **ReLU**: Nair, V., & Hinton, G. E. (2010). Rectified Linear Units Improve Restricted Boltzmann Machines
4. **Leaky ReLU**: Maas, A. L., et al. (2013). Rectifier Nonlinearities Improve Neural Network Acoustic Models
