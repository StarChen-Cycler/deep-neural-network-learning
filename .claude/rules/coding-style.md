# Coding Style Guide

## General Rules

1. **Type hints required** for all function signatures
2. **Docstrings** for all public functions
3. **No magic numbers** - use constants
4. **Descriptive names** - avoid single letters except loop vars

## Python Style

### Formatting
- Line length: 100 characters max
- Indentation: 4 spaces
- Use Black for formatting: `black .`

### Imports
```python
# Standard library
import math
import random

# Third party
import numpy as np

# Local
from .activations import sigmoid, relu
```

### Type Hints
```python
def forward(x: np.ndarray) -> np.ndarray:
    ...

def backward(self, grad_output: np.ndarray) -> np.ndarray:
    ...
```

### Docstrings
```python
def gelu(x: np.ndarray) -> np.ndarray:
    """
    Gaussian Error Linear Unit activation.

    Formula: GELU(x) = x * Φ(x)
    where Φ is the CDF of standard normal distribution

    Approximation: GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))

    Args:
        x: Input array of any shape

    Returns:
        Array of same shape as input

    References:
        - Gaussian Error Linear Units (GELU): https://arxiv.org/abs/1606.08415
    """
```

## NumPy Style

### Array Operations
```python
# Good: vectorized
y = 1 / (1 + np.exp(-x))

# Bad: explicit loop
y = np.empty_like(x)
for i in range(len(x)):
    y[i] = 1 / (1 + np.exp(-x[i]))
```

### In-place Operations
```python
# When modifying input is acceptable
np.clip(x, -500, 500, out=x)

# When creating new array
x_clipped = np.clip(x, -500, 500)
```

### Broadcasting
```python
# Explicit broadcast
result = x * W + b  # (batch, features) @ (features, hidden) + (hidden,)

# Avoid unnecessary reshape
x = x.reshape(-1, 1)  # Only when needed
```

## Documentation

### Module Docstring
```python
"""
Neural network activation functions.

This module provides implementations of common activation functions
with both forward and backward (gradient) computations.

Functions:
    sigmoid: Sigmoid activation
    tanh: Hyperbolic tangent
    relu: Rectified Linear Unit
    ...
"""
```

### README Structure
```markdown
# Activation Functions

## Theory

Formula and derivatives...

## Implementation

```python
def sigmoid(x):
    ...
```

## Usage

```python
>>> x = np.array([0, 1, 2])
>>> sigmoid(x)
array([0.5, 0.731, 0.881])
```

## Tests

Run: `pytest tests/test_activations.py -v`
```
