# PyTorch Coding Rules

## General Principles

1. **Educational first**: Code should be readable and explainable
2. **From-scratch + verification**: Implement with NumPy, verify with PyTorch
3. **Gradient checking**: All implementations must pass gradient checks (< 1e-6 error)

## Code Structure

### Activation Functions
```python
# Forward pass
def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation: σ(x) = 1 / (1 + exp(-x))"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

# Backward pass (derivative)
def sigmoid_grad(x: np.ndarray) -> np.ndarray:
    """Derivative: σ'(x) = σ(x) * (1 - σ(x))"""
    s = sigmoid(x)
    return s * (1 - s)
```

### Loss Functions
```python
class MSELoss:
    def __init__(self):
        self.pred = None
        self.target = None

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        self.pred = pred
        self.target = target
        return np.mean((pred - target) ** 2)

    def backward(self) -> np.ndarray:
        n = self.pred.shape[0]
        return 2 * (self.pred - self.target) / n
```

### Optimizers
```python
class SGD:
    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def step(self, params: list, grads: list):
        for param, grad in zip(params, grads):
            param -= self.lr * grad
```

## Testing

### Gradient Check
```python
def gradient_check(func, x, eps=1e-5):
    """Numerical gradient approximation"""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        old_val = x[idx]
        x[idx] = old_val + eps
        f_plus = func(x)
        x[idx] = old_val - eps
        f_minus = func(x)
        x[idx] = old_val
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return grad
```

### Unit Test Pattern
```python
def test_sigmoid_gradient():
    x = np.random.randn(10, 5)
    analytical = sigmoid_grad(x)
    numerical = gradient_check(lambda x: np.sum(sigmoid(x)), x)
    assert np.allclose(analytical, numerical, atol=1e-6)
```

## Documentation

### Docstring Format
```python
def relu(x: np.ndarray) -> np.ndarray:
    """
    Rectified Linear Unit activation function.

    Formula: f(x) = max(0, x)

    Derivative: f'(x) = 1 if x > 0, else 0

    Args:
        x: Input array

    Returns:
        Output array with ReLU applied

    Notes:
        - Used in most modern neural networks
        - Simple and efficient to compute
        - Suffers from "dying ReLU" problem
    """
```

## Naming Conventions

| Component | Pattern | Example |
|-----------|---------|---------|
| Activation functions | `function_name`, `function_name_grad` | `sigmoid`, `sigmoid_grad` |
| Classes | PascalCase | `MSELoss`, `Adam` |
| Test functions | `test_<component>_<behavior>` | `test_sigmoid_gradient` |
| Files | snake_case | `activations.py`, `loss.py` |

## PyTorch Verification

Always verify implementations:
```python
import torch
import torch.nn as nn

# Compare NumPy implementation with PyTorch
numpy_out = sigmoid(x)
torch_out = torch.sigmoid(torch.tensor(x)).numpy()
assert np.allclose(numpy_out, torch_out)
```
