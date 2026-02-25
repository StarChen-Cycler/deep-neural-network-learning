# Testing Rules

## Test Structure

### File Organization
```
tests/
├── test_activations.py
├── test_loss.py
├── test_optimizer.py
├── test_layers.py
└── test_gradient_check.py
```

### Test Functions
```python
import pytest
import numpy as np

class TestActivations:
    """Test suite for activation functions"""

    def test_sigmoid_forward(self):
        """Test sigmoid forward pass"""
        x = np.array([0, 1, -1, 2])
        expected = 1 / (1 + np.exp(-x))
        result = sigmoid(x)
        assert np.allclose(result, expected)

    def test_sigmoid_gradient(self):
        """Test sigmoid gradient with numerical check"""
        x = np.random.randn(10, 5) * 2
        analytical = sigmoid_grad(x)

        # Numerical gradient
        eps = 1e-5
        numerical = np.zeros_like(x)
        it = np.nditer(x, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            old = x[idx]
            x[idx] = old + eps
            f_plus = np.sum(sigmoid(x))
            x[idx] = old - eps
            f_minus = np.sum(sigmoid(x))
            x[idx] = old
            numerical[idx] = (f_plus - f_minus) / (2 * eps)
            it.iternext()

        assert np.allclose(analytical, numerical, atol=1e-6)

    def test_relu_zero_gradient(self):
        """Test ReLU gradient is 0 for x < 0"""
        x = np.array([-2, -1, -0.5])
        grad = relu_grad(x)
        assert np.allclose(grad, 0)

    def test_relu_one_gradient(self):
        """Test ReLU gradient is 1 for x > 0"""
        x = np.array([0.5, 1, 2])
        grad = relu_grad(x)
        assert np.allclose(grad, 1)
```

## Gradient Checking

### Numerical Gradient Formula
```
∂L/∂x ≈ (L(x+ε) - L(x-ε)) / 2ε
```

### Check Threshold
- **Absolute error**: < 1e-6
- **Relative error**: < 1e-5

```python
def gradient_check(param, loss_fn, eps=1e-5):
    """Check analytical gradient against numerical"""
    analytical = loss_fn.backward()

    numerical = np.zeros_like(param)
    flat_param = param.flatten()
    flat_numerical = numerical.flatten()

    for i in range(len(flat_param)):
        old = flat_param[i]
        flat_param[i] = old + eps
        f_plus = loss_fn.forward(param)
        flat_param[i] = old - eps
        f_minus = loss_fn.forward(param)
        flat_param[i] = old
        flat_numerical[i] = (f_plus - f_minus) / (2 * eps)

    numerical = flat_numerical.reshape(param.shape)
    return np.allclose(analytical, numerical, atol=1e-6)
```

## PyTorch Comparison

```python
def test_against_pytorch(impl_fn, torch_fn, x):
    """Verify NumPy implementation matches PyTorch"""
    numpy_result = impl_fn(x)
    torch_result = torch_fn(torch.tensor(x)).detach().numpy()
    assert np.allclose(numpy_result, torch_result, atol=1e-6)
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_activations.py -v

# Run with coverage
pytest tests/ --cov=phase1_basics --cov-report=html
```

## Test Data

### Random Seed
```python
np.random.seed(42)  # Reproducible tests
```

### Test Values
- Small: `x = np.array([0, 1, -1])`
- Medium: `x = np.random.randn(10, 5)`
- Edge: `x = np.array([0, 1e-10, -1e-10, 1000, -1000])`
