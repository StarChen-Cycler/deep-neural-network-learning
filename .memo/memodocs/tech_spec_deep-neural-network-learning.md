# Tech Specification: Deep Neural Network Learning

## Project Overview
- **Project Name**: deep-neural-network-learning
- **Type**: Educational Tutorial Repository
- **Core Functionality**: Hands-on implementation of deep learning components from scratch with PyTorch
- **Target Users**: ML engineers, graduate students, developers learning DL

---

## Tech Stack

### Core Technologies
| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.10+ | Programming language |
| PyTorch | 2.x | Deep learning framework |
| NumPy | 2.x | Numerical computation (from-scratch implementations) |
| Matplotlib | - | Visualization |
| scikit-learn | - | Metrics, utilities |

### Development Tools
| Tool | Purpose |
|------|---------|
| pytest | Unit testing |
| black | Code formatting |
| mypy | Type checking |

---

## Architecture

### Directory Structure
```
deep-neural-network-learning/
├── .memo/                  # Project documentation
├── .octie/                 # Task management
├── phase1_basics/          # Neural network fundamentals
│   ├── activations.py      # 6 activation functions
│   ├── neuron.py           # Neuron model
│   ├── forward_backward.py # Forward/backward propagation
│   ├── loss.py             # 5 loss functions
│   ├── optimizer.py        # 6 optimizers
│   └── init.py             # 5 initialization methods
├── phase2_architectures/   # CNN, RNN, Transformer
│   ├── cnn.py              # CNN layers
│   ├── rnn.py              # RNN/LSTM/GRU
│   └── attention.py        # Self-attention
├── phase3_training/        # Training techniques
│   ├── norm.py             # 4 normalization methods
│   ├── dropout.py          # Dropout variants
│   └── scheduler.py        # LR schedulers
├── phase4_advanced/       # Advanced training
│   ├── mixed_precision.py  # FP16/BF16
│   ├── debug.py            # Training debugging
│   └── early_stopping.py   # Early stopping
├── phase5_deployment/      # Optimization & deployment
│   ├── augmentation.py     # Data augmentation
│   ├── ddp.py              # Distributed training
│   ├── pruning.py          # Model pruning
│   ├── quantization.py     # Model quantization
│   └── onnx.py             # ONNX export
├── tests/                  # Unit tests
├── notebooks/              # Jupyter notebooks
└── README.md
```

---

## Component Design

### Activation Functions
```python
class Activation:
    def forward(x) -> np.ndarray
    def backward(grad_output) -> np.ndarray  # derivative
```

### Loss Functions
```python
class Loss:
    def forward(pred, target) -> scalar
    def backward() -> grad_input
```

### Optimizers
```python
class Optimizer:
    def step(params, grads)
    def zero_grad()
```

### Neural Network Layers
```python
class Layer:
    def forward(x) -> output
    def backward(grad_output) -> grad_input
    def parameters() -> list of (param, grad)
```

---

## Implementation Patterns

### 1. From-Scratch + PyTorch Comparison
- Implement with NumPy first (educational)
- Verify with PyTorch autograd
- Gradient check: `|numerical_grad - analytical_grad| < 1e-6`

### 2. Testing Strategy
```python
def test_gradient(name, model, x, y):
    # Numerical gradient check
    grad = compute_numerical_gradient(model, x, y)
    analytical_grad = model.backward(y)
    assert np.allclose(grad, analytical_grad, atol=1e-6)
```

### 3. Documentation Format
- Theory: Math formulas, derivatives
- Code: Well-commented Python
- Experiments: Reproducible benchmarks

---

## API Design

### Activation Functions
```python
# Forward
def sigmoid(x): ...
def tanh(x): ...
def relu(x): ...
def leaky_relu(x, alpha=0.01): ...
def gelu(x): ...
def swish(x): ...

# Backward (gradient)
def sigmoid_grad(x): ...
# ... etc
```

### Loss Functions
```python
class MSELoss:
    def __init__(self): ...
    def forward(pred, target): ...
    def backward(): ...

class CrossEntropyLoss:
    def __init__(self): ...
    def forward(pred, target): ...
    def backward(): ...
```

### Optimizers
```python
class SGD:
    def __init__(self, lr=0.01): ...
    def step(params, grads): ...

class Adam:
    def __init__(self, lr=0.001, betas=(0.9, 0.999)): ...
    def step(params, grads): ...
```

---

## Data Flow

### Phase 1: Basics
```
Input → Linear (W× + b) → Activation → ... → Loss → Backward → Gradients → Optimizer Update
```

### Phase 2: Architectures
```
CNN: Input → Conv → Pool → FC → Output
RNN: Input → RNN Cell (t) → ... → Output
Attention: Q, K, V → Softmax(QK^T/√d) → V
```

### Phase 3-5: Training Pipeline
```
Data → Augmentation → Model → Loss → Backward → Optimizer → Checkpoint
                                              ↓
                                        Gradient Clipping
                                              ↓
                                        Mixed Precision (optional)
```

---

## Dependencies

### Task Dependencies (from Octie)
```
neuron → forward_backward → loss → optimizer → init → cnn/rnn/attention
                                                               ↓
cnn/rnn/attention → norm → dropout → scheduler → gradient_stability
                                                              ↓
gradient_stability → mixed_precision → debug/early_stopping
                                                              ↓
augmentation → ddp → pruning → quantization → onnx → distillation
                                                              ↓
architecture_comparison → edge_deployment → e2e_pipeline → hyperparam_tuning
                                                              ↓
gradient_accumulation → checkpoint → memory_optimization
```

---

## Key Libraries

| Library | Purpose | Notes |
|---------|---------|-------|
| torch | PyTorch core | Tensors, autograd, nn |
| torch.nn | Neural network modules | For verification |
| numpy | Array operations | From-scratch impl |
| scipy.special | Special functions | erf, etc. |
| matplotlib | Plotting | Learning curves |
| pytest | Testing | Unit tests |

---

## Success Criteria

### Code Quality
- All implementations pass gradient checks (< 1e-6 error)
- Unit test coverage for all components
- Type hints for all functions

### Educational Value
- Clear theory + code + experiments structure
- PyTorch comparison for each implementation
- Reproducible benchmarks

### Project Completion
- All 30 Octie tasks implemented
- Dependencies resolved correctly
- README with getting started guide
