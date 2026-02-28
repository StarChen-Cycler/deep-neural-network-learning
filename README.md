# Deep Neural Network Learning

A comprehensive hands-on deep learning implementation project with 30 tasks covering neural network fundamentals to production deployment.

## Project Overview

This project implements deep learning components from scratch with NumPy, verified against PyTorch autograd. All implementations pass gradient checks (<1e-6 error) and include comprehensive documentation.

### Tech Stack

| Component | Version |
|-----------|---------|
| Python | 3.10+ |
| PyTorch | 2.4.1+cu124 |
| CUDA | 12.4 |
| NumPy | 2.x |
| pytest | Latest |

### Hardware Target

- GPU: NVIDIA RTX 3050 Ti (4GB VRAM)
- Optimized for limited VRAM with mixed precision, gradient checkpointing, and accumulation

---

## Project Structure

```
deep-neural-network-learning/
├── phase1_basics/           # Neural Network Fundamentals
│   ├── activations.py       # 6 activation functions (Sigmoid, Tanh, ReLU, LeakyReLU, GELU, Swish)
│   ├── loss.py              # 5 loss functions (MSE, CrossEntropy, Focal, LabelSmoothing, Triplet)
│   ├── mlp.py               # Forward/backward propagation with computational graph
│   ├── optimizer.py         # 6 optimizers (SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam)
│   └── weight_init.py       # 5 initialization methods (Xavier, He, Kaiming, LSUV, Zero)
│
├── phase2_architectures/    # Neural Network Architectures
│   ├── cnn_layers.py        # Conv2d, MaxPool2d, AvgPool2d with im2col/col2im
│   ├── simple_cnn.py        # ResNet-style CNN with BatchNorm, ResidualBlock
│   ├── rnn_cells.py         # RNN, LSTM, GRU with BPTT gradient verification
│   └── attention.py         # Self-Attention, Multi-Head Attention, Position Encoding
│
├── phase3_training/         # Training Techniques
│   ├── normalization.py     # BatchNorm, LayerNorm, InstanceNorm, GroupNorm
│   ├── dropout.py           # Standard, Variational, MC, Alpha Dropout
│   ├── regularization.py    # L1, L2, ElasticNet, MaxNorm, SpectralNorm
│   ├── lr_scheduler.py      # 11 schedulers (Step, Cosine, Warmup, OneCycle, etc.)
│   ├── image_augmentation.py # RandomCrop, Flip, Rotation, ColorJitter, Mixup, CutMix
│   ├── text_augmentation.py  # Token masking, synonym replacement
│   └── transfer_learning.py  # Pretrained models, freeze strategies, discriminative LR
│
├── phase4_advanced/         # Advanced Training
│   ├── mixed_precision.py   # FP16/BF16/TF32 with GradScaler
│   ├── gradient_stability.py # Gradient clipping, residual connections
│   ├── early_stopping.py    # Patience counter, best weights restoration
│   ├── training_monitor.py  # Gradient flow, activation distribution
│   ├── tensorboard_debug.py # TensorBoard integration
│   └── nan_debugger.py      # NaN detection, LR auto-adjustment
│
├── phase5_deployment/       # Optimization & Deployment
│   ├── ddp_training.py      # Distributed Data Parallel with NCCL
│   ├── multi_gpu.py         # Multi-GPU configuration
│   ├── gradient_accumulation.py # Memory-efficient large batch training
│   ├── memory_optimizer.py  # Gradient checkpointing, CPU offloading
│   ├── checkpoint_manager.py # Save/resume training state
│   ├── pruning.py           # Magnitude, Gradient, Channel pruning
│   ├── quantization.py      # PTQ, QAT, INT8/INT4 quantization
│   ├── onnx_export.py       # PyTorch to ONNX with dynamic axes
│   ├── onnx_inference.py    # ONNX Runtime inference
│   ├── distillation.py      # Knowledge distillation with temperature scaling
│   ├── tensorrt_inference.py # TensorRT FP16/INT8 acceleration
│   └── mobile_deployment.py  # NCNN, Core ML export
│
├── tests/                   # Comprehensive Test Suite (500+ tests)
│   └── test_*.py            # All modules tested with gradient verification
│
└── experiments/             # Benchmarks & Comparisons
    ├── optimizer_comparison.py
    ├── architecture_comparison.py
    └── amp_benchmark.py
```

---

## Completed Features (27/30 Tasks)

### Phase 1: Neural Network Fundamentals (5/5)

| Task | Description | File |
|------|-------------|------|
| Activation Functions | 6 functions with gradient verification | `phase1_basics/activations.py` |
| Forward/Backward Propagation | MLP with computational graph | `phase1_basics/mlp.py` |
| Loss Functions | MSE, CrossEntropy, Focal, LabelSmoothing, Triplet | `phase1_basics/loss.py` |
| Optimizers | SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam | `phase1_basics/optimizer.py` |
| Weight Initialization | Xavier, He, Kaiming, LSUV | `phase1_basics/weight_init.py` |

### Phase 2: Architectures (3/3)

| Task | Description | File |
|------|-------------|------|
| CNN Architecture | Conv2d, Pooling, ResNet-style blocks | `phase2_architectures/cnn_layers.py` |
| RNN/LSTM/GRU | Sequential models with BPTT | `phase2_architectures/rnn_cells.py` |
| Self-Attention | Multi-head attention, position encoding | `phase2_architectures/attention.py` |

### Phase 3: Training Techniques (5/5)

| Task | Description | File |
|------|-------------|------|
| Normalization | BatchNorm, LayerNorm, InstanceNorm, GroupNorm | `phase3_training/normalization.py` |
| Dropout & Regularization | 4 dropout variants, L1/L2/ElasticNet | `phase3_training/dropout.py` |
| LR Schedulers | 11 scheduling strategies | `phase3_training/lr_scheduler.py` |
| Data Augmentation | Image (Mixup, CutMix) + Text (Token masking) | `phase3_training/image_augmentation.py` |
| Transfer Learning | Freeze strategies, discriminative LR | `phase3_training/transfer_learning.py` |

### Phase 4: Advanced Training (4/4)

| Task | Description | File |
|------|-------------|------|
| Mixed Precision | FP16/BF16 with GradScaler | `phase4_advanced/mixed_precision.py` |
| Gradient Stability | Clipping, residual connections | `phase4_advanced/gradient_stability.py` |
| Training Debugging | TensorBoard, gradient visualization | `phase4_advanced/training_monitor.py` |
| Early Stopping | Patience counter, best weights | `phase4_advanced/early_stopping.py` |

### Phase 5: Deployment (10/13)

| Task | Description | File |
|------|-------------|------|
| DDP Training | Distributed data parallel | `phase5_deployment/ddp_training.py` |
| Gradient Accumulation | Memory-efficient training | `phase5_deployment/gradient_accumulation.py` |
| Memory Optimization | Gradient checkpointing, CPU offloading | `phase5_deployment/memory_optimizer.py` |
| Checkpoint/Resume | Save and restore training state | `phase5_deployment/checkpoint_manager.py` |
| Model Pruning | Magnitude, gradient, channel pruning | `phase5_deployment/pruning.py` |
| Model Quantization | PTQ, QAT, INT8/INT4 | `phase5_deployment/quantization.py` |
| ONNX Export | PyTorch to ONNX conversion | `phase5_deployment/onnx_export.py` |
| Knowledge Distillation | Teacher-student training | `phase5_deployment/distillation.py` |
| Architecture Comparison | CNN vs Transformer vs RNN | `experiments/architecture_comparison.py` |
| NaN Debugging | Loss instability diagnosis | `phase4_advanced/nan_debugger.py` |

---

## In Progress (1/30)

| Task | Description | Status |
|------|-------------|--------|
| Edge Deployment | TensorRT, NCNN, Core ML | Deliverables complete, hardware validation pending |

---

## Planned Features (2/30)

| Task | Description | Blocker |
|------|-------------|---------|
| E2E Image Classification | Full pipeline: data -> training -> deployment | Edge deployment |
| Hyperparameter Tuning | Grid, Random, Bayesian, Hyperband | E2E pipeline |

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/StarChen-Cycler/deep-neural-network-learning.git
cd deep-neural-network-learning

# Create conda environment
conda create -n dnn python=3.10
conda activate dnn

# Install dependencies
pip install torch torchvision numpy pytest

# Run tests
pytest tests/ -v

# Run specific module
python -m phase1_basics.activations
```

---

## Key Implementation Highlights

### Gradient Verification
All implementations pass numerical gradient checking:
```python
# Example: Activation gradient verification
from phase1_basics.activations import sigmoid, sigmoid_grad
from tests.test_activations import gradient_check

x = np.random.randn(10, 5)
analytical = sigmoid_grad(x)
numerical = gradient_check(lambda x: np.sum(sigmoid(x)), x)
assert np.allclose(analytical, numerical, atol=1e-6)
```

### Memory Optimization for 4GB VRAM
```python
from phase5_deployment.gradient_accumulation import GradientAccumulationTrainer
from phase5_deployment.memory_optimizer import enable_gradient_checkpointing

# Effective batch size = 32 with only 4 samples in VRAM
trainer = GradientAccumulationTrainer(model, accumulation_steps=8)

# Enable gradient checkpointing for 50% memory savings
enable_gradient_checkpointing(model)
```

### Mixed Precision Training
```python
from phase4_advanced.mixed_precision import MixedPrecisionTrainer

trainer = MixedPrecisionTrainer(model, optimizer, precision="fp16")
trainer.train(dataloader, epochs=10)
```

### Model Compression Pipeline
```python
from phase5_deployment.pruning import MagnitudePruner
from phase5_deployment.quantization import StaticQuantizer
from phase5_deployment.distillation import KnowledgeDistiller

# Prune 50% weights
pruner = MagnitudePruner(sparsity=0.5)
pruner.prune(model)

# Quantize to INT8
quantizer = StaticQuantizer()
quantized_model = quantizer.quantize(model)

# Distill to smaller model
distiller = KnowledgeDistiller(teacher=large_model)
distiller.train(student_model, dataloader)
```

---

## Test Coverage

| Module | Tests | Status |
|--------|-------|--------|
| Activations | 50 | Passing |
| MLP | 31 | Passing |
| Loss | 44 | Passing |
| Optimizer | 25 | Passing |
| CNN | 41 | Passing |
| RNN/LSTM/GRU | 28 | Passing |
| Attention | 22 | Passing |
| Normalization | 30 | Passing |
| Dropout | 24 | Passing |
| LR Scheduler | 35 | Passing |
| Mixed Precision | 27 | Passing |
| DDP Training | 33 | Passing |
| Pruning | 43 | Passing |
| Quantization | 30 | Passing |
| ONNX | 32 | Passing |
| Distillation | 35 | Passing |
| Checkpoint | 22 | Passing |
| Gradient Accumulation | 37 | Passing |
| Early Stopping | 31 | Passing |

---

## References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Deep Learning Book](https://www.deeplearningbook.org/)
- [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/)
- [ONNX Runtime](https://onnxruntime.ai/)

---

## License

MIT License
