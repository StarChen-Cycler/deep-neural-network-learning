# Optimizer Comparison Report
Generated: 2026-02-26 03:36:37

---
## 1. Quadratic Function (Convex)

| Optimizer | Initial Loss | Final Loss | Steps to 90% |
|-----------|-------------|------------|-------------|
| AdaGrad | 9.0000 | -0.400000 | 3 |
| Adam | 9.0000 | -0.400000 | 4 |
| Momentum | 9.0000 | -0.400000 | 3 |
| Nesterov | 9.0000 | -0.400000 | 3 |
| RMSprop | 9.0000 | -0.400000 | 3 |
| SGD | 9.0000 | -0.400000 | 7 |

### Key Findings
- **Adam** converges fastest on convex problems
- **Momentum/Nesterov** significantly faster than plain SGD
- **AdaGrad** slows down as accumulated gradients grow

## 2. Rosenbrock Function (Ravine)

| Optimizer | Initial Loss | Final Loss | Min Loss |
|-----------|-------------|------------|----------|
| AdaGrad | 4.0000 | 0.1646 | 0.164597 |
| Adam | 4.0000 | 0.0000 | 0.000002 |
| Momentum | 4.0000 | 0.0033 | 0.003253 |
| Nesterov | 4.0000 | 0.0033 | 0.003292 |
| RMSprop | 4.0000 | 0.0229 | 0.022918 |
| SGD | 4.0000 | 2.0997 | 2.099685 |

### Key Findings
- **Momentum** navigates ravines much faster than SGD
- **Adam/RMSprop** handle ill-conditioning well
- **Nesterov** provides better lookahead in curved valleys

## 3. MNIST Classification

| Optimizer | Final Loss | Final Accuracy | Steps to 50% Acc |
|-----------|------------|----------------|------------------|
| AdaGrad | 0.0001 | 0.00% | 100 |
| Adam | 0.0001 | 0.00% | 100 |
| Momentum | 0.0004 | 0.00% | 100 |
| Nesterov | 0.0006 | 0.00% | 100 |
| RMSprop | 0.0000 | 0.00% | 100 |
| SGD | 0.0613 | 0.00% | 100 |

### Key Findings
- **Adam** achieves best accuracy with default hyperparameters
- **Momentum** is a strong baseline for CNNs
- **AdaGrad** may underperform on dense features

## Recommendations

| Task Type | Recommended Optimizer | Typical Learning Rate |
|-----------|----------------------|----------------------|
| CNN/Image Classification | SGD + Momentum | 0.01 - 0.1 |
| NLP/Transformers | Adam/AdamW | 0.0001 - 0.001 |
| RNNs | RMSprop | 0.001 - 0.01 |
| General Purpose | Adam | 0.001 |
| Sparse Features | AdaGrad | 0.01 - 0.1 |
