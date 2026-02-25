# Optimizer Theory Notes

This document covers the mathematical derivations and theory behind gradient descent optimizers.

## Table of Contents

1. [Gradient Descent Fundamentals](#1-gradient-descent-fundamentals)
2. [SGD (Stochastic Gradient Descent)](#2-sgd-stochastic-gradient-descent)
3. [Momentum](#3-momentum)
4. [Nesterov Accelerated Gradient](#4-nesterov-accelerated-gradient)
5. [AdaGrad](#5-adagrad)
6. [RMSprop](#6-rmsprop)
7. [Adam](#7-adam)
8. [AdamW](#8-adamw)
9. [Learning Rate Scheduling](#9-learning-rate-scheduling)
10. [Practical Recommendations](#10-practical-recommendations)

---

## 1. Gradient Descent Fundamentals

### Objective

Given a loss function $L(\theta)$ parameterized by $\theta$, find parameters that minimize the loss:

$$\theta^* = \arg\min_\theta L(\theta)$$

### Update Rule

The basic gradient descent update:

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

where:
- $\eta$ is the learning rate
- $\nabla L(\theta_t)$ is the gradient at current parameters

### Convergence Conditions

For convex functions, gradient descent converges when:
- Learning rate $\eta < \frac{2}{\lambda_{max}}$ where $\lambda_{max}$ is the largest eigenvalue of the Hessian
- For non-convex functions, converges to a local minimum or saddle point

---

## 2. SGD (Stochastic Gradient Descent)

### Formula

$$\theta_{t+1} = \theta_t - \eta g_t$$

where $g_t = \nabla L(\theta_t; x_i)$ is the gradient computed on a single sample or mini-batch.

### Properties

- **Pros**: Simple, low memory, works well with large datasets
- **Cons**: Noisy gradients, slow convergence, sensitive to learning rate

### Learning Rate Guidelines

| Dataset Size | Typical Learning Rate |
|--------------|----------------------|
| Small (<10K) | 0.01 - 0.1 |
| Medium (10K-1M) | 0.001 - 0.01 |
| Large (>1M) | 0.0001 - 0.001 |

---

## 3. Momentum

### Formula

$$v_t = \gamma v_{t-1} - \eta g_t$$
$$\theta_{t+1} = \theta_t + v_t$$

where:
- $v_t$ is the velocity
- $\gamma$ is the momentum coefficient (typically 0.9)

### Interpretation

Momentum accumulates gradient history:
- Accelerates in directions of persistent gradient (consistent descent)
- Dampens oscillations in directions with alternating gradients

### Derivation

The velocity update can be rewritten as an exponential moving average of gradients:

$$v_t = -\eta \sum_{i=0}^{t} \gamma^{t-i} g_i$$

This shows that older gradients are weighted exponentially less.

### Convergence Speed

On quadratic functions with condition number $\kappa$:
- SGD: $O(\kappa \log(1/\epsilon))$
- Momentum: $O(\sqrt{\kappa} \log(1/\epsilon))$

For ill-conditioned problems ($\kappa >> 1$), momentum provides significant speedup.

---

## 4. Nesterov Accelerated Gradient (NAG)

### Formula

$$v_t = \gamma v_{t-1} - \eta \nabla L(\theta_t + \gamma v_{t-1})$$
$$\theta_{t+1} = \theta_t + v_t$$

### Key Difference from Momentum

Nesterov computes the gradient at the **lookahead position** $\theta_t + \gamma v_{t-1}$ rather than at $\theta_t$.

### Equivalent Formulation

For practical implementation (gradient at current position):

$$v_t = \gamma v_{t-1} - \eta g_t$$
$$\theta_{t+1} = \theta_t - \gamma v_{t-1} + (1 + \gamma) v_t$$

### Theoretical Guarantee

For convex functions, NAG achieves optimal convergence rate:
$$O(1/t^2)$$ vs $O(1/t)$ for standard gradient descent.

---

## 5. AdaGrad

### Formula

$$G_t = G_{t-1} + g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t} + \epsilon} g_t$$

where $G_t$ accumulates squared gradients (element-wise).

### Adaptive Learning Rate

Effective learning rate for parameter $i$:
$$\eta_{eff,i} = \frac{\eta}{\sqrt{\sum_{j=1}^{t} g_{j,i}^2} + \epsilon}$$

### Properties

- **Pros**: No need to tune learning rate per parameter, good for sparse data
- **Cons**: Monotonically decreasing learning rate, can stop learning

### Best Use Cases

- NLP with sparse word embeddings
- Recommendation systems
- Any problem with infrequent features

---

## 6. RMSprop

### Formula

$$E[g^2]_t = \beta E[g^2]_{t-1} + (1-\beta) g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t} + \epsilon} g_t$$

### Key Improvement over AdaGrad

Uses exponential moving average instead of cumulative sum:
- Learning rate doesn't monotonically decrease
- Adapts to non-stationary objectives

### Hyperparameters

| Parameter | Default | Range |
|-----------|---------|-------|
| $\eta$ (lr) | 0.001 | 0.0001 - 0.01 |
| $\beta$ | 0.9 | 0.9 - 0.99 |
| $\epsilon$ | 1e-8 | 1e-8 - 1e-6 |

### Best Use Cases

- RNNs and LSTMs
- Non-stationary objectives
- Online learning

---

## 7. Adam

### Formula

**First moment (mean of gradients):**
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$

**Second moment (uncentered variance):**
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

**Bias correction:**
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**Update:**
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

### Why Bias Correction?

In early steps, $m_t$ and $v_t$ are biased toward zero (initialized at 0). The correction ensures proper scaling:
- $E[\hat{m}_t] \approx E[g]$ after correction
- Essential for good performance early in training

### Hyperparameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| $\eta$ | 0.001 | Learning rate |
| $\beta_1$ | 0.9 | Momentum decay |
| $\beta_2$ | 0.999 | RMSprop decay |
| $\epsilon$ | 1e-8 | Numerical stability |

### Properties

- Combines momentum (first moment) with RMSprop (second moment)
- Invariant to gradient scale
- Works well out-of-the-box for most problems

---

## 8. AdamW

### Problem with Adam + L2 Regularization

Standard Adam with L2 regularization:
$$g_t = \nabla L(\theta_t) + \lambda \theta_t$$

This couples weight decay with the adaptive learning rate, which is suboptimal.

### AdamW Formula

Decouple weight decay from gradient:
$$\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$

### Why Better?

- Weight decay is applied uniformly regardless of gradient history
- Better generalization
- Standard choice for Transformer training

### Hyperparameters

| Parameter | Default | Range |
|-----------|---------|-------|
| $\eta$ | 0.001 | Same as Adam |
| $\lambda$ | 0.01 | 0.01 - 0.1 |

---

## 9. Learning Rate Scheduling

### Step Decay

$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t / T \rfloor}$$

Every $T$ epochs, multiply learning rate by $\gamma$.

### Exponential Decay

$$\eta_t = \eta_0 \cdot \gamma^t$$

Continuous decay at rate $\gamma$ per epoch.

### Cosine Annealing

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t}{T_{max}}\pi\right)\right)$$

Smoothly decreases learning rate following a cosine curve.

### Warmup

For the first $T_{warmup}$ steps:
$$\eta_t = \eta_{base} \cdot \frac{t}{T_{warmup}}$$

Essential for Transformer training to prevent early instability.

---

## 10. Practical Recommendations

### By Task Type

| Task | Optimizer | LR | Notes |
|------|-----------|-----|-------|
| CNN Image Classification | SGD + Momentum | 0.1 | Use step decay |
| NLP / Transformers | AdamW | 1e-4 | Use warmup |
| RNN / LSTM | RMSprop | 1e-3 | Good for sequences |
| GANs | Adam | 2e-4 | Lower β₁=0.5 |
| Reinforcement Learning | Adam | 1e-4 | Often needs tuning |
| Meta-learning | Adam | 1e-3 | Task-dependent |

### Hyperparameter Tuning Priority

1. **Learning rate** - Most important, tune first
2. **Batch size** - Affects optimal learning rate
3. **Weight decay** - Regularization strength
4. **Momentum β** - Usually 0.9 works well
5. **Adam betas** - Rarely need to change

### Common Issues and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| Learning too slow | Loss decreases slowly | Increase learning rate |
| Divergence | Loss explodes | Decrease learning rate, add gradient clipping |
| Oscillation | Loss oscillates | Add momentum, decrease LR |
| Plateau | Loss stops decreasing | Use learning rate scheduler |
| Overfitting | Train loss << Val loss | Add weight decay, dropout |

### Debugging Checklist

1. **Start with Adam** at default LR (0.001)
2. **Monitor loss curves** - should decrease smoothly
3. **Check gradients** - should not be too large/small
4. **Try different LRs** - 10x smaller/larger
5. **Add LR scheduling** - cosine or step decay

---

## References

1. Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. arXiv:1412.6980
2. Nesterov, Y. (1983). A method for unconstrained convex minimization problem with the rate of convergence O(1/k²)
3. Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive Subgradient Methods for Online Learning and Stochastic Optimization
4. Loshchilov, I., & Hutter, F. (2017). Decoupled Weight Decay Regularization. arXiv:1711.05101
5. Ruder, S. (2016). An overview of gradient descent optimization algorithms. arXiv:1609.04747
