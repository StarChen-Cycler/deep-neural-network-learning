"""
Learning rate schedulers for neural network training.

This module provides advanced learning rate scheduling strategies:
- StepLR: Decay by gamma every step_size epochs
- ExponentialLR: Exponential decay
- CosineAnnealingLR: Cosine annealing schedule
- LinearWarmup: Linear warmup from 0 to base_lr
- CosineWarmup: Cosine warmup from 0 to base_lr
- CyclicLR: Triangular/policy-based cycling
- OneCycleLR: Single cycle policy (warmup + decay)
- ReduceLROnPlateau: Reduce when metric plateaus
- WarmupDecayScheduler: Warmup + any decay scheduler

References:
    - SGDR: Stochastic Gradient Descent with Warm Restarts
      (Loshchilov & Hutter, 2017): https://arxiv.org/abs/1608.03983
    - Cyclical Learning Rates for Training Neural Networks
      (Smith, 2017): https://arxiv.org/abs/1506.01186
    - Super-Convergence: Very Fast Training of Neural Networks
      (Smith & Topin, 2018): https://arxiv.org/abs/1708.07120
"""

from typing import Optional, List, Union, Callable, Any
import numpy as np

# Type alias
ArrayLike = np.ndarray


class LRSchedulerBase:
    """Base class for learning rate schedulers."""

    def __init__(self, base_lr: float = 0.01):
        """
        Initialize scheduler.

        Args:
            base_lr: Base learning rate
        """
        self.base_lr = base_lr
        self.current_lr = base_lr
        self.step_count = 0
        self._lr_history: List[float] = []

    def step(self) -> float:
        """
        Update learning rate and return current value.

        Returns:
            Current learning rate
        """
        self.step_count += 1
        self._lr_history.append(self.current_lr)
        return self.current_lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.current_lr

    def reset(self) -> None:
        """Reset scheduler to initial state."""
        self.step_count = 0
        self.current_lr = self.base_lr
        self._lr_history = []

    def get_history(self) -> List[float]:
        """Get learning rate history."""
        return self._lr_history.copy()


class StepLR(LRSchedulerBase):
    """
    Step learning rate scheduler.

    Decays learning rate by gamma every step_size epochs.

    Formula: lr = base_lr * gamma^(step // step_size)

    Args:
        base_lr: Initial learning rate
        step_size: Period of learning rate decay
        gamma: Multiplicative factor of learning rate decay

    Example:
        >>> scheduler = StepLR(base_lr=0.1, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        ...     lr = scheduler.step()
        ...     # At epoch 30, lr = 0.01
        ...     # At epoch 60, lr = 0.001
    """

    def __init__(self, base_lr: float = 0.01, step_size: int = 30, gamma: float = 0.1):
        super().__init__(base_lr)
        self.step_size = step_size
        self.gamma = gamma

    def step(self) -> float:
        """Update learning rate."""
        self.step_count += 1
        self.current_lr = self.base_lr * (self.gamma ** (self.step_count // self.step_size))
        self._lr_history.append(self.current_lr)
        return self.current_lr


class ExponentialLR(LRSchedulerBase):
    """
    Exponential learning rate scheduler.

    Formula: lr = base_lr * gamma^step

    Args:
        base_lr: Initial learning rate
        gamma: Multiplicative factor of learning rate decay

    Example:
        >>> scheduler = ExponentialLR(base_lr=0.1, gamma=0.99)
        >>> for epoch in range(100):
        ...     lr = scheduler.step()  # lr = 0.1 * 0.99^epoch
    """

    def __init__(self, base_lr: float = 0.01, gamma: float = 0.99):
        super().__init__(base_lr)
        self.gamma = gamma

    def step(self) -> float:
        """Update learning rate."""
        self.step_count += 1
        self.current_lr = self.base_lr * (self.gamma**self.step_count)
        self._lr_history.append(self.current_lr)
        return self.current_lr


class CosineAnnealingLR(LRSchedulerBase):
    """
    Cosine annealing learning rate scheduler.

    Formula: lr = η_min + (base_lr - η_min) * (1 + cos(π * step / T_max)) / 2

    Args:
        base_lr: Initial learning rate
        T_max: Maximum number of iterations (half cycle)
        eta_min: Minimum learning rate (default: 0)

    Example:
        >>> scheduler = CosineAnnealingLR(base_lr=0.1, T_max=100, eta_min=0)
        >>> for epoch in range(100):
        ...     lr = scheduler.step()
        ...     # At epoch 0: lr ≈ 0.1
        ...     # At epoch 50: lr ≈ 0.05
        ...     # At epoch 100: lr ≈ 0
    """

    def __init__(self, base_lr: float = 0.01, T_max: int = 100, eta_min: float = 0.0):
        super().__init__(base_lr)
        self.T_max = T_max
        self.eta_min = eta_min

    def step(self) -> float:
        """Update learning rate."""
        self.step_count += 1
        self.current_lr = self.eta_min + (self.base_lr - self.eta_min) * (
            1 + np.cos(np.pi * self.step_count / self.T_max)
        ) / 2
        self._lr_history.append(self.current_lr)
        return self.current_lr


class LinearWarmup(LRSchedulerBase):
    """
    Linear warmup scheduler.

    Linearly increases learning rate from start_lr to base_lr over warmup_steps.

    Formula: lr = start_lr + (base_lr - start_lr) * step / warmup_steps

    Args:
        base_lr: Target learning rate after warmup
        warmup_steps: Number of warmup steps
        start_lr: Starting learning rate (default: 0)

    Example:
        >>> scheduler = LinearWarmup(base_lr=0.1, warmup_steps=1000)
        >>> for step in range(1000):
        ...     lr = scheduler.step()
        ...     # lr goes from 0 to 0.1 linearly
    """

    def __init__(self, base_lr: float = 0.01, warmup_steps: int = 1000, start_lr: float = 0.0):
        super().__init__(base_lr)
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.current_lr = start_lr

    def step(self) -> float:
        """Update learning rate."""
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            self.current_lr = self.start_lr + (self.base_lr - self.start_lr) * (
                self.step_count / self.warmup_steps
            )
        self._lr_history.append(self.current_lr)
        return self.current_lr

    def is_warmup_complete(self) -> bool:
        """Check if warmup phase is complete."""
        return self.step_count >= self.warmup_steps


class CosineWarmup(LRSchedulerBase):
    """
    Cosine warmup scheduler.

    Uses cosine curve to increase learning rate from start_lr to base_lr.

    Formula: lr = start_lr + (base_lr - start_lr) * (1 - cos(π * step / warmup_steps)) / 2

    Args:
        base_lr: Target learning rate after warmup
        warmup_steps: Number of warmup steps
        start_lr: Starting learning rate (default: 0)
    """

    def __init__(self, base_lr: float = 0.01, warmup_steps: int = 1000, start_lr: float = 0.0):
        super().__init__(base_lr)
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.current_lr = start_lr

    def step(self) -> float:
        """Update learning rate."""
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            self.current_lr = self.start_lr + (self.base_lr - self.start_lr) * (
                1 - np.cos(np.pi * self.step_count / self.warmup_steps)
            ) / 2
        self._lr_history.append(self.current_lr)
        return self.current_lr


class CyclicLR(LRSchedulerBase):
    """
    Cyclic learning rate scheduler.

    Oscillates learning rate between base_lr and max_lr using triangular policy.

    Formula (triangular):
        cycle = floor(1 + step / (2 * step_size))
        x = abs(step / step_size - 2 * cycle + 1)
        lr = base_lr + (max_lr - base_lr) * max(0, 1 - x)

    Args:
        base_lr: Lower learning rate bound
        max_lr: Upper learning rate bound
        step_size: Number of iterations per half cycle
        mode: 'triangular', 'triangular2', or 'exp_range'
        gamma: Factor for 'exp_range' mode

    References:
        - Cyclical Learning Rates (Smith, 2017)

    Example:
        >>> scheduler = CyclicLR(base_lr=0.001, max_lr=0.01, step_size=100)
        >>> for step in range(500):
        ...     lr = scheduler.step()  # Oscillates between 0.001 and 0.01
    """

    def __init__(
        self,
        base_lr: float = 0.001,
        max_lr: float = 0.01,
        step_size: int = 100,
        mode: str = "triangular",
        gamma: float = 1.0,
    ):
        super().__init__(base_lr)
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma

        if mode not in ["triangular", "triangular2", "exp_range"]:
            raise ValueError(f"Unknown mode: {mode}")

    def step(self) -> float:
        """Update learning rate."""
        self.step_count += 1

        cycle = np.floor(1 + self.step_count / (2 * self.step_size))
        x = np.abs(self.step_count / self.step_size - 2 * cycle + 1)

        if self.mode == "triangular":
            scale = 1.0
        elif self.mode == "triangular2":
            scale = 1.0 / (2.0 ** (cycle - 1))
        else:  # exp_range
            scale = self.gamma ** self.step_count

        self.current_lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) * scale
        self._lr_history.append(self.current_lr)
        return self.current_lr


class OneCycleLR(LRSchedulerBase):
    """
    One Cycle Learning Rate Scheduler.

    Single cycle policy: warmup from initial_lr to max_lr, then decay to min_lr.
    Achieves "super-convergence" with much fewer epochs.

    Phases:
        1. Warmup: initial_lr → max_lr (pct_start of total_steps)
        2. Decay: max_lr → min_lr (remaining steps)

    Formula:
        Warmup: lr = initial_lr + (max_lr - initial_lr) * step / (pct_start * total_steps)
        Decay: lr = max_lr - (max_lr - min_lr) * (step - warmup_steps) / (total_steps - warmup_steps)

    Args:
        max_lr: Maximum learning rate (peak of cycle)
        total_steps: Total number of training steps
        pct_start: Percentage of total_steps for warmup (default: 0.3)
        initial_lr: Starting learning rate (default: max_lr/25)
        min_lr: Final learning rate (default: max_lr/10000)
        final_div_factor: Determines min_lr via min_lr = max_lr / final_div_factor

    References:
        - Super-Convergence (Smith & Topin, 2018)

    Example:
        >>> scheduler = OneCycleLR(max_lr=0.01, total_steps=1000)
        >>> for step in range(1000):
        ...     lr = scheduler.step()
        ...     # Warmup: 0.0004 → 0.01 (steps 0-300)
        ...     # Decay: 0.01 → 0.000001 (steps 300-1000)
    """

    def __init__(
        self,
        max_lr: float = 0.01,
        total_steps: int = 1000,
        pct_start: float = 0.3,
        initial_lr: Optional[float] = None,
        min_lr: Optional[float] = None,
        final_div_factor: float = 1e4,
    ):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.initial_lr = initial_lr if initial_lr is not None else max_lr / 25
        self.min_lr = min_lr if min_lr is not None else max_lr / final_div_factor

        super().__init__(self.initial_lr)

        self.warmup_steps = int(total_steps * pct_start)
        self._validate()

    def _validate(self) -> None:
        """Validate parameters."""
        if self.pct_start < 0 or self.pct_start > 1:
            raise ValueError(f"pct_start must be in [0, 1], got {self.pct_start}")
        if self.total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {self.total_steps}")
        if self.max_lr <= 0:
            raise ValueError(f"max_lr must be positive, got {self.max_lr}")

    def step(self) -> float:
        """Update learning rate."""
        self.step_count += 1

        if self.step_count <= self.warmup_steps:
            # Warmup phase: linear increase
            pct = self.step_count / self.warmup_steps
            self.current_lr = self.initial_lr + (self.max_lr - self.initial_lr) * pct
        else:
            # Decay phase: linear decrease
            pct = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            self.current_lr = self.max_lr - (self.max_lr - self.min_lr) * pct

        self._lr_history.append(self.current_lr)
        return self.current_lr


class ReduceLROnPlateau(LRSchedulerBase):
    """
    Reduce learning rate when a metric has stopped improving.

    Monitors a metric and reduces learning rate by factor when no improvement
    is seen for patience number of epochs.

    Args:
        base_lr: Initial learning rate
        mode: 'min' or 'max' - whether lower/higher metric is better
        factor: Factor to reduce lr by (default: 0.1)
        patience: Number of epochs with no improvement (default: 10)
        threshold: Threshold for measuring improvement (default: 1e-4)
        min_lr: Minimum learning rate (default: 0)

    Example:
        >>> scheduler = ReduceLROnPlateau(base_lr=0.1, mode='min')
        >>> for epoch in range(100):
        ...     val_loss = validate()
        ...     lr = scheduler.step(val_loss)
    """

    def __init__(
        self,
        base_lr: float = 0.01,
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        min_lr: float = 0.0,
    ):
        super().__init__(base_lr)
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr

        self.best_metric: Optional[float] = None
        self.bad_epochs = 0

        if mode not in ["min", "max"]:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

    def step(self, metric: Optional[float] = None) -> float:
        """
        Update learning rate based on metric.

        Args:
            metric: Metric to monitor (optional)

        Returns:
            Current learning rate
        """
        self.step_count += 1

        if metric is not None:
            if self.best_metric is None:
                self.best_metric = metric
            else:
                improved = (
                    (metric < self.best_metric - self.threshold)
                    if self.mode == "min"
                    else (metric > self.best_metric + self.threshold)
                )

                if improved:
                    self.best_metric = metric
                    self.bad_epochs = 0
                else:
                    self.bad_epochs += 1

                if self.bad_epochs >= self.patience:
                    self.current_lr = max(self.current_lr * self.factor, self.min_lr)
                    self.bad_epochs = 0

        self._lr_history.append(self.current_lr)
        return self.current_lr

    def reset(self) -> None:
        """Reset scheduler."""
        super().reset()
        self.best_metric = None
        self.bad_epochs = 0


class WarmupDecayScheduler(LRSchedulerBase):
    """
    Combined warmup + decay scheduler.

    Applies linear warmup followed by any decay scheduler.

    Args:
        warmup_steps: Number of warmup steps
        warmup_start_lr: Starting learning rate for warmup
        decay_scheduler: Scheduler to use after warmup

    Example:
        >>> decay = CosineAnnealingLR(base_lr=0.1, T_max=100)
        >>> scheduler = WarmupDecayScheduler(
        ...     warmup_steps=10, warmup_start_lr=0, decay_scheduler=decay
        ... )
    """

    def __init__(
        self,
        warmup_steps: int,
        warmup_start_lr: float,
        decay_scheduler: LRSchedulerBase,
    ):
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.decay_scheduler = decay_scheduler

        super().__init__(decay_scheduler.base_lr)
        self.current_lr = warmup_start_lr

    def step(self) -> float:
        """Update learning rate."""
        self.step_count += 1

        if self.step_count <= self.warmup_steps:
            # Linear warmup
            self.current_lr = self.warmup_start_lr + (
                self.base_lr - self.warmup_start_lr
            ) * (self.step_count / self.warmup_steps)
        else:
            # Use decay scheduler
            self.decay_scheduler.step()
            self.current_lr = self.decay_scheduler.get_lr()

        self._lr_history.append(self.current_lr)
        return self.current_lr


class CosineAnnealingWarmRestarts(LRSchedulerBase):
    """
    Cosine annealing with warm restarts (SGDR).

    Applies cosine annealing with periodic restarts. Each restart starts
    from the base learning rate but with a longer period.

    Formula:
        T_cur = T_0 * (T_mult)^(n) where n is number of restarts
        lr = η_min + (base_lr - η_min) * (1 + cos(π * T_cur / T_i)) / 2

    Args:
        base_lr: Initial learning rate
        T_0: Period of first restart
        T_mult: Multiplier for period increase after restart
        eta_min: Minimum learning rate

    References:
        - SGDR (Loshchilov & Hutter, 2017)

    Example:
        >>> scheduler = CosineAnnealingWarmRestarts(base_lr=0.1, T_0=10, T_mult=2)
        >>> for epoch in range(100):
        ...     lr = scheduler.step()
        ...     # Restart at epoch 10, 30, 70 (periods: 10, 20, 40)
    """

    def __init__(
        self,
        base_lr: float = 0.01,
        T_0: int = 10,
        T_mult: int = 2,
        eta_min: float = 0.0,
    ):
        super().__init__(base_lr)
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min

        self._T_cur = 0
        self._T_i = T_0
        self._restart_count = 0

    def step(self) -> float:
        """Update learning rate."""
        self.step_count += 1
        self._T_cur += 1

        # Check for restart
        if self._T_cur >= self._T_i:
            self._T_cur = 0
            self._T_i *= self.T_mult
            self._restart_count += 1

        self.current_lr = self.eta_min + (self.base_lr - self.eta_min) * (
            1 + np.cos(np.pi * self._T_cur / self._T_i)
        ) / 2

        self._lr_history.append(self.current_lr)
        return self.current_lr

    def reset(self) -> None:
        """Reset scheduler."""
        super().reset()
        self._T_cur = 0
        self._T_i = self.T_0
        self._restart_count = 0


class PolynomialLR(LRSchedulerBase):
    """
    Polynomial learning rate decay.

    Formula: lr = (base_lr - min_lr) * (1 - step / total_steps)^power + min_lr

    Args:
        base_lr: Initial learning rate
        total_steps: Total number of steps
        power: Power of polynomial (default: 1.0 for linear)
        min_lr: Minimum learning rate (default: 0)
    """

    def __init__(
        self,
        base_lr: float = 0.01,
        total_steps: int = 1000,
        power: float = 1.0,
        min_lr: float = 0.0,
    ):
        super().__init__(base_lr)
        self.total_steps = total_steps
        self.power = power
        self.min_lr = min_lr

    def step(self) -> float:
        """Update learning rate."""
        self.step_count += 1
        factor = (1 - self.step_count / self.total_steps) ** self.power
        self.current_lr = (self.base_lr - self.min_lr) * factor + self.min_lr
        self._lr_history.append(self.current_lr)
        return self.current_lr


def get_scheduler(name: str, **kwargs) -> LRSchedulerBase:
    """
    Get scheduler by name.

    Args:
        name: Name of scheduler
        **kwargs: Arguments to pass to constructor

    Returns:
        Scheduler instance

    Raises:
        ValueError: If unknown scheduler name
    """
    schedulers = {
        "step": StepLR,
        "exponential": ExponentialLR,
        "cosine": CosineAnnealingLR,
        "cosine_annealing": CosineAnnealingLR,
        "linear_warmup": LinearWarmup,
        "cosine_warmup": CosineWarmup,
        "cyclic": CyclicLR,
        "one_cycle": OneCycleLR,
        "onecycle": OneCycleLR,
        "reduce_on_plateau": ReduceLROnPlateau,
        "warmup_decay": WarmupDecayScheduler,
        "cosine_restart": CosineAnnealingWarmRestarts,
        "polynomial": PolynomialLR,
    }

    name_lower = name.lower().replace("-", "_")
    if name_lower not in schedulers:
        available = ", ".join(schedulers.keys())
        raise ValueError(f"Unknown scheduler '{name}'. Available: {available}")

    return schedulers[name_lower](**kwargs)


def plot_learning_rate_curve(
    scheduler: LRSchedulerBase,
    steps: int,
    reset: bool = True,
) -> tuple:
    """
    Generate learning rate curve for visualization.

    Args:
        scheduler: Scheduler instance
        steps: Number of steps to simulate
        reset: Whether to reset scheduler before simulation

    Returns:
        Tuple of (steps_array, lr_array)
    """
    if reset:
        scheduler.reset()

    lrs = []
    for _ in range(steps):
        lr = scheduler.step()
        lrs.append(lr)

    return np.arange(1, steps + 1), np.array(lrs)


# Registry
LR_SCHEDULERS = {
    "StepLR": StepLR,
    "ExponentialLR": ExponentialLR,
    "CosineAnnealingLR": CosineAnnealingLR,
    "LinearWarmup": LinearWarmup,
    "CosineWarmup": CosineWarmup,
    "CyclicLR": CyclicLR,
    "OneCycleLR": OneCycleLR,
    "ReduceLROnPlateau": ReduceLROnPlateau,
    "WarmupDecayScheduler": WarmupDecayScheduler,
    "CosineAnnealingWarmRestarts": CosineAnnealingWarmRestarts,
    "PolynomialLR": PolynomialLR,
}
