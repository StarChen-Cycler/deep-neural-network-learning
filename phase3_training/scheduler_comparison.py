"""
Learning Rate Scheduler Comparison Experiment.

Compares different LR scheduling strategies on a simple optimization task
to demonstrate convergence behavior.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
import json

from .lr_scheduler import (
    StepLR,
    ExponentialLR,
    CosineAnnealingLR,
    LinearWarmup,
    CyclicLR,
    OneCycleLR,
    PolynomialLR,
    CosineAnnealingWarmRestarts,
    get_scheduler,
    plot_learning_rate_curve,
)


def quadratic_function(x: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Simple quadratic function for testing optimizers.

    f(x) = x^T A x - b^T x
    grad = 2Ax - b

    Minimum at x = A^(-1) * b / 2
    """
    # Create positive definite A
    A = np.array([[10.0, 2.0], [2.0, 5.0]])
    b = np.array([1.0, 3.0])

    loss = x @ A @ x - b @ x
    grad = 2 * A @ x - b
    return loss, grad


def rosenbrock_function(x: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Rosenbrock function - classic optimization benchmark.

    f(x, y) = (a - x)^2 + b(y - x^2)^2
    Minimum at (a, a^2) = (1, 1)
    """
    a, b = 1.0, 100.0

    loss = (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2

    grad = np.array(
        [
            -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0] ** 2),
            2 * b * (x[1] - x[0] ** 2),
        ]
    )
    return loss, grad


def run_optimization(
    scheduler: Any,
    obj_fn: callable,
    x_init: np.ndarray,
    max_steps: int = 500,
    grad_clip: float = 1.0,
) -> Dict[str, Any]:
    """
    Run gradient descent optimization with a given scheduler.

    Args:
        scheduler: Learning rate scheduler
        obj_fn: Objective function returning (loss, gradient)
        x_init: Initial point
        max_steps: Maximum optimization steps
        grad_clip: Gradient clipping threshold

    Returns:
        Dictionary with optimization history
    """
    x = x_init.copy()
    losses = []
    lrs = []
    xs = []

    for step in range(max_steps):
        # Get learning rate
        lr = scheduler.get_lr()

        # Compute loss and gradient
        loss, grad = obj_fn(x)

        # Gradient clipping
        grad_norm = np.linalg.norm(grad)
        if grad_norm > grad_clip:
            grad = grad * grad_clip / grad_norm

        # Gradient descent step
        x = x - lr * grad

        # Update scheduler
        scheduler.step()

        # Record history
        losses.append(loss)
        lrs.append(lr)
        xs.append(x.copy())

    return {
        "losses": np.array(losses),
        "lrs": np.array(lrs),
        "xs": np.array(xs),
        "final_loss": losses[-1],
        "final_x": x,
        "converged": losses[-1] < 1e-6,
    }


def compare_schedulers(
    base_lr: float = 0.1,
    max_steps: int = 200,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Compare different LR schedulers on optimization task.

    Args:
        base_lr: Base learning rate for all schedulers
        max_steps: Number of optimization steps
        seed: Random seed

    Returns:
        Dictionary with comparison results
    """
    np.random.seed(seed)
    x_init = np.array([-2.0, 2.0])  # Start away from minimum

    # Define schedulers to compare
    schedulers = {
        "Constant": lambda: get_scheduler("step", base_lr=base_lr, step_size=max_steps * 2, gamma=1.0),
        "StepLR": lambda: StepLR(base_lr=base_lr, step_size=50, gamma=0.5),
        "Exponential": lambda: ExponentialLR(base_lr=base_lr, gamma=0.98),
        "CosineAnnealing": lambda: CosineAnnealingLR(base_lr=base_lr, T_max=max_steps),
        "Cyclic": lambda: CyclicLR(base_lr=base_lr * 0.1, max_lr=base_lr, step_size=25),
        "OneCycle": lambda: OneCycleLR(max_lr=base_lr, total_steps=max_steps),
        "Polynomial": lambda: PolynomialLR(base_lr=base_lr, total_steps=max_steps, power=2.0),
        "CosineRestart": lambda: CosineAnnealingWarmRestarts(base_lr=base_lr, T_0=40, T_mult=1),
    }

    results = {}
    for name, scheduler_fn in schedulers.items():
        scheduler = scheduler_fn()
        result = run_optimization(
            scheduler=scheduler,
            obj_fn=rosenbrock_function,
            x_init=x_init.copy(),
            max_steps=max_steps,
        )
        result["scheduler_name"] = name
        results[name] = result

    return results


def compute_convergence_metrics(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Compute convergence metrics for each scheduler.

    Args:
        results: Results from compare_schedulers

    Returns:
        List of metric dictionaries
    """
    metrics = []

    for name, data in results.items():
        losses = data["losses"]

        # Find step to reach 90% of improvement
        initial_loss = losses[0]
        final_loss = losses[-1]
        target_90 = initial_loss - 0.9 * (initial_loss - final_loss)

        steps_to_90 = None
        for i, loss in enumerate(losses):
            if loss <= target_90:
                steps_to_90 = i + 1
                break

        metrics.append(
            {
                "scheduler": name,
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "improvement": initial_loss - final_loss,
                "steps_to_90": steps_to_90,
                "converged": data["converged"],
            }
        )

    # Sort by final loss
    metrics.sort(key=lambda x: x["final_loss"])

    return metrics


def generate_lr_curves(
    base_lr: float = 0.1,
    max_steps: int = 200,
) -> Dict[str, np.ndarray]:
    """
    Generate learning rate curves for visualization.

    Args:
        base_lr: Base learning rate
        max_steps: Number of steps

    Returns:
        Dictionary mapping scheduler names to LR arrays
    """
    schedulers = {
        "Constant": StepLR(base_lr=base_lr, step_size=max_steps * 2, gamma=1.0),
        "StepLR": StepLR(base_lr=base_lr, step_size=50, gamma=0.5),
        "Exponential": ExponentialLR(base_lr=base_lr, gamma=0.98),
        "CosineAnnealing": CosineAnnealingLR(base_lr=base_lr, T_max=max_steps),
        "Cyclic": CyclicLR(base_lr=base_lr * 0.1, max_lr=base_lr, step_size=25),
        "OneCycle": OneCycleLR(max_lr=base_lr, total_steps=max_steps),
        "Polynomial": PolynomialLR(base_lr=base_lr, total_steps=max_steps, power=2.0),
        "CosineRestart": CosineAnnealingWarmRestarts(base_lr=base_lr, T_0=40, T_mult=1),
    }

    curves = {}
    for name, scheduler in schedulers.items():
        _, lrs = plot_learning_rate_curve(scheduler, max_steps)
        curves[name] = lrs

    return curves


def save_comparison_results(
    results: Dict[str, Any],
    metrics: List[Dict[str, Any]],
    output_dir: str = "experiments/results",
) -> None:
    """
    Save comparison results to files.

    Args:
        results: Optimization results
        metrics: Convergence metrics
        output_dir: Directory to save results
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Save metrics
    metrics_path = os.path.join(output_dir, "scheduler_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save LR curves
    curves = generate_lr_curves()
    curves_path = os.path.join(output_dir, "lr_curves.json")
    curves_serializable = {k: v.tolist() for k, v in curves.items()}
    with open(curves_path, "w") as f:
        json.dump(curves_serializable, f, indent=2)

    # Save loss curves
    losses = {}
    for name, data in results.items():
        losses[name] = data["losses"].tolist()

    losses_path = os.path.join(output_dir, "scheduler_losses.json")
    with open(losses_path, "w") as f:
        json.dump(losses, f, indent=2)


def run_experiment():
    """Run the full comparison experiment."""
    print("=" * 60)
    print("Learning Rate Scheduler Comparison Experiment")
    print("=" * 60)

    # Run comparison
    results = compare_schedulers(base_lr=0.1, max_steps=200)

    # Compute metrics
    metrics = compute_convergence_metrics(results)

    # Print results
    print("\nConvergence Metrics (sorted by final loss):")
    print("-" * 60)
    print(f"{'Scheduler':<15} {'Final Loss':>12} {'Improvement':>12} {'Steps 90%':>10}")
    print("-" * 60)

    for m in metrics:
        print(
            f"{m['scheduler']:<15} {m['final_loss']:>12.6f} "
            f"{m['improvement']:>12.6f} "
            f"{m['steps_to_90'] or 'N/A':>10}"
        )

    # Find best scheduler
    best = metrics[0]
    print(f"\nBest scheduler: {best['scheduler']} (final loss: {best['final_loss']:.6f})")

    # Verify success criteria
    print("\nVerifying Success Criteria:")
    print("-" * 40)

    # StepLR criterion
    steplr_scheduler = StepLR(base_lr=0.1, step_size=30, gamma=0.1)
    for _ in range(30):
        steplr_scheduler.step()
    lr_at_30 = steplr_scheduler.get_lr()
    steplr_pass = abs(lr_at_30 - 0.01) < 1e-6
    print(f"StepLR at step 30: lr = {lr_at_30:.6f} (expected 0.01) - {'PASS' if steplr_pass else 'FAIL'}")

    # CosineAnnealing criterion
    cosine_scheduler = CosineAnnealingLR(base_lr=0.1, T_max=100, eta_min=0)
    for _ in range(100):
        cosine_scheduler.step()
    lr_at_100 = cosine_scheduler.get_lr()
    cosine_pass = lr_at_100 < 0.001  # Should be close to 0
    print(f"CosineAnnealing at step 100: lr = {lr_at_100:.6f} (expected ~0) - {'PASS' if cosine_pass else 'FAIL'}")

    # OneCycleLR criterion
    onecycle_scheduler = OneCycleLR(max_lr=0.01, total_steps=1000)
    max_lr_seen = 0
    for _ in range(1000):
        onecycle_scheduler.step()
        max_lr_seen = max(max_lr_seen, onecycle_scheduler.get_lr())
    onecycle_pass = abs(max_lr_seen - 0.01) < 1e-6
    print(
        f"OneCycleLR max_lr: {max_lr_seen:.6f} (expected 0.01) - {'PASS' if onecycle_pass else 'FAIL'}"
    )

    return results, metrics


if __name__ == "__main__":
    run_experiment()
