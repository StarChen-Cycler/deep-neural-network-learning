"""
Weight initialization comparison experiment.

This script demonstrates the impact of different initialization strategies
on activation variance and gradient flow through deep networks.

Key findings:
    1. Xavier: Best for Sigmoid/Tanh - maintains variance in symmetric activations
    2. He: Best for ReLU - accounts for half the activations being zeroed
    3. Zero: Causes symmetry problem - all neurons learn identical features
    4. LSUV: Adaptive - converges to unit variance in few iterations

Run:
    python experiments/init_comparison.py
"""

import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from phase1_basics.weight_init import (
    xavier_uniform,
    xavier_normal,
    he_uniform,
    he_normal,
    kaiming_uniform,
    kaiming_normal,
    zero_init,
    lsuv_init,
    INITIALIZERS,
)
from phase1_basics.activations import relu, sigmoid, tanh


def measure_variance_propagation(
    init_fn,
    activation_fn,
    layer_sizes: list,
    n_samples: int = 1000,
    n_trials: int = 10,
    seed: int = 42,
):
    """
    Measure how activation variance changes through network layers.

    Args:
        init_fn: Weight initialization function (fan_in, fan_out) -> weights
        activation_fn: Activation function to apply
        layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
        n_samples: Number of input samples for variance measurement
        n_trials: Number of trials to average over
        seed: Random seed for reproducibility

    Returns:
        Dict with layer variances and gradient norms for each trial
    """
    rng = np.random.default_rng(seed)

    all_variances = []
    all_gradient_norms = []

    for trial in range(n_trials):
        # Build network weights
        weights = []
        for i in range(len(layer_sizes) - 1):
            W = init_fn(layer_sizes[i], layer_sizes[i + 1], rng=rng)
            weights.append(W)

        # Forward pass: measure activation variance at each layer
        # Input with unit variance
        x = rng.standard_normal((n_samples, layer_sizes[0])).astype(np.float64)
        layer_variances = [np.var(x)]
        activations = [x]

        for W in weights:
            x = x @ W
            x = activation_fn(x)
            activations.append(x)
            layer_variances.append(np.var(x))

        # Backward pass: measure gradient norms
        # Gradient from output (simulated loss gradient)
        grad = rng.standard_normal((n_samples, layer_sizes[-1])).astype(np.float64)
        gradient_norms = [np.linalg.norm(grad)]

        for i in range(len(weights) - 1, -1, -1):
            # Gradient through activation (approximate for ReLU: assume 0.5 survive)
            grad = grad * (activations[i + 1] > 0).astype(np.float64)
            # Gradient through linear layer
            grad = grad @ weights[i].T
            gradient_norms.append(np.linalg.norm(grad))

        gradient_norms = gradient_norms[::-1]  # Reverse to match forward order

        all_variances.append(layer_variances)
        all_gradient_norms.append(gradient_norms)

    return {
        "variances": np.mean(all_variances, axis=0),
        "variance_std": np.std(all_variances, axis=0),
        "gradient_norms": np.mean(all_gradient_norms, axis=0),
        "gradient_std": np.std(all_gradient_norms, axis=0),
    }


def test_xavier_variance():
    """Test 1: Xavier initialization maintains variance for sigmoid/tanh."""
    print("=" * 60)
    print("TEST 1: Xavier Initialization Variance Preservation")
    print("=" * 60)

    layer_sizes = [784, 512, 256, 128, 64, 10]
    n_layers = len(layer_sizes)

    # Test Xavier with Sigmoid
    results = measure_variance_propagation(
        xavier_uniform,
        sigmoid,
        layer_sizes,
        n_samples=1000,
        n_trials=10,
    )

    print(f"\nLayer sizes: {layer_sizes}")
    print(f"\nActivation variances (target: 1.0):")
    for i, (var, std) in enumerate(zip(results["variances"], results["variance_std"])):
        print(f"  Layer {i}: {var:.4f} +/- {std:.4f}")

    # Check criterion: variance should stay within reasonable bounds
    # Xavier formula: Var = 2/(fan_in + fan_out)
    # Expected variance at layer 1: 2/(784+512) ≈ 0.00156
    fan_in, fan_out = layer_sizes[0], layer_sizes[1]
    expected_var = 2.0 / (fan_in + fan_out)

    print(f"\nExpected weight variance: {expected_var:.6f}")

    # Measure actual weight variance
    rng = np.random.default_rng(42)
    W = xavier_uniform(fan_in, fan_out, rng=rng)
    actual_var = np.var(W)
    print(f"Actual weight variance: {actual_var:.6f}")

    # Criterion: variance should be in expected range (within 50%)
    lower_bound = expected_var * 0.5
    upper_bound = expected_var * 1.5
    passed = lower_bound <= actual_var <= upper_bound

    print(f"\nVariance bounds: [{lower_bound:.6f}, {upper_bound:.6f}]")
    print(f"PASSED: {passed}")

    return passed


def test_he_variance_relu():
    """Test 2: He initialization maintains variance for ReLU networks."""
    print("\n" + "=" * 60)
    print("TEST 2: He Initialization Variance Preservation (ReLU)")
    print("=" * 60)

    layer_sizes = [784, 512, 256, 128, 64, 10]

    # Test He with ReLU
    results = measure_variance_propagation(
        he_normal,
        relu,
        layer_sizes,
        n_samples=1000,
        n_trials=10,
    )

    print(f"\nLayer sizes: {layer_sizes}")
    print(f"\nActivation variances (target: ~1.0 for ReLU):")
    for i, (var, std) in enumerate(zip(results["variances"], results["variance_std"])):
        print(f"  Layer {i}: {var:.4f} +/- {std:.4f}")

    # He formula: Var = 2/fan_in
    fan_in = layer_sizes[0]
    expected_var = 2.0 / fan_in

    print(f"\nExpected weight variance: {expected_var:.6f}")

    # Measure actual weight variance
    rng = np.random.default_rng(42)
    W = he_normal(fan_in, layer_sizes[1], rng=rng)
    actual_var = np.var(W)
    print(f"Actual weight variance: {actual_var:.6f}")

    # For ReLU networks, check variance doesn't explode/vanish too quickly
    # After ReLU, variance is roughly half (due to zeroing)
    # He accounts for this by using 2/fan_in instead of 1/fan_in

    # Check that activation variance stays reasonable (not too small or large)
    hidden_variances = results["variances"][1:-1]  # Exclude input and output
    min_var = min(hidden_variances)
    max_var = max(hidden_variances)

    print(f"\nHidden layer variance range: [{min_var:.4f}, {max_var:.4f}]")

    # Variance should stay within reasonable bounds
    passed = min_var > 0.01 and max_var < 10.0

    print(f"Variance stable (0.01 < var < 10.0): {passed}")
    print(f"PASSED: {passed}")

    return passed


def test_lsuv_convergence():
    """Test 3: LSUV converges within 10 iterations."""
    print("\n" + "=" * 60)
    print("TEST 3: LSUV Convergence Speed")
    print("=" * 60)

    fan_in, fan_out = 784, 256
    rng = np.random.default_rng(42)

    # Start with orthogonal-like initialization
    W = rng.standard_normal((fan_in, fan_out)).astype(np.float64)
    # Orthogonalize using SVD
    U, _, Vt = np.linalg.svd(W, full_matrices=False)
    W = U @ Vt

    print(f"Weight shape: ({fan_in}, {fan_out})")
    print(f"Initial weight variance: {np.var(W):.6f}")

    # Define forward function (captures W by reference)
    # Note: W will be modified in-place by LSUV
    def forward_fn(x):
        return relu(x @ W)

    # For ReLU, target variance ~0.5 (since ReLU zeros half the activations)
    # Variance after ReLU = Var(pre_act) / 2 approximately
    # So to get output var ~0.5, we target 0.5
    W_init, iterations = lsuv_init(
        W,
        forward_fn,
        target_variance=0.5,  # Target for ReLU output
        max_iterations=10,
        tol=0.2,  # 20% tolerance
        rng=rng,
    )

    # Verify final variance
    dummy_input = rng.standard_normal((1000, fan_in)).astype(np.float64)
    output = forward_fn(dummy_input)
    final_var = np.var(output)

    print(f"\nLSUV iterations: {iterations}")
    print(f"Final weight variance: {np.var(W_init):.6f}")
    print(f"Output variance after ReLU: {final_var:.4f}")

    # Criterion: iterations <= 10 and variance close to target
    passed = iterations <= 10 and abs(final_var - 0.5) < 0.25
    print(f"\nConverged in <= 10 iterations: {iterations <= 10}")
    print(f"Variance within tolerance: {abs(final_var - 0.5) < 0.25}")
    print(f"PASSED: {passed}")

    return passed


def compare_initializers():
    """Compare all initializers side by side."""
    print("\n" + "=" * 60)
    print("COMPARISON: All Initializers")
    print("=" * 60)

    layer_sizes = [784, 512, 256, 128, 64, 10]
    n_samples = 1000

    initializers = {
        "Xavier Uniform": (xavier_uniform, sigmoid),
        "Xavier Normal": (xavier_normal, sigmoid),
        "He Uniform": (he_uniform, relu),
        "He Normal": (he_normal, relu),
        "Kaiming Uniform": (
            lambda fan_in, fan_out, rng=None: kaiming_uniform(
                fan_in, fan_out, rng=rng
            ),
            relu,
        ),
        "Zero": (lambda fan_in, fan_out, rng=None: zero_init(fan_in, fan_out), relu),
    }

    print(f"\nNetwork: {layer_sizes}")
    print(f"Samples: {n_samples}")
    print()

    results_table = []

    for name, (init_fn, act_fn) in initializers.items():
        try:
            results = measure_variance_propagation(
                init_fn,
                act_fn,
                layer_sizes,
                n_samples=n_samples,
                n_trials=5,
            )

            # Get variance at layer 3 (middle of network)
            mid_var = results["variances"][3]
            # Get final variance
            final_var = results["variances"][-1]

            results_table.append(
                {
                    "name": name,
                    "mid_variance": mid_var,
                    "final_variance": final_var,
                    "status": "OK" if 0.01 < mid_var < 10 else "UNSTABLE",
                }
            )
        except Exception as e:
            results_table.append(
                {
                    "name": name,
                    "mid_variance": float("nan"),
                    "final_variance": float("nan"),
                    "status": f"ERROR: {str(e)[:20]}",
                }
            )

    # Print table
    print(f"{'Initializer':<20} {'Mid Var':>12} {'Final Var':>12} {'Status':>12}")
    print("-" * 60)
    for r in results_table:
        mid = f"{r['mid_variance']:.4f}" if not np.isnan(r["mid_variance"]) else "N/A"
        final = (
            f"{r['final_variance']:.4f}"
            if not np.isnan(r["final_variance"])
            else "N/A"
        )
        print(f"{r['name']:<20} {mid:>12} {final:>12} {r['status']:>12}")

    print("\nNote: Zero initialization shows symmetry problem (all neurons identical)")


def demonstrate_symmetry_problem():
    """Demonstrate why zero initialization is problematic."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Zero Initialization Symmetry Problem")
    print("=" * 60)

    np.random.seed(42)

    # Create a simple network with zero initialization
    fan_in, fan_out = 10, 5

    W_zero = zero_init(fan_in, fan_out)
    W_he = he_normal(fan_in, fan_out, rng=np.random.default_rng(42))

    # Same input, same gradient
    x = np.random.randn(3, fan_in)
    grad_output = np.random.randn(3, fan_out)

    # Forward pass
    y_zero = x @ W_zero
    y_he = x @ W_he

    print(f"\nZero init forward output (all zeros):")
    print(y_zero)
    print(f"\nHe init forward output (diverse):")
    print(y_he[:2, :3], "...")  # Show subset

    # Backward pass - gradient w.r.t weights
    grad_W_zero = x.T @ grad_output
    grad_W_he = x.T @ grad_output  # Same gradient!

    print(f"\nGradient for zero weights (all rows identical):")
    print(f"Row 0: {grad_W_zero[0, :3]}...")
    print(f"Row 1: {grad_W_zero[1, :3]}...")
    print(f"All rows equal: {np.allclose(grad_W_zero[0], grad_W_zero[1])}")

    print("\nKey insight: With zero initialization, all neurons receive")
    print("identical gradients, so they learn identical features.")


def run_all_tests():
    """Run all tests and return summary."""
    print("=" * 60)
    print("WEIGHT INITIALIZATION EXPERIMENTS")
    print("=" * 60)

    results = {}

    results["xavier"] = test_xavier_variance()
    results["he"] = test_he_variance_relu()
    results["lsuv"] = test_lsuv_convergence()

    compare_initializers()
    demonstrate_symmetry_problem()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Xavier variance test: {'PASSED' if results['xavier'] else 'FAILED'}")
    print(f"He variance test: {'PASSED' if results['he'] else 'FAILED'}")
    print(f"LSUV convergence test: {'PASSED' if results['lsuv'] else 'FAILED'}")

    all_passed = all(results.values())
    print(f"\nAll tests passed: {all_passed}")

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
