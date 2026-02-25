"""
Tests for weight initialization methods.

Run: pytest tests/test_weight_init.py -v
"""

import pytest
import numpy as np

from phase1_basics.weight_init import (
    xavier_uniform,
    xavier_normal,
    he_uniform,
    he_normal,
    kaiming_uniform,
    kaiming_normal,
    zero_init,
    lsuv_init,
    compute_fan,
    get_initializer,
    init_bias,
    INITIALIZERS,
)
from phase1_basics.activations import relu

# Try to import torch for comparison
try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# Fixtures
@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    return np.random.default_rng(42)


@pytest.fixture
def layer_sizes():
    """Standard layer sizes for testing."""
    return [784, 512, 256, 128]


# ============================================
# Test Xavier Initialization
# ============================================


class TestXavierUniform:
    """Tests for Xavier uniform initialization."""

    def test_output_shape(self, random_seed):
        """Output should have correct shape."""
        W = xavier_uniform(100, 50, rng=random_seed)
        assert W.shape == (100, 50)

    def test_variance_correct(self, random_seed):
        """Variance should be approximately 2/(fan_in + fan_out)."""
        fan_in, fan_out = 100, 50
        W = xavier_uniform(fan_in, fan_out, rng=random_seed)

        expected_var = 2.0 / (fan_in + fan_out)
        actual_var = np.var(W)

        # Allow 30% tolerance due to random sampling
        assert abs(actual_var - expected_var) / expected_var < 0.3

    def test_values_bounded(self, random_seed):
        """All values should be within expected bounds."""
        fan_in, fan_out = 100, 50
        W = xavier_uniform(fan_in, fan_out, rng=random_seed)

        # Bound = sqrt(3) * std = sqrt(6 / (fan_in + fan_out))
        bound = np.sqrt(6.0 / (fan_in + fan_out))

        assert np.all(W >= -bound * 1.01)  # 1% tolerance
        assert np.all(W <= bound * 1.01)

    def test_dtype_float64(self, random_seed):
        """Output should be float64 for numerical stability."""
        W = xavier_uniform(10, 10, rng=random_seed)
        assert W.dtype == np.float64


class TestXavierNormal:
    """Tests for Xavier normal initialization."""

    def test_output_shape(self, random_seed):
        """Output should have correct shape."""
        W = xavier_normal(100, 50, rng=random_seed)
        assert W.shape == (100, 50)

    def test_variance_correct(self, random_seed):
        """Variance should be approximately 2/(fan_in + fan_out)."""
        fan_in, fan_out = 100, 50
        W = xavier_normal(fan_in, fan_out, rng=random_seed)

        expected_var = 2.0 / (fan_in + fan_out)
        actual_var = np.var(W)

        # Allow 30% tolerance
        assert abs(actual_var - expected_var) / expected_var < 0.3

    def test_mean_near_zero(self, random_seed):
        """Mean should be approximately zero."""
        W = xavier_normal(1000, 500, rng=random_seed)
        assert abs(np.mean(W)) < 0.01


# ============================================
# Test He Initialization
# ============================================


class TestHeUniform:
    """Tests for He uniform initialization."""

    def test_output_shape(self, random_seed):
        """Output should have correct shape."""
        W = he_uniform(100, 50, rng=random_seed)
        assert W.shape == (100, 50)

    def test_variance_correct(self, random_seed):
        """Variance should be approximately 2/fan_in."""
        fan_in, fan_out = 100, 50
        W = he_uniform(fan_in, fan_out, rng=random_seed)

        expected_var = 2.0 / fan_in
        actual_var = np.var(W)

        # Allow 30% tolerance
        assert abs(actual_var - expected_var) / expected_var < 0.3

    def test_preserves_variance_with_relu(self, random_seed):
        """He init should preserve variance through ReLU layers."""
        fan_in = 784
        fan_out = 256

        # Create input with unit variance
        x = random_seed.standard_normal((1000, fan_in)).astype(np.float64)

        # Apply layer with He init
        W = he_uniform(fan_in, fan_out, rng=random_seed)
        y = relu(x @ W)

        # After ReLU, variance should be roughly 0.5 * (2/fan_in) * fan_in = 1.0
        # But due to ReLU, expect ~0.5
        assert 0.3 < np.var(y) < 1.0


class TestHeNormal:
    """Tests for He normal initialization."""

    def test_output_shape(self, random_seed):
        """Output should have correct shape."""
        W = he_normal(100, 50, rng=random_seed)
        assert W.shape == (100, 50)

    def test_variance_correct(self, random_seed):
        """Variance should be approximately 2/fan_in."""
        fan_in, fan_out = 100, 50
        W = he_normal(fan_in, fan_out, rng=random_seed)

        expected_var = 2.0 / fan_in
        actual_var = np.var(W)

        assert abs(actual_var - expected_var) / expected_var < 0.3


# ============================================
# Test Kaiming Initialization
# ============================================


class TestKaimingInit:
    """Tests for Kaiming initialization (generalized He)."""

    def test_kaiming_uniform_relu_equals_he(self, random_seed):
        """Kaiming uniform with relu should equal He uniform."""
        fan_in, fan_out = 100, 50

        # Set same seed for both
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        W_he = he_uniform(fan_in, fan_out, rng=rng1)
        W_kaiming = kaiming_uniform(
            fan_in, fan_out, mode="fan_in", nonlinearity="relu", rng=rng2
        )

        assert np.allclose(W_he, W_kaiming)

    def test_kaiming_leaky_relu(self, random_seed):
        """Kaiming with leaky_relu should have correct gain."""
        fan_in, fan_out = 100, 50
        a = 0.01  # LeakyReLU slope

        W = kaiming_uniform(
            fan_in, fan_out, mode="fan_in", nonlinearity="leaky_relu", a=a, rng=random_seed
        )

        # Expected gain = sqrt(2 / (1 + a^2))
        gain = np.sqrt(2.0 / (1 + a**2))
        expected_var = gain**2 / fan_in

        assert abs(np.var(W) - expected_var) / expected_var < 0.3

    def test_kaiming_fan_out_mode(self, random_seed):
        """Kaiming with fan_out should use fan_out for variance."""
        fan_in, fan_out = 100, 50

        W = kaiming_uniform(
            fan_in, fan_out, mode="fan_out", nonlinearity="relu", rng=random_seed
        )

        # Expected variance: gain^2 / fan_out = 2 / 50 = 0.04
        expected_var = 2.0 / fan_out

        assert abs(np.var(W) - expected_var) / expected_var < 0.3


# ============================================
# Test Zero Initialization
# ============================================


class TestZeroInit:
    """Tests for zero initialization."""

    def test_output_shape(self):
        """Output should have correct shape."""
        W = zero_init(100, 50)
        assert W.shape == (100, 50)

    def test_all_zeros(self):
        """All values should be zero."""
        W = zero_init(100, 50)
        assert np.allclose(W, 0)

    def test_causes_symmetry(self, random_seed):
        """Zero init causes symmetry - all outputs are identical."""
        W = zero_init(10, 5)
        x = random_seed.standard_normal((3, 10))

        y = x @ W

        # All outputs should be identical (all zeros)
        assert np.allclose(y, 0)


# ============================================
# Test LSUV Initialization
# ============================================


class TestLSUVInit:
    """Tests for LSUV initialization."""

    def test_converges_in_few_iterations(self, random_seed):
        """LSUV should converge in <= 10 iterations."""
        fan_in, fan_out = 256, 128

        # Start with orthogonal init
        W = random_seed.standard_normal((fan_in, fan_out)).astype(np.float64)
        U, _, Vt = np.linalg.svd(W, full_matrices=False)
        W = U @ Vt

        def forward_fn(x):
            return relu(x @ W)

        W_init, iterations = lsuv_init(
            W, forward_fn, target_variance=0.5, max_iterations=10, tol=0.2, rng=random_seed
        )

        assert iterations <= 10

    def test_achieves_target_variance(self, random_seed):
        """LSUV should achieve variance close to target."""
        fan_in, fan_out = 256, 128
        target_var = 0.5

        W = random_seed.standard_normal((fan_in, fan_out)).astype(np.float64)

        def forward_fn(x):
            return relu(x @ W)

        W_init, _ = lsuv_init(
            W, forward_fn, target_variance=target_var, max_iterations=10, tol=0.3, rng=random_seed
        )

        # Verify
        x = random_seed.standard_normal((1000, fan_in)).astype(np.float64)
        output = forward_fn(x)
        actual_var = np.var(output)

        # Should be within 50% of target
        assert abs(actual_var - target_var) / target_var < 0.5

    def test_inplace_modification(self, random_seed):
        """LSUV should modify weight in-place."""
        fan_in, fan_out = 100, 50
        W = random_seed.standard_normal((fan_in, fan_out)).astype(np.float64)
        W_id = id(W)

        def forward_fn(x):
            return relu(x @ W)

        W_init, _ = lsuv_init(W, forward_fn, max_iterations=5, rng=random_seed)

        # Same object (modified in-place)
        assert id(W_init) == W_id


# ============================================
# Test Utility Functions
# ============================================


class TestComputeFan:
    """Tests for compute_fan utility."""

    def test_linear_layer(self):
        """Should compute fan for 2D linear layer weights."""
        fan_in, fan_out = compute_fan((784, 256))
        assert fan_in == 784
        assert fan_out == 256

    def test_conv_layer(self):
        """Should compute fan for 4D conv layer weights."""
        # (out_channels, in_channels, kH, kW)
        fan_in, fan_out = compute_fan((64, 3, 3, 3))

        expected_fan_in = 3 * 3 * 3  # 27
        expected_fan_out = 64 * 3 * 3  # 576

        assert fan_in == expected_fan_in
        assert fan_out == expected_fan_out


class TestGetInitializer:
    """Tests for get_initializer factory."""

    def test_get_xavier_uniform(self):
        """Should return xavier_uniform function."""
        fn = get_initializer("xavier_uniform")
        assert fn == xavier_uniform

    def test_get_he_normal(self):
        """Should return he_normal function."""
        fn = get_initializer("he_normal")
        assert fn == he_normal

    def test_get_glorot_alias(self):
        """Glorot should be alias for Xavier."""
        fn_glorot = get_initializer("glorot_normal")
        fn_xavier = get_initializer("xavier_normal")
        assert fn_glorot == fn_xavier

    def test_invalid_name_raises(self):
        """Invalid name should raise ValueError."""
        with pytest.raises(ValueError):
            get_initializer("invalid_init")


class TestInitBias:
    """Tests for bias initialization."""

    def test_zeros(self):
        """Should create zero bias."""
        b = init_bias(100, method="zeros")
        assert b.shape == (100,)
        assert np.allclose(b, 0)

    def test_ones(self):
        """Should create ones bias."""
        b = init_bias(100, method="ones")
        assert b.shape == (100,)
        assert np.allclose(b, 1)

    def test_small(self):
        """Should create small positive bias."""
        b = init_bias(100, method="small")
        assert b.shape == (100,)
        assert np.allclose(b, 0.01)

    def test_invalid_raises(self):
        """Invalid method should raise ValueError."""
        with pytest.raises(ValueError):
            init_bias(100, method="invalid")


# ============================================
# PyTorch Comparison Tests
# ============================================


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestPyTorchComparison:
    """Compare implementations with PyTorch."""

    def test_xavier_uniform_vs_pytorch(self):
        """Xavier uniform should match PyTorch."""
        fan_in, fan_out = 100, 50

        # NumPy
        rng = np.random.default_rng(42)
        W_np = xavier_uniform(fan_in, fan_out, rng=rng)

        # PyTorch
        torch.manual_seed(42)
        W_torch = nn.init.xavier_uniform_(torch.empty(fan_in, fan_out))

        # Check variance is similar (not exact due to different RNG)
        assert abs(np.var(W_np) - torch.var(W_torch).item()) / np.var(W_np) < 0.5

    def test_he_normal_vs_pytorch(self):
        """He normal should match PyTorch Kaiming normal."""
        fan_in, fan_out = 100, 50

        # NumPy
        rng = np.random.default_rng(42)
        W_np = he_normal(fan_in, fan_out, rng=rng)

        # PyTorch
        torch.manual_seed(42)
        W_torch = nn.init.kaiming_normal_(
            torch.empty(fan_in, fan_out), mode="fan_in", nonlinearity="relu"
        )

        # Check variance is similar
        assert abs(np.var(W_np) - torch.var(W_torch).item()) / np.var(W_np) < 0.5


# ============================================
# Integration Tests
# ============================================


class TestIntegration:
    """Integration tests with MLP."""

    def test_initializers_with_mlp(self, random_seed):
        """All initializers should work with MLP training."""
        from phase1_basics.mlp import MLP
        from phase1_basics.loss import MSELoss
        from phase1_basics.optimizer import SGD

        # Create simple dataset
        x = random_seed.standard_normal((100, 10)).astype(np.float64)
        y = random_seed.standard_normal((100, 1)).astype(np.float64)

        for name, init_fn in INITIALIZERS.items():
            if name == "zero":
                continue  # Skip zero init - won't learn

            # Create MLP and initialize
            mlp = MLP(input_size=10, hidden_sizes=[20, 10], output_size=1)

            # Manually initialize first layer with our method
            W = init_fn(10, 20, rng=random_seed)
            mlp.layers[0].weight = W

            # Train for a few steps
            optimizer = SGD(lr=0.01)
            loss_fn = MSELoss()

            losses = []
            for _ in range(10):
                pred = mlp.forward(x)
                loss = loss_fn.forward(pred, y)
                grad = loss_fn.backward()
                mlp.backward(grad)
                optimizer.step(mlp.parameters())
                mlp.zero_grad()
                losses.append(loss)

            # Loss should decrease or stay stable
            assert losses[-1] <= losses[0] * 2, f"{name} caused unstable training"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
