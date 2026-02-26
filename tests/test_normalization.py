"""
Unit tests for normalization techniques.

Tests cover:
    - BatchNorm1d: (N, C) and (N, C, L) cases
    - BatchNorm2d: (N, C, H, W) case
    - LayerNorm: (N, D) and (N, L, D) cases
    - InstanceNorm2d: (N, C, H, W) case
    - GroupNorm: (N, C, H, W) with various group sizes

Each test verifies:
    - Forward pass shape and values
    - Backward pass gradient shapes
    - Training vs inference behavior
    - PyTorch comparison (when available)
"""

import pytest
import numpy as np

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from phase3_training.normalization import (
    BatchNorm1d,
    BatchNorm2d,
    LayerNorm,
    InstanceNorm2d,
    GroupNorm,
    get_normalization,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    if HAS_TORCH:
        torch.manual_seed(42)


@pytest.fixture
def small_2d_input():
    """Small 2D input for BatchNorm1d: (N, C)."""
    return np.random.randn(4, 8).astype(np.float64)


@pytest.fixture
def small_3d_input():
    """Small 3D input for BatchNorm1d: (N, C, L)."""
    return np.random.randn(4, 8, 16).astype(np.float64)


@pytest.fixture
def small_4d_input():
    """Small 4D input for 2D norms: (N, C, H, W)."""
    return np.random.randn(4, 8, 16, 16).astype(np.float64)


# =============================================================================
# BatchNorm1d Tests
# =============================================================================


class TestBatchNorm1d:
    """Test suite for BatchNorm1d."""

    def test_forward_shape_2d(self, small_2d_input):
        """Test forward pass preserves shape for (N, C)."""
        bn = BatchNorm1d(8)
        out = bn.forward(small_2d_input)
        assert out.shape == small_2d_input.shape

    def test_forward_shape_3d(self, small_3d_input):
        """Test forward pass preserves shape for (N, C, L)."""
        bn = BatchNorm1d(8)
        out = bn.forward(small_3d_input)
        assert out.shape == small_3d_input.shape

    def test_training_statistics(self, small_2d_input):
        """Test training mode uses batch statistics."""
        bn = BatchNorm1d(8)
        bn.train()
        out = bn.forward(small_2d_input)

        # Check that running stats were updated
        assert not np.allclose(bn.running_mean, 0)
        assert not np.allclose(bn.running_var, 1)

    def test_inference_statistics(self, small_2d_input):
        """Test inference mode uses running statistics."""
        bn = BatchNorm1d(8)
        bn.train()
        # First update running stats
        _ = bn.forward(small_2d_input)

        bn.eval()
        new_input = np.random.randn(4, 8).astype(np.float64)
        out = bn.forward(new_input)

        # Running stats should not change during inference
        saved_mean = bn.running_mean.copy()
        saved_var = bn.running_var.copy()

        _ = bn.forward(new_input)
        assert np.allclose(bn.running_mean, saved_mean)
        assert np.allclose(bn.running_var, saved_var)

    def test_backward_shape_2d(self, small_2d_input):
        """Test backward pass preserves shape for (N, C)."""
        bn = BatchNorm1d(8)
        out = bn.forward(small_2d_input)
        grad_output = np.ones_like(out)
        grad_input = bn.backward(grad_output)
        assert grad_input.shape == small_2d_input.shape

    def test_backward_shape_3d(self, small_3d_input):
        """Test backward pass preserves shape for (N, C, L)."""
        bn = BatchNorm1d(8)
        out = bn.forward(small_3d_input)
        grad_output = np.ones_like(out)
        grad_input = bn.backward(grad_output)
        assert grad_input.shape == small_3d_input.shape

    def test_normalized_output(self, small_2d_input):
        """Test that normalized output has mean~0 and var~1."""
        bn = BatchNorm1d(8)
        bn.train()
        out = bn.forward(small_2d_input)

        # Per-channel mean should be close to 0
        channel_means = np.mean(out, axis=0)
        assert np.allclose(channel_means, 0, atol=1e-8)

        # Per-channel variance should be close to 1 (within numerical precision)
        channel_vars = np.var(out, axis=0)
        assert np.allclose(channel_vars, 1, atol=1e-4)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_pytorch_comparison_2d(self, small_2d_input):
        """Compare forward pass with PyTorch."""
        # NumPy implementation
        bn_np = BatchNorm1d(8)
        bn_np.train()
        out_np = bn_np.forward(small_2d_input)

        # PyTorch implementation - use double precision for comparison
        bn_torch = nn.BatchNorm1d(8)
        bn_torch.train()
        bn_torch.double()  # Set to double precision
        out_torch = bn_torch(torch.tensor(small_2d_input, dtype=torch.float64)).detach().numpy()

        # Outputs should be similar (not exact due to different running mean updates)
        assert np.allclose(out_np, out_torch, atol=1e-5)


# =============================================================================
# BatchNorm2d Tests
# =============================================================================


class TestBatchNorm2d:
    """Test suite for BatchNorm2d."""

    def test_forward_shape(self, small_4d_input):
        """Test forward pass preserves shape."""
        bn = BatchNorm2d(8)
        out = bn.forward(small_4d_input)
        assert out.shape == small_4d_input.shape

    def test_training_statistics(self, small_4d_input):
        """Test training mode uses batch statistics."""
        bn = BatchNorm2d(8)
        bn.train()
        out = bn.forward(small_4d_input)

        # Normalized output should have mean~0 and var~1 per channel
        for c in range(8):
            channel_data = out[:, c, :, :]
            assert np.allclose(np.mean(channel_data), 0, atol=1e-8)
            assert np.allclose(np.var(channel_data), 1, atol=1e-4)

    def test_inference_mode(self, small_4d_input):
        """Test eval mode uses running statistics."""
        bn = BatchNorm2d(8)
        bn.train()
        _ = bn.forward(small_4d_input)

        bn.eval()
        new_input = np.random.randn(4, 8, 16, 16).astype(np.float64)
        out = bn.forward(new_input)

        # Should not crash
        assert out.shape == small_4d_input.shape

    def test_backward_shape(self, small_4d_input):
        """Test backward pass preserves shape."""
        bn = BatchNorm2d(8)
        out = bn.forward(small_4d_input)
        grad_output = np.ones_like(out)
        grad_input = bn.backward(grad_output)
        assert grad_input.shape == small_4d_input.shape

    def test_backward_gradient_flow(self, small_4d_input):
        """Test that gradients flow through backward pass."""
        bn = BatchNorm2d(8)
        out = bn.forward(small_4d_input)
        # Use random gradient instead of ones to ensure non-uniform gradients
        np.random.seed(42)
        grad_output = np.random.randn(*out.shape).astype(np.float64)
        grad_input = bn.backward(grad_output)

        # Gradients should not be zero
        assert not np.allclose(grad_input, 0, atol=1e-10)
        assert not np.allclose(bn.grad_gamma, 0, atol=1e-10)
        assert not np.allclose(bn.grad_beta, 0, atol=1e-10)

    def test_parameters_gradients_shape(self, small_4d_input):
        """Test parameter gradient shapes."""
        bn = BatchNorm2d(8)
        out = bn.forward(small_4d_input)
        grad_output = np.ones_like(out)
        _ = bn.backward(grad_output)

        assert bn.grad_gamma.shape == (8,)
        assert bn.grad_beta.shape == (8,)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_pytorch_comparison(self, small_4d_input):
        """Compare forward pass with PyTorch."""
        # NumPy implementation
        bn_np = BatchNorm2d(8)
        bn_np.train()
        out_np = bn_np.forward(small_4d_input)

        # PyTorch implementation - use double precision for comparison
        bn_torch = nn.BatchNorm2d(8)
        bn_torch.train()
        bn_torch.double()  # Set to double precision
        out_torch = bn_torch(torch.tensor(small_4d_input, dtype=torch.float64)).detach().numpy()

        assert np.allclose(out_np, out_torch, atol=1e-5)


# =============================================================================
# LayerNorm Tests
# =============================================================================


class TestLayerNorm:
    """Test suite for LayerNorm."""

    def test_forward_shape_2d(self, small_2d_input):
        """Test forward pass preserves shape for (N, D)."""
        ln = LayerNorm(8)
        out = ln.forward(small_2d_input)
        assert out.shape == small_2d_input.shape

    def test_forward_shape_3d(self, random_seed):
        """Test forward pass preserves shape for (N, L, D)."""
        x = np.random.randn(4, 16, 8).astype(np.float64)
        ln = LayerNorm(8)
        out = ln.forward(x)
        assert out.shape == x.shape

    def test_normalized_on_hidden_dim(self, random_seed):
        """Test normalization is computed over hidden dimension."""
        x = np.random.randn(4, 8).astype(np.float64)
        ln = LayerNorm(8)
        out = ln.forward(x)

        # Each sample should have mean~0 and var~1
        for i in range(4):
            assert np.allclose(np.mean(out[i]), 0, atol=1e-8)
            assert np.allclose(np.var(out[i]), 1, atol=1e-4)

    def test_batch_independence(self, random_seed):
        """Test that each sample is normalized independently."""
        # Create input with very different distributions per sample
        x = np.array(
            [
                [1, 2, 3, 4],
                [100, 200, 300, 400],
                [-1000, -2000, -3000, -4000],
            ],
            dtype=np.float64,
        )
        ln = LayerNorm(4)
        out = ln.forward(x)

        # All samples should have similar statistics after normalization
        for i in range(3):
            assert np.allclose(np.mean(out[i]), 0, atol=1e-10)
            assert np.allclose(np.var(out[i]), 1, atol=1e-10)

    def test_backward_shape_2d(self, small_2d_input):
        """Test backward pass preserves shape for (N, D)."""
        ln = LayerNorm(8)
        out = ln.forward(small_2d_input)
        grad_output = np.ones_like(out)
        grad_input = ln.backward(grad_output)
        assert grad_input.shape == small_2d_input.shape

    def test_backward_shape_3d(self, random_seed):
        """Test backward pass preserves shape for (N, L, D)."""
        x = np.random.randn(4, 16, 8).astype(np.float64)
        ln = LayerNorm(8)
        out = ln.forward(x)
        grad_output = np.ones_like(out)
        grad_input = ln.backward(grad_output)
        assert grad_input.shape == x.shape

    def test_gradient_numerical(self, random_seed):
        """Test gradient with numerical approximation."""
        x = np.random.randn(4, 8).astype(np.float64)
        ln = LayerNorm(8)

        # Forward pass
        out = ln.forward(x)

        # Analytical gradient
        grad_output = np.ones_like(out)
        grad_analytical = ln.backward(grad_output)

        # Numerical gradient
        eps = 1e-5
        grad_numerical = np.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x[i, j] += eps
                f_plus = np.sum(ln.forward(x))
                x[i, j] -= 2 * eps
                f_minus = np.sum(ln.forward(x))
                x[i, j] += eps
                grad_numerical[i, j] = (f_plus - f_minus) / (2 * eps)

        assert np.allclose(grad_analytical, grad_numerical, atol=1e-5)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_pytorch_comparison(self, random_seed):
        """Compare forward pass with PyTorch."""
        x = np.random.randn(4, 8).astype(np.float64)

        # NumPy implementation
        ln_np = LayerNorm(8)
        out_np = ln_np.forward(x)

        # PyTorch implementation - use double precision
        ln_torch = nn.LayerNorm(8)
        ln_torch.double()
        out_torch = ln_torch(torch.tensor(x, dtype=torch.float64)).detach().numpy()

        assert np.allclose(out_np, out_torch, atol=1e-5)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_pytorch_gradient_comparison(self, random_seed):
        """Compare gradients with PyTorch."""
        x = np.random.randn(4, 8).astype(np.float64)

        # NumPy implementation
        ln_np = LayerNorm(8)
        out_np = ln_np.forward(x)
        grad_output = np.ones_like(out_np)
        grad_np = ln_np.backward(grad_output)

        # PyTorch implementation - use double precision
        x_torch = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        ln_torch = nn.LayerNorm(8)
        ln_torch.double()
        out_torch = ln_torch(x_torch)
        out_torch.backward(torch.ones_like(out_torch))
        grad_torch = x_torch.grad.numpy()

        assert np.allclose(grad_np, grad_torch, atol=1e-5)


# =============================================================================
# InstanceNorm2d Tests
# =============================================================================


class TestInstanceNorm2d:
    """Test suite for InstanceNorm2d."""

    def test_forward_shape(self, small_4d_input):
        """Test forward pass preserves shape."""
        ins = InstanceNorm2d(8)
        out = ins.forward(small_4d_input)
        assert out.shape == small_4d_input.shape

    def test_per_instance_per_channel_normalization(self, random_seed):
        """Test that each (sample, channel) is normalized independently."""
        x = np.random.randn(2, 4, 8, 8).astype(np.float64)
        ins = InstanceNorm2d(4)
        out = ins.forward(x)

        # Each (n, c) should have mean~0 and var~1
        for n in range(2):
            for c in range(4):
                assert np.allclose(np.mean(out[n, c]), 0, atol=1e-8)
                assert np.allclose(np.var(out[n, c]), 1, atol=1e-4)

    def test_backward_shape(self, small_4d_input):
        """Test backward pass preserves shape."""
        ins = InstanceNorm2d(8)
        out = ins.forward(small_4d_input)
        grad_output = np.ones_like(out)
        grad_input = ins.backward(grad_output)
        assert grad_input.shape == small_4d_input.shape

    def test_no_affine(self, random_seed):
        """Test without learnable parameters."""
        x = np.random.randn(2, 4, 8, 8).astype(np.float64)
        ins = InstanceNorm2d(4, affine=False)
        out = ins.forward(x)

        # Should still normalize correctly
        for n in range(2):
            for c in range(4):
                assert np.allclose(np.mean(out[n, c]), 0, atol=1e-8)
                assert np.allclose(np.var(out[n, c]), 1, atol=1e-4)

        # No learnable parameters
        assert len(ins.parameters()) == 0

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_pytorch_comparison(self, random_seed):
        """Compare forward pass with PyTorch."""
        x = np.random.randn(2, 4, 8, 8).astype(np.float64)

        # NumPy implementation
        ins_np = InstanceNorm2d(4)
        out_np = ins_np.forward(x)

        # PyTorch implementation - use double precision
        ins_torch = nn.InstanceNorm2d(4, affine=True)
        ins_torch.double()
        out_torch = ins_torch(torch.tensor(x, dtype=torch.float64)).detach().numpy()

        assert np.allclose(out_np, out_torch, atol=1e-5)


# =============================================================================
# GroupNorm Tests
# =============================================================================


class TestGroupNorm:
    """Test suite for GroupNorm."""

    def test_forward_shape(self, small_4d_input):
        """Test forward pass preserves shape."""
        gn = GroupNorm(num_groups=2, num_channels=8)
        out = gn.forward(small_4d_input)
        assert out.shape == small_4d_input.shape

    def test_groups_divisibility_error(self):
        """Test error when channels not divisible by groups."""
        with pytest.raises(ValueError):
            GroupNorm(num_groups=3, num_channels=8)  # 8 % 3 != 0

    def test_group_4_channels_32(self, random_seed):
        """Test GroupNorm with group=4, C=32."""
        x = np.random.randn(2, 32, 8, 8).astype(np.float64)
        gn = GroupNorm(num_groups=4, num_channels=32)
        out = gn.forward(x)

        assert out.shape == x.shape

        # Each group should have mean~0 and var~1
        # Groups: channels 0-7, 8-15, 16-23, 24-31
        for n in range(2):
            for g in range(4):
                group_data = out[n, g * 8 : (g + 1) * 8, :, :]
                assert np.allclose(np.mean(group_data), 0, atol=1e-8)
                assert np.allclose(np.var(group_data), 1, atol=1e-4)

    def test_backward_shape(self, small_4d_input):
        """Test backward pass preserves shape."""
        gn = GroupNorm(num_groups=2, num_channels=8)
        out = gn.forward(small_4d_input)
        grad_output = np.ones_like(out)
        grad_input = gn.backward(grad_output)
        assert grad_input.shape == small_4d_input.shape

    def test_single_group_equals_layer_norm(self, random_seed):
        """Test that G=1 is equivalent to LayerNorm over channels."""
        x = np.random.randn(2, 8, 4, 4).astype(np.float64)

        # GroupNorm with 1 group
        gn = GroupNorm(num_groups=1, num_channels=8)
        out_gn = gn.forward(x)

        # Check that all channels have same normalization
        for n in range(2):
            all_channels = out_gn[n, :, :, :]
            assert np.allclose(np.mean(all_channels), 0, atol=1e-8)
            assert np.allclose(np.var(all_channels), 1, atol=1e-4)

    def test_groups_equal_channels_equals_instance_norm(self, random_seed):
        """Test that G=C is equivalent to InstanceNorm."""
        x = np.random.randn(2, 4, 8, 8).astype(np.float64)

        # GroupNorm with C groups (each channel is a group)
        gn = GroupNorm(num_groups=4, num_channels=4, affine=False)
        out_gn = gn.forward(x)

        # InstanceNorm
        ins = InstanceNorm2d(4, affine=False)
        out_ins = ins.forward(x)

        assert np.allclose(out_gn, out_ins, atol=1e-10)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_pytorch_comparison(self, random_seed):
        """Compare forward pass with PyTorch."""
        x = np.random.randn(2, 32, 8, 8).astype(np.float64)

        # NumPy implementation
        gn_np = GroupNorm(num_groups=4, num_channels=32)
        out_np = gn_np.forward(x)

        # PyTorch implementation - use double precision
        gn_torch = nn.GroupNorm(num_groups=4, num_channels=32)
        gn_torch.double()
        out_torch = gn_torch(torch.tensor(x, dtype=torch.float64)).detach().numpy()

        assert np.allclose(out_np, out_torch, atol=1e-5)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_pytorch_gradient_comparison(self, random_seed):
        """Compare gradients with PyTorch."""
        x = np.random.randn(2, 32, 8, 8).astype(np.float64)

        # NumPy implementation
        gn_np = GroupNorm(num_groups=4, num_channels=32)
        out_np = gn_np.forward(x)
        grad_output = np.ones_like(out_np)
        grad_np = gn_np.backward(grad_output)

        # PyTorch implementation - use double precision
        x_torch = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        gn_torch = nn.GroupNorm(num_groups=4, num_channels=32)
        gn_torch.double()
        out_torch = gn_torch(x_torch)
        out_torch.backward(torch.ones_like(out_torch))
        grad_torch = x_torch.grad.numpy()

        assert np.allclose(grad_np, grad_torch, atol=1e-5)


# =============================================================================
# Registry Tests
# =============================================================================


class TestGetNormalization:
    """Test normalization factory function."""

    def test_get_batchnorm1d(self):
        """Test getting BatchNorm1d."""
        bn = get_normalization("batchnorm1d", num_features=8)
        assert isinstance(bn, BatchNorm1d)

    def test_get_batchnorm2d(self):
        """Test getting BatchNorm2d."""
        bn = get_normalization("batchnorm2d", num_features=8)
        assert isinstance(bn, BatchNorm2d)

    def test_get_layernorm(self):
        """Test getting LayerNorm."""
        ln = get_normalization("layernorm", normalized_shape=8)
        assert isinstance(ln, LayerNorm)

    def test_get_instancenorm(self):
        """Test getting InstanceNorm2d."""
        ins = get_normalization("instancenorm", num_features=8)
        assert isinstance(ins, InstanceNorm2d)

    def test_get_groupnorm(self):
        """Test getting GroupNorm."""
        gn = get_normalization("groupnorm", num_groups=4, num_channels=32)
        assert isinstance(gn, GroupNorm)

    def test_invalid_name(self):
        """Test error for invalid normalization name."""
        with pytest.raises(ValueError):
            get_normalization("invalid_norm")


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for normalization layers."""

    def test_batch_size_1_compatibility(self, random_seed):
        """Test that LayerNorm and GroupNorm work with batch_size=1."""
        x = np.random.randn(1, 32, 8, 8).astype(np.float64)

        # LayerNorm should work with 2D input (N, D)
        x_pooled = x.mean(axis=(2, 3))  # Shape: (1, 32)
        ln = LayerNorm(32)
        out_ln = ln.forward(x_pooled)
        assert out_ln.shape == x_pooled.shape

        # GroupNorm should work (batch-independent)
        gn = GroupNorm(num_groups=4, num_channels=32)
        out_gn = gn.forward(x)
        assert out_gn.shape == x.shape

    def test_train_eval_consistency(self, random_seed):
        """Test that train/eval modes are consistent across layers."""
        x = np.random.randn(4, 8, 16, 16).astype(np.float64)

        # Test BatchNorm2d
        bn = BatchNorm2d(8)
        bn.train()
        out_train = bn.forward(x)
        bn.eval()
        out_eval = bn.forward(x)

        # Outputs should differ (running stats vs batch stats)
        assert not np.allclose(out_train, out_eval)

    def test_all_norms_gradient_flow(self, random_seed):
        """Test that all norms allow gradient flow."""
        x = np.random.randn(2, 8, 4, 4).astype(np.float64)
        np.random.seed(42)
        grad_output = np.random.randn(2, 8, 4, 4).astype(np.float64)

        norms = [
            BatchNorm2d(8),
            LayerNorm((8, 4, 4)),
            InstanceNorm2d(8),
            GroupNorm(2, 8),
        ]

        for norm in norms:
            out = norm.forward(x)
            grad_input = norm.backward(grad_output)

            # Gradient should not be all zeros
            assert not np.allclose(grad_input, 0, atol=1e-10), f"{type(norm).__name__} has zero gradient"


# =============================================================================
# Run Tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
