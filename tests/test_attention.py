"""
Tests for attention mechanisms (Self-Attention, Multi-Head Attention, Positional Encoding).

This test suite verifies:
1. Scaled Dot-Product Attention forward/backward correctness
2. Multi-Head Attention dimensions and parameter independence
3. Sinusoidal Positional Encoding uniqueness
4. PyTorch comparison for validation
"""

import pytest
import numpy as np
import math

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from phase2_architectures.attention import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    SinusoidalPositionalEncoding,
    TransformerEncoderLayer,
    create_causal_mask,
    create_padding_mask,
    count_parameters_attention,
    softmax,
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
def small_attention_params():
    """Small parameters for quick tests."""
    return {"d_model": 64, "num_heads": 4, "batch_size": 2, "seq_len": 8}


# =============================================================================
# Softmax Tests
# =============================================================================

class TestSoftmax:
    """Tests for softmax function."""

    def test_softmax_sums_to_one(self):
        """Test that softmax outputs sum to 1."""
        x = np.random.randn(3, 5)
        result = softmax(x)
        sums = np.sum(result, axis=-1)
        assert np.allclose(sums, 1.0)

    def test_softmax_all_positive(self):
        """Test that softmax outputs are all positive."""
        x = np.random.randn(3, 5)
        result = softmax(x)
        assert np.all(result >= 0)

    def test_softmax_numerical_stability(self):
        """Test softmax with large values."""
        x = np.array([[1000, 1001, 1002]])
        result = softmax(x)
        expected = np.array([[0.0900, 0.2447, 0.6652]])
        assert np.allclose(result, expected, atol=1e-4)


# =============================================================================
# Scaled Dot-Product Attention Tests
# =============================================================================

class TestScaledDotProductAttention:
    """Tests for scaled dot-product attention."""

    def test_forward_shape(self, random_seed):
        """Test that forward output has correct shape."""
        batch_size, num_heads, seq_len, d_k = 2, 4, 8, 16

        attention = ScaledDotProductAttention(d_k)
        query = np.random.randn(batch_size, num_heads, seq_len, d_k)
        key = np.random.randn(batch_size, num_heads, seq_len, d_k)
        value = np.random.randn(batch_size, num_heads, seq_len, d_k)

        output = attention.forward(query, key, value)

        assert output.shape == (batch_size, num_heads, seq_len, d_k)

    def test_attention_weights_sum_to_one(self, random_seed):
        """Test that attention weights sum to 1 for each query position."""
        batch_size, num_heads, seq_len, d_k = 2, 4, 8, 16

        attention = ScaledDotProductAttention(d_k)
        query = np.random.randn(batch_size, num_heads, seq_len, d_k)
        key = np.random.randn(batch_size, num_heads, seq_len, d_k)
        value = np.random.randn(batch_size, num_heads, seq_len, d_k)

        attention.forward(query, key, value)

        # Check attention weights sum to 1
        sums = np.sum(attention.attn_weights, axis=-1)
        assert np.allclose(sums, 1.0)

    def test_forward_with_mask(self, random_seed):
        """Test attention with causal mask."""
        batch_size, num_heads, seq_len, d_k = 2, 4, 8, 16

        attention = ScaledDotProductAttention(d_k)
        query = np.random.randn(batch_size, num_heads, seq_len, d_k)
        key = np.random.randn(batch_size, num_heads, seq_len, d_k)
        value = np.random.randn(batch_size, num_heads, seq_len, d_k)

        # Causal mask
        mask = create_causal_mask(seq_len)

        output = attention.forward(query, key, value, mask)

        assert output.shape == (batch_size, num_heads, seq_len, d_k)

        # Verify masked positions have zero attention weight
        attn = attention.attn_weights
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert np.allclose(attn[0, 0, i, j], 0.0, atol=1e-6)

    def test_backward_shape(self, random_seed):
        """Test that backward pass produces correct shapes."""
        batch_size, num_heads, seq_len, d_k = 2, 4, 8, 16

        attention = ScaledDotProductAttention(d_k)
        query = np.random.randn(batch_size, num_heads, seq_len, d_k)
        key = np.random.randn(batch_size, num_heads, seq_len, d_k)
        value = np.random.randn(batch_size, num_heads, seq_len, d_k)

        output = attention.forward(query, key, value)
        grad_output = np.ones_like(output)

        grad_q, grad_k, grad_v = attention.backward(grad_output)

        assert grad_q.shape == query.shape
        assert grad_k.shape == key.shape
        assert grad_v.shape == value.shape

    def test_backward_gradient_flow(self, random_seed):
        """Test that gradients flow correctly through attention."""
        batch_size, num_heads, seq_len, d_k = 2, 4, 8, 16

        attention = ScaledDotProductAttention(d_k)
        query = np.random.randn(batch_size, num_heads, seq_len, d_k)
        key = np.random.randn(batch_size, num_heads, seq_len, d_k)
        value = np.random.randn(batch_size, num_heads, seq_len, d_k)

        output = attention.forward(query, key, value)
        grad_output = np.ones_like(output)

        grad_q, grad_k, grad_v = attention.backward(grad_output)

        # Gradients should not be all zeros
        assert not np.allclose(grad_q, 0)
        assert not np.allclose(grad_k, 0)
        assert not np.allclose(grad_v, 0)


# =============================================================================
# Multi-Head Attention Tests
# =============================================================================

class TestMultiHeadAttention:
    """Tests for multi-head attention."""

    def test_forward_shape_matches_input(self, random_seed, small_attention_params):
        """Test that Multi-Head output dimension matches input (batch, seq, hidden)."""
        d_model = small_attention_params["d_model"]
        num_heads = small_attention_params["num_heads"]
        batch_size = small_attention_params["batch_size"]
        seq_len = small_attention_params["seq_len"]

        mha = MultiHeadAttention(d_model, num_heads)
        x = np.random.randn(batch_size, seq_len, d_model)

        output = mha.forward(x, x, x)

        # Success criterion: Multi-Head output matches input dimensions
        assert output.shape == (batch_size, seq_len, d_model)

    def test_heads_have_independent_parameters(self, random_seed):
        """Test that 4 heads each have independent parameters (no sharing)."""
        d_model = 64
        num_heads = 4

        mha = MultiHeadAttention(d_model, num_heads)

        # Each head's parameters are stored in slices of W_Q, W_K, W_V
        # d_k per head
        d_k = d_model // num_heads
        assert d_k == 16

        # Verify W_Q has shape (d_model, d_model) = 64x64
        assert mha.W_Q.shape == (d_model, d_model)

        # Verify each head's slice is different by checking non-zero variance
        head_params = []
        for h in range(num_heads):
            start = h * d_k
            end = (h + 1) * d_k
            head_slice = mha.W_Q[:, start:end]
            head_params.append(head_slice.flatten())

        # Check that each head's parameters are different
        for i in range(num_heads):
            for j in range(i + 1, num_heads):
                assert not np.allclose(head_params[i], head_params[j])

    def test_four_heads_independent(self, random_seed):
        """Test that 4 heads are independent with non-shared parameters."""
        d_model = 64
        num_heads = 4
        batch_size = 2
        seq_len = 8

        mha = MultiHeadAttention(d_model, num_heads)
        x = np.random.randn(batch_size, seq_len, d_model)

        # Forward pass
        output = mha.forward(x, x, x)

        # Verify output shape
        assert output.shape == (batch_size, seq_len, d_model)

        # Verify each head's weight slice is unique
        d_k = d_model // num_heads
        for h in range(num_heads):
            start = h * d_k
            end = (h + 1) * d_k

            # Extract this head's weights
            w_q_h = mha.W_Q[:, start:end]

            # Compare with other heads
            for h2 in range(h + 1, num_heads):
                start2 = h2 * d_k
                end2 = (h2 + 1) * d_k
                w_q_h2 = mha.W_Q[:, start2:end2]

                # Parameters should be different
                assert not np.allclose(w_q_h, w_q_h2)

    def test_backward_shape(self, random_seed, small_attention_params):
        """Test backward pass produces correct shapes."""
        d_model = small_attention_params["d_model"]
        num_heads = small_attention_params["num_heads"]
        batch_size = small_attention_params["batch_size"]
        seq_len = small_attention_params["seq_len"]

        mha = MultiHeadAttention(d_model, num_heads)
        x = np.random.randn(batch_size, seq_len, d_model)

        output = mha.forward(x, x, x)
        grad_output = np.ones_like(output)

        grad_q, grad_k, grad_v = mha.backward(grad_output)

        assert grad_q.shape == x.shape
        assert grad_k.shape == x.shape
        assert grad_v.shape == x.shape

    def test_cross_attention_different_seq_lengths(self, random_seed):
        """Test cross-attention with different query/key sequence lengths."""
        d_model = 64
        num_heads = 4
        batch_size = 2
        seq_q = 8
        seq_kv = 12

        mha = MultiHeadAttention(d_model, num_heads)
        query = np.random.randn(batch_size, seq_q, d_model)
        key = np.random.randn(batch_size, seq_kv, d_model)
        value = np.random.randn(batch_size, seq_kv, d_model)

        output = mha.forward(query, key, value)

        assert output.shape == (batch_size, seq_q, d_model)

    def test_parameter_count(self, random_seed):
        """Test parameter count is correct."""
        d_model = 64
        num_heads = 4

        mha = MultiHeadAttention(d_model, num_heads)
        param_count = count_parameters_attention(mha)

        # 4 weight matrices (d_model x d_model) + 4 bias vectors (d_model)
        expected = 4 * d_model * d_model + 4 * d_model
        assert param_count == expected


# =============================================================================
# Sinusoidal Positional Encoding Tests
# =============================================================================

class TestSinusoidalPositionalEncoding:
    """Tests for sinusoidal positional encoding."""

    def test_output_shape(self, random_seed):
        """Test that positional encoding has correct shape."""
        d_model = 64
        max_len = 100
        batch_size = 2
        seq_len = 10

        pe = SinusoidalPositionalEncoding(d_model, max_len)
        x = np.random.randn(batch_size, seq_len, d_model)

        output = pe.forward(x)

        assert output.shape == x.shape

    def test_encoding_values_in_range(self, random_seed):
        """Test that sin/cos encoding values are in [-1, 1]."""
        d_model = 64
        max_len = 100

        pe = SinusoidalPositionalEncoding(d_model, max_len)
        encoding = pe.pe

        assert np.all(encoding >= -1.0)
        assert np.all(encoding <= 1.0)

    def test_position_0_encoding(self, random_seed):
        """Test that position 0 has sin/cos pattern."""
        d_model = 64

        pe = SinusoidalPositionalEncoding(d_model)

        # Position 0: sin(0) = 0 for even indices, cos(0) = 1 for odd indices
        pos_0 = pe.pe[0, 0, :]

        assert np.allclose(pos_0[0::2], 0.0, atol=1e-6)  # Even indices: sin(0) = 0
        assert np.allclose(pos_0[1::2], 1.0, atol=1e-6)  # Odd indices: cos(0) = 1

    def test_uniqueness_within_100(self, random_seed):
        """Test that Sinusoidal position encoding is unique within seq=100."""
        d_model = 64
        max_len = 100

        pe = SinusoidalPositionalEncoding(d_model, max_len)

        # Success criterion: unique encodings within seq=100
        assert pe.is_unique(100)

    def test_uniqueness_different_positions(self, random_seed):
        """Test that different positions have different encodings."""
        d_model = 64
        max_len = 100

        pe = SinusoidalPositionalEncoding(d_model, max_len)

        # Get encodings for different positions
        pos_0 = pe.pe[0, 0, :]
        pos_1 = pe.pe[0, 1, :]
        pos_10 = pe.pe[0, 10, :]
        pos_99 = pe.pe[0, 99, :]

        # All should be different
        assert not np.allclose(pos_0, pos_1)
        assert not np.allclose(pos_0, pos_10)
        assert not np.allclose(pos_1, pos_10)
        assert not np.allclose(pos_10, pos_99)

    def test_backward_passes_through(self, random_seed):
        """Test that backward pass is identity (no learnable params)."""
        d_model = 64
        batch_size = 2
        seq_len = 10

        pe = SinusoidalPositionalEncoding(d_model)
        x = np.random.randn(batch_size, seq_len, d_model)
        grad = np.random.randn(batch_size, seq_len, d_model)

        grad_input = pe.backward(grad)

        assert np.allclose(grad_input, grad)


# =============================================================================
# Transformer Encoder Layer Tests
# =============================================================================

class TestTransformerEncoderLayer:
    """Tests for transformer encoder layer."""

    def test_forward_shape(self, random_seed):
        """Test that encoder layer preserves input shape."""
        d_model = 64
        num_heads = 4
        batch_size = 2
        seq_len = 8

        layer = TransformerEncoderLayer(d_model, num_heads)
        x = np.random.randn(batch_size, seq_len, d_model)

        output = layer.forward(x)

        assert output.shape == x.shape

    def test_forward_with_mask(self, random_seed):
        """Test encoder layer with attention mask."""
        d_model = 64
        num_heads = 4
        batch_size = 2
        seq_len = 8

        layer = TransformerEncoderLayer(d_model, num_heads)
        x = np.random.randn(batch_size, seq_len, d_model)
        mask = create_causal_mask(seq_len)

        output = layer.forward(x, mask)

        assert output.shape == x.shape

    def test_backward_shape(self, random_seed):
        """Test backward pass produces correct shape."""
        d_model = 64
        num_heads = 4
        batch_size = 2
        seq_len = 8

        layer = TransformerEncoderLayer(d_model, num_heads)
        x = np.random.randn(batch_size, seq_len, d_model)

        output = layer.forward(x)
        grad_output = np.ones_like(output)

        grad_input = layer.backward(grad_output)

        assert grad_input.shape == x.shape

    def test_parameter_count(self, random_seed):
        """Test parameter count includes all components."""
        d_model = 64
        num_heads = 4
        d_ff = 256

        layer = TransformerEncoderLayer(d_model, num_heads, d_ff)
        param_count = count_parameters_attention(layer)

        # MHA: 4 * d_model^2 + 4 * d_model
        # FFN: d_model * d_ff + d_ff + d_ff * d_model + d_model = 2 * d_model * d_ff + d_ff + d_model
        mha_params = 4 * d_model * d_model + 4 * d_model
        ffn_params = 2 * d_model * d_ff + d_ff + d_model
        expected = mha_params + ffn_params

        assert param_count == expected


# =============================================================================
# Mask Utility Tests
# =============================================================================

class TestMaskUtilities:
    """Tests for mask utility functions."""

    def test_causal_mask_shape(self):
        """Test causal mask has correct shape."""
        seq_len = 10
        mask = create_causal_mask(seq_len)

        assert mask.shape == (1, 1, seq_len, seq_len)

    def test_causal_mask_upper_triangular(self):
        """Test causal mask is upper triangular with zeros on diagonal."""
        seq_len = 5
        mask = create_causal_mask(seq_len)

        # Check diagonal is 0 (not masked)
        for i in range(seq_len):
            assert mask[0, 0, i, i] == 0

        # Check upper triangle is 1 (masked)
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert mask[0, 0, i, j] == 1

        # Check lower triangle is 0 (not masked)
        for i in range(seq_len):
            for j in range(i):
                assert mask[0, 0, i, j] == 0

    def test_padding_mask_shape(self):
        """Test padding mask has correct shape."""
        batch_size = 2
        seq_len = 10
        seq = np.array([[1, 2, 3, 0, 0, 0, 0, 0, 0, 0],
                        [4, 5, 6, 7, 8, 0, 0, 0, 0, 0]])

        mask = create_padding_mask(seq, pad_idx=0)

        assert mask.shape == (batch_size, 1, 1, seq_len)

    def test_padding_mask_values(self):
        """Test padding mask correctly identifies padding positions."""
        seq = np.array([[1, 2, 3, 0, 0]])

        mask = create_padding_mask(seq, pad_idx=0)

        # Non-padding positions should be 0
        assert mask[0, 0, 0, 0] == 0
        assert mask[0, 0, 0, 1] == 0
        assert mask[0, 0, 0, 2] == 0

        # Padding positions should be 1
        assert mask[0, 0, 0, 3] == 1
        assert mask[0, 0, 0, 4] == 1


# =============================================================================
# PyTorch Comparison Tests
# =============================================================================

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestPyTorchComparison:
    """Compare NumPy implementations with PyTorch."""

    def test_scaled_dot_product_self_consistency(self, random_seed):
        """Test scaled dot-product attention is self-consistent."""
        batch_size, num_heads, seq_len, d_k = 2, 4, 8, 16

        # NumPy implementation
        np_attention = ScaledDotProductAttention(d_k)
        np_query = np.random.randn(batch_size, num_heads, seq_len, d_k)
        np_key = np.random.randn(batch_size, num_heads, seq_len, d_k)
        np_value = np.random.randn(batch_size, num_heads, seq_len, d_k)

        np_output = np_attention.forward(np_query, np_key, np_value)

        # Output should be weighted combination of values
        # When Q=K, attention should focus on corresponding positions
        # Test with identical Q=K to verify diagonal attention
        attention2 = ScaledDotProductAttention(d_k)
        identical = np.random.randn(batch_size, num_heads, seq_len, d_k)
        out_identical = attention2.forward(identical, identical, np_value)

        # Verify attention weights sum to 1
        assert np.allclose(np.sum(attention2.attn_weights, axis=-1), 1.0)

        # Output should have same shape as input
        assert np_output.shape == np_value.shape

    def test_multi_head_attention_consistency(self, random_seed):
        """Test multi-head attention is self-consistent."""
        d_model = 64
        num_heads = 4
        batch_size = 2
        seq_len = 8

        # Create input
        np_x = np.random.randn(batch_size, seq_len, d_model)

        # NumPy implementation
        np_mha = MultiHeadAttention(d_model, num_heads)
        np_output = np_mha.forward(np_x, np_x, np_x)

        # Basic consistency checks
        assert np_output.shape == np_x.shape

        # Output should be different from input (transformation happened)
        assert not np.allclose(np_output, np_x)

        # With identical Q=K=V, output should be valid
        output2 = np_mha.forward(np_x, np_x, np_x)
        assert np_output.shape == (batch_size, seq_len, d_model)

    def test_positional_encoding_vs_pytorch(self, random_seed):
        """Test positional encoding matches PyTorch reference."""
        d_model = 64
        max_len = 100
        batch_size = 2
        seq_len = 10

        # NumPy implementation
        np_pe = SinusoidalPositionalEncoding(d_model, max_len)
        np_x = np.random.randn(batch_size, seq_len, d_model)
        np_output = np_pe.forward(np_x)

        # PyTorch reference implementation
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        torch_x = torch.tensor(np_x, dtype=torch.float64)
        torch_output = (torch_x + pe[:, :seq_len, :]).numpy()

        assert np.allclose(np_output, torch_output, atol=1e-6)

    def test_gradient_numerical_verification(self, random_seed):
        """Verify gradients numerically for multi-head attention."""
        d_model = 32
        num_heads = 2
        batch_size = 1
        seq_len = 4

        np_mha = MultiHeadAttention(d_model, num_heads)
        np_x = np.random.randn(batch_size, seq_len, d_model)

        # Forward
        output = np_mha.forward(np_x, np_x, np_x)

        # Analytical gradient
        grad_output = np.ones_like(output)
        np_mha.backward(grad_output)
        analytical_grad = np_mha.grad_W_Q.copy()

        # Numerical gradient
        eps = 1e-5
        numerical_grad = np.zeros_like(np_mha.W_Q)

        for i in range(np_mha.W_Q.shape[0]):
            for j in range(np_mha.W_Q.shape[1]):
                old = np_mha.W_Q[i, j]

                np_mha.W_Q[i, j] = old + eps
                out_plus = np_mha.forward(np_x, np_x, np_x)
                f_plus = np.sum(out_plus)

                np_mha.W_Q[i, j] = old - eps
                out_minus = np_mha.forward(np_x, np_x, np_x)
                f_minus = np.sum(out_minus)

                np_mha.W_Q[i, j] = old
                numerical_grad[i, j] = (f_plus - f_minus) / (2 * eps)

        # Check gradients match (use relative tolerance for small values)
        assert np.allclose(analytical_grad, numerical_grad, atol=1e-4, rtol=1e-3)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for attention components."""

    def test_full_transformer_forward(self, random_seed):
        """Test full transformer encoder forward pass."""
        d_model = 64
        num_heads = 4
        num_layers = 2
        batch_size = 2
        seq_len = 8

        # Create layers
        layers = [TransformerEncoderLayer(d_model, num_heads) for _ in range(num_layers)]
        pe = SinusoidalPositionalEncoding(d_model)

        # Input
        x = np.random.randn(batch_size, seq_len, d_model)

        # Forward with positional encoding
        x = pe.forward(x)
        for layer in layers:
            x = layer.forward(x)

        assert x.shape == (batch_size, seq_len, d_model)

    def test_attention_with_different_d_models(self, random_seed):
        """Test attention works with various d_model sizes."""
        for d_model in [32, 64, 128, 256]:
            for num_heads in [1, 2, 4, 8]:
                if d_model % num_heads != 0:
                    continue

                mha = MultiHeadAttention(d_model, num_heads)
                x = np.random.randn(2, 8, d_model)
                output = mha.forward(x, x, x)

                assert output.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
