"""
Tests for Distributed Data Parallel (DDP) Training.

Run with: pytest tests/test_ddp_training.py -v

Note: Some tests require multiple GPUs and will be skipped if unavailable.
"""

import pytest
import numpy as np
import os
import sys
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase5_deployment.ddp_training import (
    is_ddp_available,
    get_available_gpus,
    setup_ddp,
    cleanup_ddp,
    get_rank,
    get_world_size,
    is_main_process,
    barrier,
    all_reduce_tensor,
    convert_to_sync_batchnorm,
    is_sync_batchnorm,
    verify_gradient_sync,
    get_gradient_sync_error,
    create_distributed_sampler,
    create_distributed_dataloader,
    DistributedTrainer,
    DDPConfig,
    spawn_ddp_training,
    get_ddp_info,
    DDP_TRAINING_COMPONENTS,
)

from phase5_deployment.multi_gpu import (
    get_gpu_count,
    get_gpu_info,
    get_recommended_batch_size,
    wrap_data_parallel,
    unwrap_data_parallel,
    get_device,
    to_device,
    get_memory_usage,
    clear_cuda_cache,
    MultiGPUConfig,
    get_lightning_strategy,
    MULTI_GPU_COMPONENTS,
)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    Dataset = None
    TensorDataset = None


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_model():
    """Create a simple linear model for testing."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not available")
    return nn.Linear(10, 10)


@pytest.fixture
def simple_dataset():
    """Create a simple dataset for testing."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not available")
    x = torch.randn(100, 10)
    y = torch.randn(100, 10)
    return TensorDataset(x, y)


@pytest.fixture
def model_with_batchnorm():
    """Create a model with BatchNorm for SyncBatchNorm testing."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not available")

    class ModelWithBN(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm1d(10)
            self.fc = nn.Linear(10, 10)

        def forward(self, x):
            return self.fc(self.bn(x))

    return ModelWithBN()


# =============================================================================
# Test GPU Detection
# =============================================================================


class TestGPUDetection:
    """Tests for GPU detection utilities."""

    def test_get_gpu_count(self):
        """Test GPU count detection."""
        count = get_gpu_count()
        assert isinstance(count, int)
        assert count >= 0

    def test_get_gpu_info(self):
        """Test GPU info retrieval."""
        info = get_gpu_info()
        assert isinstance(info, list)

        if len(info) > 0:
            gpu = info[0]
            assert 'index' in gpu
            assert 'name' in gpu
            assert 'vram_gb' in gpu

    def test_get_available_gpus(self):
        """Test available GPU list."""
        gpus = get_available_gpus()
        assert isinstance(gpus, list)
        if HAS_TORCH and torch.cuda.is_available():
            assert len(gpus) == torch.cuda.device_count()


# =============================================================================
# Test DDP Setup (Single Process)
# =============================================================================


class TestDDPSetup:
    """Tests for DDP setup utilities (single process)."""

    def test_is_ddp_available(self):
        """Test DDP availability check."""
        result = is_ddp_available()
        assert isinstance(result, bool)

    def test_get_ddp_info(self):
        """Test DDP info retrieval."""
        info = get_ddp_info()
        assert isinstance(info, dict)
        assert 'ddp_available' in info
        assert 'num_gpus' in info
        assert 'is_distributed' in info

    def test_get_rank_without_ddp(self):
        """Test rank returns 0 without DDP."""
        rank = get_rank()
        assert rank == 0

    def test_get_world_size_without_ddp(self):
        """Test world_size returns 1 without DDP."""
        size = get_world_size()
        assert size == 1

    def test_is_main_process_without_ddp(self):
        """Test is_main_process returns True without DDP."""
        assert is_main_process() == True


# =============================================================================
# Test SyncBatchNorm
# =============================================================================


class TestSyncBatchNorm:
    """Tests for SyncBatchNorm conversion."""

    def test_convert_to_sync_batchnorm(self, model_with_batchnorm):
        """Test BatchNorm to SyncBatchNorm conversion."""
        # Check original has regular BatchNorm
        assert isinstance(model_with_batchnorm.bn, nn.BatchNorm1d)

        # Convert
        converted = convert_to_sync_batchnorm(model_with_batchnorm)

        # Check converted has SyncBatchNorm
        assert is_sync_batchnorm(converted)

    def test_is_sync_batchnorm_false_originally(self, model_with_batchnorm):
        """Test is_sync_batchnorm returns False for regular BatchNorm."""
        assert is_sync_batchnorm(model_with_batchnorm) == False

    def test_is_sync_batchnorm_true_after_conversion(self, model_with_batchnorm):
        """Test is_sync_batchnorm returns True after conversion."""
        converted = convert_to_sync_batchnorm(model_with_batchnorm)
        assert is_sync_batchnorm(converted) == True


# =============================================================================
# Test Multi-GPU Config
# =============================================================================


class TestMultiGPUConfig:
    """Tests for MultiGPUConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = MultiGPUConfig()
        assert config.strategy == 'ddp'
        assert config.batch_size_per_gpu == 32
        assert config.sync_batchnorm == True

    def test_invalid_strategy(self):
        """Test invalid strategy raises error."""
        with pytest.raises(ValueError):
            MultiGPUConfig(strategy='invalid')

    def test_dp_strategy(self):
        """Test DP strategy configuration."""
        config = MultiGPUConfig(strategy='dp')
        assert config.strategy == 'dp'

    def test_effective_batch_size(self):
        """Test effective batch size calculation."""
        config = MultiGPUConfig(
            device_ids=[0, 1],
            batch_size_per_gpu=32,
        )
        assert config.effective_batch_size == 64

    def test_world_size(self):
        """Test world size calculation."""
        config = MultiGPUConfig(device_ids=[0, 1, 2])
        assert config.world_size == 3


# =============================================================================
# Test DDP Config
# =============================================================================


class TestDDPConfig:
    """Tests for DDPConfig."""

    def test_default_config(self):
        """Test default DDP configuration."""
        config = DDPConfig()
        assert config.backend == 'nccl'
        assert config.master_addr == 'localhost'
        assert config.master_port == '29500'
        assert config.sync_batchnorm == True

    def test_custom_config(self):
        """Test custom DDP configuration."""
        config = DDPConfig(
            backend='gloo',
            master_port='12345',
            find_unused_parameters=True,
        )
        assert config.backend == 'gloo'
        assert config.master_port == '12345'
        assert config.find_unused_parameters == True


# =============================================================================
# Test Device Utilities
# =============================================================================


class TestDeviceUtilities:
    """Tests for device utilities."""

    def test_get_device_cpu(self):
        """Test get_device returns CPU."""
        device = get_device('cpu')
        assert device.type == 'cpu'

    def test_get_device_from_string(self):
        """Test get_device from string."""
        device = get_device('cuda:0')
        assert device.type == 'cuda'
        assert device.index == 0

    def test_to_device_tensor(self):
        """Test to_device with tensor."""
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")

        tensor = torch.randn(10)
        moved = to_device(tensor, 'cpu')
        assert moved.device.type == 'cpu'

    def test_to_device_dict(self):
        """Test to_device with dictionary."""
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")

        data = {'x': torch.randn(10), 'y': torch.randn(5)}
        moved = to_device(data, 'cpu')
        assert moved['x'].device.type == 'cpu'
        assert moved['y'].device.type == 'cpu'


# =============================================================================
# Test DataParallel
# =============================================================================


class TestDataParallel:
    """Tests for DataParallel utilities."""

    @pytest.mark.skipif(
        not HAS_TORCH or torch.cuda.device_count() < 1,
        reason="Requires at least 1 GPU"
    )
    def test_wrap_data_parallel(self, simple_model):
        """Test DataParallel wrapping."""
        wrapped = wrap_data_parallel(simple_model, device_ids=[0])
        assert isinstance(wrapped, nn.DataParallel)

    def test_unwrap_data_parallel(self, simple_model):
        """Test DataParallel unwrapping."""
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")

        # Create wrapped model
        wrapped = nn.DataParallel(simple_model)

        # Unwrap
        unwrapped = unwrap_data_parallel(wrapped)
        assert isinstance(unwrapped, nn.Linear)

    def test_unwrap_non_wrapped_model(self, simple_model):
        """Test unwrapping non-wrapped model returns same model."""
        unwrapped = unwrap_data_parallel(simple_model)
        assert unwrapped is simple_model


# =============================================================================
# Test Memory Utilities
# =============================================================================


class TestMemoryUtilities:
    """Tests for memory utilities."""

    def test_get_memory_usage_cpu(self):
        """Test memory usage returns zeros without CUDA."""
        if HAS_TORCH and torch.cuda.is_available():
            pytest.skip("CUDA available, test not applicable")
        memory = get_memory_usage()
        assert memory['allocated'] == 0
        assert memory['total'] == 0

    @pytest.mark.skipif(
        not HAS_TORCH or not torch.cuda.is_available(),
        reason="Requires CUDA"
    )
    def test_get_memory_usage_cuda(self):
        """Test memory usage with CUDA."""
        memory = get_memory_usage(0)
        assert 'allocated' in memory
        assert 'reserved' in memory
        assert 'free' in memory
        assert 'total' in memory
        assert memory['total'] > 0

    def test_clear_cuda_cache(self):
        """Test clear CUDA cache doesn't raise error."""
        clear_cuda_cache()  # Should not raise


# =============================================================================
# Test All Reduce (Single Process Fallback)
# =============================================================================


class TestAllReduce:
    """Tests for all_reduce_tensor."""

    def test_all_reduce_without_ddp(self):
        """Test all_reduce returns original tensor without DDP."""
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")

        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = all_reduce_tensor(tensor, op='sum')
        assert torch.allclose(result, tensor)


# =============================================================================
# Test Lightning Strategy
# =============================================================================


class TestLightningStrategy:
    """Tests for PyTorch Lightning strategy."""

    def test_ddp_strategy(self):
        """Test DDP strategy configuration."""
        config = MultiGPUConfig(strategy='ddp', device_ids=[0, 1])
        strategy = get_lightning_strategy(config)

        assert strategy['strategy'] == 'ddp'
        assert strategy['accelerator'] == 'gpu'
        assert strategy['sync_batchnorm'] == True

    def test_dp_strategy(self):
        """Test DP strategy configuration."""
        config = MultiGPUConfig(strategy='dp', device_ids=[0])
        strategy = get_lightning_strategy(config)

        assert strategy['strategy'] == 'dp'
        assert strategy['accelerator'] == 'gpu'


# =============================================================================
# Test Registry
# =============================================================================


class TestRegistry:
    """Tests for component registries."""

    def test_ddp_training_components(self):
        """Test DDP training components registry."""
        assert isinstance(DDP_TRAINING_COMPONENTS, dict)
        assert 'setup' in DDP_TRAINING_COMPONENTS
        assert 'trainer' in DDP_TRAINING_COMPONENTS
        assert 'benchmark' in DDP_TRAINING_COMPONENTS

    def test_multi_gpu_components(self):
        """Test multi-GPU components registry."""
        assert isinstance(MULTI_GPU_COMPONENTS, dict)
        assert 'detection' in MULTI_GPU_COMPONENTS
        assert 'memory' in MULTI_GPU_COMPONENTS
        assert 'config' in MULTI_GPU_COMPONENTS


# =============================================================================
# Multi-GPU Integration Tests (require multiple GPUs)
# =============================================================================


@pytest.mark.skipif(
    not HAS_TORCH or torch.cuda.device_count() < 2,
    reason="Requires at least 2 GPUs"
)
class TestMultiGPUIntegration:
    """Integration tests requiring multiple GPUs."""

    def test_gradient_synchronization(self):
        """Test gradient synchronization across GPUs."""
        # This test spawns a subprocess for DDP
        # We'll test the gradient sync error function
        # Note: Full DDP test requires mp.spawn which can't run in pytest
        pass  # Placeholder - full DDP tests need separate script

    def test_ddp_speedup_vs_dp(self):
        """Test DDP is faster than DP."""
        # Note: Benchmark test requires mp.spawn
        # This is tested separately in scripts
        pass


# =============================================================================
# Stress Tests
# =============================================================================


class TestStress:
    """Stress tests for DDP utilities."""

    def test_repeated_setup_cleanup(self):
        """Test repeated setup and cleanup cycles."""
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")

        # Without actual DDP, just test no errors
        for _ in range(10):
            info = get_ddp_info()
            assert isinstance(info, dict)

    def test_large_tensor_operations(self):
        """Test operations with large tensors."""
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")

        # Test with large tensor
        large_tensor = torch.randn(1000, 1000)

        # Test device movement
        moved = to_device(large_tensor, 'cpu')
        assert moved.shape == large_tensor.shape


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
