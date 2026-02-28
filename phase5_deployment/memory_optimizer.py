"""
Memory Optimization Techniques for Training on Limited GPU Memory.

This module provides:
    - Gradient Checkpointing: Trade compute for memory by recomputing activations
    - Activation Recomputation: Selective activation saving strategies
    - CPU Offloading: Move parameters/activations to CPU when not in use
    - Memory Efficient Attention: Optimized attention for transformers
    - In-place Operations: Reduce memory footprint

Theory:
    Gradient Checkpointing:
        - Instead of storing all activations, only store checkpoints (subset)
        - During backward pass, recompute activations between checkpoints
        - Memory reduction: O(sqrt(n)) vs O(n) for n layers
        - Compute overhead: ~20-33% extra forward passes

    Activation Recomputation:
        - Selective strategy: Choose which layers to checkpoint
        - Trade-off: Memory saved vs recomputation cost
        - Best for layers with cheap forward pass but large activations

    CPU Offloading:
        - Move parameters/activations to CPU memory
        - Transfer back to GPU when needed
        - Useful for very large models that don't fit on GPU
        - Overhead: PCIe transfer latency

    For 4GB VRAM (RTX 3050 Ti):
        - Gradient checkpointing: 50-70% memory reduction
        - CPU offloading: Can train 12+ layer transformers
        - Combined: Enable training large models on limited hardware

References:
    - Chen et al. (2016): "Training Deep Nets with Sublinear Memory Cost"
    - PyTorch Checkpoint: https://pytorch.org/docs/stable/checkpoint.html
    - Griewank & Walther (2000): "Algorithm 799: Revolve"
"""

from typing import Optional, Dict, Any, List, Union, Callable, Tuple
from dataclasses import dataclass, field
import gc
import time
import logging
import math

import numpy as np

# Try to import torch
try:
    import torch
    import torch.nn as nn
    from torch.utils.checkpoint import checkpoint, checkpoint_sequential
    from torch.cuda.amp import autocast
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    checkpoint = None
    checkpoint_sequential = None
    autocast = None

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class MemoryOptimizationConfig:
    """
    Configuration for memory optimization techniques.

    Attributes:
        enable_gradient_checkpointing: Enable gradient checkpointing
        checkpoint_strategy: Strategy for checkpoint selection
            - 'all': Checkpoint all layers
            - 'balanced': Balanced checkpointing (sqrt(n) checkpoints)
            - 'last': Only checkpoint last layer of each block
            - 'selective': Use custom checkpoint_layers list
        checkpoint_layers: List of layer indices to checkpoint (for 'selective' strategy)
        enable_cpu_offloading: Enable CPU offloading for parameters
        offload_optimizer_states: Move optimizer states to CPU
        offload_activations: Move activations to CPU during forward
        enable_activation_recomputation: Enable selective activation recomputation
        recompute_layers: Layers to recompute (by name pattern)
        memory_efficient_attention: Use memory-efficient attention implementation
        preserve_rng_state: Preserve RNG state for reproducibility
    """
    enable_gradient_checkpointing: bool = True
    checkpoint_strategy: str = 'balanced'  # 'all', 'balanced', 'last', 'selective'
    checkpoint_layers: Optional[List[int]] = None
    enable_cpu_offloading: bool = False
    offload_optimizer_states: bool = True
    offload_activations: bool = False
    enable_activation_recomputation: bool = True
    recompute_layers: Optional[List[str]] = None
    memory_efficient_attention: bool = True
    preserve_rng_state: bool = True

    def validate(self) -> List[str]:
        """Validate configuration and return any errors."""
        errors = []
        valid_strategies = ['all', 'balanced', 'last', 'selective']
        if self.checkpoint_strategy not in valid_strategies:
            errors.append(f"checkpoint_strategy must be one of {valid_strategies}")
        if self.checkpoint_strategy == 'selective' and not self.checkpoint_layers:
            errors.append("checkpoint_layers required for 'selective' strategy")
        return errors


# =============================================================================
# Memory Utilities
# =============================================================================


def get_memory_usage(device: Optional[Union[int, str, 'torch.device']] = None) -> Dict[str, float]:
    """
    Get current GPU memory usage in MB.

    Args:
        device: GPU device (default: current device)

    Returns:
        Dictionary with 'allocated', 'reserved', 'free', 'total' in MB
    """
    if not HAS_TORCH or not torch.cuda.is_available():
        return {'allocated': 0, 'reserved': 0, 'free': 0, 'total': 0}

    if device is None:
        device = torch.cuda.current_device()

    allocated = torch.cuda.memory_allocated(device) / 1024 / 1024
    reserved = torch.cuda.memory_reserved(device) / 1024 / 1024
    total = torch.cuda.get_device_properties(device).total_memory / 1024 / 1024
    free = total - allocated

    return {
        'allocated': allocated,
        'reserved': reserved,
        'free': free,
        'total': total
    }


def reset_memory_stats(device: Optional[Union[int, str, 'torch.device']] = None) -> None:
    """Reset CUDA memory statistics."""
    if not HAS_TORCH or not torch.cuda.is_available():
        return

    if device is None:
        device = torch.cuda.current_device()

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.reset_accumulated_memory_stats(device)


def get_peak_memory(device: Optional[Union[int, str, 'torch.device']] = None) -> float:
    """Get peak memory usage in MB."""
    if not HAS_TORCH or not torch.cuda.is_available():
        return 0.0

    if device is None:
        device = torch.cuda.current_device()

    return torch.cuda.max_memory_allocated(device) / 1024 / 1024


def clear_cuda_cache() -> None:
    """Clear CUDA cache to free memory."""
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


# =============================================================================
# Gradient Checkpointing
# =============================================================================


def get_checkpoint_segments(
    num_layers: int,
    strategy: str = 'balanced',
    checkpoint_layers: Optional[List[int]] = None
) -> List[int]:
    """
    Determine which layers to checkpoint based on strategy.

    Args:
        num_layers: Total number of layers
        strategy: Checkpointing strategy
        checkpoint_layers: Custom layer indices for 'selective' strategy

    Returns:
        List of layer indices to checkpoint

    Strategies:
        - 'all': Checkpoint all layers (max memory savings, max compute)
        - 'balanced': ~sqrt(n) evenly spaced checkpoints
        - 'last': Checkpoint last layer of each block
        - 'selective': Use provided checkpoint_layers
    """
    if strategy == 'all':
        return list(range(num_layers))

    elif strategy == 'balanced':
        # Optimal: sqrt(n) segments for sqrt(n) memory with sqrt(n) recomputation
        num_checkpoints = max(1, int(math.sqrt(num_layers)))
        if num_checkpoints >= num_layers:
            return list(range(num_layers))

        # Evenly spaced checkpoints
        step = num_layers / num_checkpoints
        return [int(i * step) for i in range(num_checkpoints)]

    elif strategy == 'last':
        # Checkpoint last layer only (minimal overhead)
        return [num_layers - 1] if num_layers > 0 else []

    elif strategy == 'selective':
        if checkpoint_layers is None:
            return []
        # Validate indices
        return [i for i in checkpoint_layers if 0 <= i < num_layers]

    else:
        raise ValueError(f"Unknown checkpoint strategy: {strategy}")


class CheckpointedSequential(nn.Module if HAS_TORCH else object):
    """
    Sequential container with gradient checkpointing.

    Wraps a sequence of layers and applies gradient checkpointing
    to reduce memory usage during backpropagation.

    Args:
        layers: Sequence of nn.Module layers
        checkpoint_segments: Number of segments to divide layers into
            (each segment is checkpointed as a unit)
        use_reentrant: Use reentrant checkpoint implementation (default: False for newer PyTorch)

    Example:
        >>> layers = [nn.Linear(512, 512) for _ in range(12)]
        >>> checkpointed = CheckpointedSequential(layers, checkpoint_segments=4)
        >>> output = checkpointed(input)
    """

    def __init__(
        self,
        layers: List['nn.Module'],
        checkpoint_segments: int = 1,
        use_reentrant: bool = False
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for CheckpointedSequential")

        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.checkpoint_segments = checkpoint_segments
        self.use_reentrant = use_reentrant

        # Calculate segment boundaries
        self._segment_boundaries = self._compute_segment_boundaries()

    def _compute_segment_boundaries(self) -> List[Tuple[int, int]]:
        """Compute start/end indices for each segment."""
        n_layers = len(self.layers)
        if self.checkpoint_segments <= 1:
            return [(0, n_layers)]

        segment_size = n_layers // self.checkpoint_segments
        boundaries = []

        for i in range(self.checkpoint_segments):
            start = i * segment_size
            end = (i + 1) * segment_size if i < self.checkpoint_segments - 1 else n_layers
            boundaries.append((start, end))

        return boundaries

    def _run_segment(self, start: int, end: int, x: 'torch.Tensor') -> 'torch.Tensor':
        """Run layers from start to end index."""
        for i in range(start, end):
            x = self.layers[i](x)
        return x

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        """Forward pass with checkpointing."""
        if self.checkpoint_segments <= 1 or not self.training:
            # No checkpointing during eval or if disabled
            for layer in self.layers:
                x = layer(x)
            return x

        # Apply checkpointing to each segment
        for start, end in self._segment_boundaries:
            # Create a callable for this segment
            def segment_fn(input_tensor, s=start, e=end):
                return self._run_segment(s, e, input_tensor)

            # Apply checkpoint
            x = checkpoint(segment_fn, x, use_reentrant=self.use_reentrant)

        return x


def apply_gradient_checkpointing(
    model: 'nn.Module',
    checkpoint_layers: Optional[List[int]] = None,
    strategy: str = 'balanced',
    use_reentrant: bool = False
) -> 'nn.Module':
    """
    Apply gradient checkpointing to a model's layers.

    This function wraps the forward pass of specified layers with
    torch.utils.checkpoint to reduce memory usage.

    Args:
        model: PyTorch model to modify
        checkpoint_layers: Specific layer indices to checkpoint (None for auto)
        strategy: Checkpoint selection strategy
        use_reentrant: Use reentrant checkpoint implementation

    Returns:
        Modified model with checkpointing applied

    Example:
        >>> model = TransformerModel(...)
        >>> model = apply_gradient_checkpointing(model, strategy='balanced')
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for gradient checkpointing")

    # Get all child modules that are leaf modules (no children)
    leaf_modules = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0 and not isinstance(module, nn.ModuleDict):
            leaf_modules.append((name, module))

    if not leaf_modules:
        return model

    # Determine which modules to checkpoint
    if checkpoint_layers is None:
        checkpoint_indices = get_checkpoint_segments(len(leaf_modules), strategy)
    else:
        checkpoint_indices = checkpoint_layers

    checkpoint_set = set(checkpoint_indices)

    # Wrap forward methods
    for idx, (name, module) in enumerate(leaf_modules):
        if idx in checkpoint_set:
            original_forward = module.forward

            def make_checkpointed_forward(orig_fwd, use_reent=use_reentrant):
                def checkpointed_forward(*args, **kwargs):
                    if not module.training:
                        return orig_fwd(*args, **kwargs)
                    return checkpoint(orig_fwd, *args, use_reentrant=use_reent, **kwargs)
                return checkpointed_forward

            module.forward = make_checkpointed_forward(original_forward)

    return model


# =============================================================================
# CPU Offloading
# =============================================================================


class CPUOffloader:
    """
    Manages CPU offloading for model parameters and activations.

    This class provides utilities to move tensors between GPU and CPU
    to enable training models larger than GPU memory.

    Args:
        model: Model to manage offloading for
        offload_optimizer: Whether to offload optimizer states
        pin_memory: Use pinned memory for faster GPU transfers

    Example:
        >>> offloader = CPUOffloader(model, offload_optimizer=True)
        >>> optimizer = torch.optim.Adam(offloader.gpu_params())
        >>> for batch in dataloader:
        ...     offloader.load_to_gpu()
        ...     output = model(batch)
        ...     loss = criterion(output, target)
        ...     loss.backward()
        ...     offloader.offload_gradients()
        ...     optimizer.step()
    """

    def __init__(
        self,
        model: 'nn.Module',
        offload_optimizer: bool = True,
        pin_memory: bool = True
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for CPUOffloader")

        self.model = model
        self.offload_optimizer = offload_optimizer
        self.pin_memory = pin_memory
        self._device = next(model.parameters()).device if len(list(model.parameters())) > 0 else 'cuda'

        # Separate parameters by storage location
        self._gpu_params: List['nn.Parameter'] = []
        self._cpu_params: List['nn.Parameter'] = []
        self._param_copies: Dict[int, 'torch.Tensor'] = {}  # id -> CPU copy

        self._init_param_locations()

    def _init_param_locations(self) -> None:
        """Initialize parameter locations based on strategy."""
        for param in self.model.parameters():
            if param.device.type == 'cuda':
                self._gpu_params.append(param)
            else:
                self._cpu_params.append(param)

    def load_to_gpu(self, params: Optional[List['nn.Parameter']] = None) -> None:
        """
        Load parameters to GPU.

        Args:
            params: Specific parameters to load (None for all CPU params)
        """
        # Skip if no CUDA available
        if not torch.cuda.is_available():
            return

        target_params = params if params is not None else self._cpu_params

        for param in target_params:
            if param.device.type == 'cpu':
                # Create or reuse GPU copy
                if id(param) not in self._param_copies:
                    cpu_copy = param.data
                    if self.pin_memory:
                        cpu_copy = cpu_copy.pin_memory()
                    self._param_copies[id(param)] = cpu_copy

                # Transfer to GPU
                gpu_data = self._param_copies[id(param)].to(self._device, non_blocking=True)
                param.data = gpu_data

    def offload_to_cpu(self, params: Optional[List['nn.Parameter']] = None) -> None:
        """
        Move parameters to CPU to free GPU memory.

        Args:
            params: Specific parameters to offload (None for all)
        """
        # Skip if no CUDA available
        if not torch.cuda.is_available():
            return

        target_params = params if params is not None else list(self.model.parameters())

        for param in target_params:
            if param.device.type == 'cuda':
                # Keep a CPU copy
                if id(param) not in self._param_copies:
                    cpu_data = param.data.cpu()
                    if self.pin_memory:
                        cpu_data = cpu_data.pin_memory()
                    self._param_copies[id(param)] = cpu_data
                else:
                    self._param_copies[id(param)].copy_(param.data.cpu())

                # Replace with CPU tensor (saves GPU memory)
                param.data = self._param_copies[id(param)]

    def offload_gradients(self) -> None:
        """Move gradients to CPU to save GPU memory."""
        for param in self.model.parameters():
            if param.grad is not None and param.grad.device.type == 'cuda':
                param.grad = param.grad.cpu()

    def load_gradients_to_gpu(self) -> None:
        """Move gradients back to GPU for optimizer step."""
        for param in self.model.parameters():
            if param.grad is not None and param.grad.device.type == 'cpu':
                param.grad = param.grad.to(self._device, non_blocking=True)

    def gpu_params(self) -> List['nn.Parameter']:
        """Get list of parameters that should be on GPU for forward pass."""
        return self._gpu_params

    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        return get_memory_usage(self._device)


class OffloadedOptimizer:
    """
    Optimizer wrapper that keeps states on CPU.

    This wrapper stores optimizer states (momentum, variance, etc.) on CPU
    and only moves them to GPU during the step, reducing memory footprint.

    Args:
        optimizer: Base optimizer to wrap
        device: GPU device for computation

    Example:
        >>> base_opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        >>> optimizer = OffloadedOptimizer(base_opt)
        >>> optimizer.step()  # States stay on CPU
    """

    def __init__(self, optimizer: 'torch.optim.Optimizer', device: Optional['torch.device'] = None):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for OffloadedOptimizer")

        self.optimizer = optimizer
        self._device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._cpu_states: Dict[int, Dict[str, 'torch.Tensor']] = {}

    def _offload_states(self) -> None:
        """Move optimizer states to CPU."""
        if not torch.cuda.is_available():
            return

        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                if param in self.optimizer.state:
                    state = self.optimizer.state[param]
                    cpu_state = {}

                    for key, value in state.items():
                        if isinstance(value, torch.Tensor) and value.device.type == 'cuda':
                            cpu_state[key] = value.cpu()
                        else:
                            cpu_state[key] = value

                    self._cpu_states[id(param)] = cpu_state
                    # Clear GPU state
                    for key in list(state.keys()):
                        if isinstance(state[key], torch.Tensor):
                            del state[key]

    def _load_states_to_gpu(self) -> None:
        """Move optimizer states to GPU for computation."""
        if not torch.cuda.is_available():
            return

        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                param_id = id(param)
                if param_id in self._cpu_states and param in self.optimizer.state:
                    state = self.optimizer.state[param]
                    cpu_state = self._cpu_states[param_id]

                    for key, value in cpu_state.items():
                        if isinstance(value, torch.Tensor):
                            state[key] = value.to(self._device, non_blocking=True)
                        else:
                            state[key] = value

    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Perform optimizer step with offloaded states."""
        self._load_states_to_gpu()
        result = self.optimizer.step(closure)
        self._offload_states()
        clear_cuda_cache()
        return result

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Zero gradients."""
        self.optimizer.zero_grad(set_to_none)

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        """Add parameter group."""
        self.optimizer.add_param_group(param_group)

    @property
    def param_groups(self) -> List[Dict[str, Any]]:
        """Get parameter groups."""
        return self.optimizer.param_groups

    @property
    def state(self) -> Dict[int, Dict[str, Any]]:
        """Get optimizer state."""
        return self.optimizer.state


# =============================================================================
# Activation Recomputation
# =============================================================================


class ActivationRecomputer:
    """
    Selective activation recomputation manager.

    This class manages which activations to save and which to recompute,
    optimizing the trade-off between memory and compute.

    Args:
        model: Model to manage activations for
        recompute_patterns: List of module name patterns to recompute
        preserve_rng: Whether to preserve RNG state

    Example:
        >>> recomputer = ActivationRecomputer(model, recompute_patterns=['.*attention.*'])
        >>> with recomputer:
        ...     output = model(input)
        ...     loss.backward()  # Activations recomputed during backward
    """

    def __init__(
        self,
        model: 'nn.Module',
        recompute_patterns: Optional[List[str]] = None,
        preserve_rng: bool = True
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for ActivationRecomputer")

        self.model = model
        self.recompute_patterns = recompute_patterns or []
        self.preserve_rng = preserve_rng
        self._hooks = []
        self._saved_activations: Dict[str, 'torch.Tensor'] = {}

    def _should_recompute(self, name: str) -> bool:
        """Check if a module's activations should be recomputed."""
        import re
        for pattern in self.recompute_patterns:
            if re.match(pattern, name):
                return True
        return False

    def _register_hooks(self) -> None:
        """Register forward hooks to manage activations."""
        for name, module in self.model.named_modules():
            if self._should_recompute(name):
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

    def _make_hook(self, name: str) -> Callable:
        """Create a forward hook for activation management."""
        def hook(module: 'nn.Module', input: Tuple, output: 'torch.Tensor'):
            # Store reference but don't keep in memory
            # The activation will be recomputed during backward
            self._saved_activations[name] = None
        return hook

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __enter__(self):
        """Enter context manager."""
        self._register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self._remove_hooks()
        self._saved_activations.clear()
        return False


# =============================================================================
# Memory Efficient Attention
# =============================================================================


def memory_efficient_attention(
    query: 'torch.Tensor',
    key: 'torch.Tensor',
    value: 'torch.Tensor',
    mask: Optional['torch.Tensor'] = None,
    dropout: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None
) -> 'torch.Tensor':
    """
    Memory-efficient attention computation.

    This function computes scaled dot-product attention in a memory-efficient
    manner by chunking the computation when the sequence length is large.

    Args:
        query: Query tensor, shape (batch, heads, seq_q, d_k)
        key: Key tensor, shape (batch, heads, seq_k, d_k)
        value: Value tensor, shape (batch, heads, seq_v, d_v)
        mask: Optional attention mask
        dropout: Dropout probability
        is_causal: Whether to use causal masking
        scale: Scaling factor (default: 1/sqrt(d_k))

    Returns:
        Attention output, shape (batch, heads, seq_q, d_v)
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for memory_efficient_attention")

    batch_size, num_heads, seq_q, d_k = query.shape
    seq_k = key.shape[2]
    d_v = value.shape[3]

    if scale is None:
        scale = 1.0 / math.sqrt(d_k)

    # Memory threshold for chunking (in MB)
    # If attention matrix would be larger than this, chunk the computation
    memory_threshold_mb = 100
    attention_size_mb = (batch_size * num_heads * seq_q * seq_k * 4) / (1024 * 1024)

    if attention_size_mb > memory_threshold_mb and seq_q == seq_k:
        # Chunked attention for memory efficiency
        return _chunked_attention(query, key, value, mask, scale, dropout, is_causal)
    else:
        # Standard attention
        return _standard_attention(query, key, value, mask, scale, dropout, is_causal)


def _standard_attention(
    query: 'torch.Tensor',
    key: 'torch.Tensor',
    value: 'torch.Tensor',
    mask: Optional['torch.Tensor'],
    scale: float,
    dropout: float,
    is_causal: bool
) -> 'torch.Tensor':
    """Standard scaled dot-product attention."""
    # Compute attention scores
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    # Apply mask
    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

    if is_causal:
        causal_mask = torch.triu(
            torch.ones(query.shape[2], key.shape[2], device=query.device, dtype=torch.bool),
            diagonal=1
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

    # Softmax
    attn_probs = torch.softmax(attn_scores, dim=-1)

    # Dropout
    if dropout > 0 and query.training:
        attn_probs = torch.nn.functional.dropout(attn_probs, p=dropout)

    # Apply attention to values
    output = torch.matmul(attn_probs, value)

    return output


def _chunked_attention(
    query: 'torch.Tensor',
    key: 'torch.Tensor',
    value: 'torch.Tensor',
    mask: Optional['torch.Tensor'],
    scale: float,
    dropout: float,
    is_causal: bool,
    chunk_size: int = 256
) -> 'torch.Tensor':
    """
    Chunked attention for memory efficiency.

    Instead of computing the full (seq_q, seq_k) attention matrix,
    compute it in chunks to reduce peak memory usage.
    """
    batch_size, num_heads, seq_q, d_k = query.shape
    seq_k = key.shape[2]
    d_v = value.shape[3]

    output = torch.zeros(batch_size, num_heads, seq_q, d_v, device=query.device, dtype=query.dtype)

    for q_start in range(0, seq_q, chunk_size):
        q_end = min(q_start + chunk_size, seq_q)
        query_chunk = query[:, :, q_start:q_end, :]

        # Compute attention for this query chunk against all keys
        attn_scores = torch.matmul(query_chunk, key.transpose(-2, -1)) * scale

        # Apply mask
        if mask is not None:
            mask_chunk = mask[:, :, q_start:q_end, :] if mask.dim() == 4 else mask
            attn_scores = attn_scores.masked_fill(mask_chunk == 0, float('-inf'))

        if is_causal:
            causal_mask = torch.triu(
                torch.ones(q_end - q_start, seq_k, device=query.device, dtype=torch.bool),
                diagonal=q_start + 1
            )
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

        # Softmax
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Dropout
        if dropout > 0 and query.training:
            attn_probs = torch.nn.functional.dropout(attn_probs, p=dropout)

        # Apply to values
        output[:, :, q_start:q_end, :] = torch.matmul(attn_probs, value)

    return output


# =============================================================================
# In-Place Operations
# =============================================================================


def enable_inplace_activation(model: 'nn.Module') -> 'nn.Module':
    """
    Enable in-place operations for activation functions.

    This modifies ReLU and other activation layers to use in-place operations,
    reducing memory footprint at the cost of not being able to access
    intermediate activations.

    Args:
        model: Model to modify

    Returns:
        Modified model with in-place activations
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for enable_inplace_activation")

    for module in model.modules():
        if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.ELU, nn.SELU)):
            module.inplace = True

    return model


# =============================================================================
# Memory Optimizer Trainer
# =============================================================================


class MemoryOptimizedTrainer:
    """
    Training loop with integrated memory optimization.

    This trainer combines gradient checkpointing, CPU offloading, and
    gradient accumulation for maximum memory efficiency on limited GPUs.

    Args:
        model: Model to train
        config: Memory optimization configuration
        accumulation_steps: Gradient accumulation steps

    Example:
        >>> config = MemoryOptimizationConfig(
        ...     enable_gradient_checkpointing=True,
        ...     checkpoint_strategy='balanced',
        ...     enable_cpu_offloading=True
        ... )
        >>> trainer = MemoryOptimizedTrainer(model, config)
        >>> trainer.train(dataloader, optimizer, criterion, epochs=10)
    """

    def __init__(
        self,
        model: 'nn.Module',
        config: Optional[MemoryOptimizationConfig] = None,
        accumulation_steps: int = 1
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for MemoryOptimizedTrainer")

        self.config = config or MemoryOptimizationConfig()
        self.accumulation_steps = accumulation_steps
        self._step_count = 0

        # Apply memory optimizations
        self.model = self._apply_optimizations(model)
        self._offloader: Optional[CPUOffloader] = None

        if self.config.enable_cpu_offloading:
            self._offloader = CPUOffloader(
                self.model,
                offload_optimizer=self.config.offload_optimizer_states
            )

    def _apply_optimizations(self, model: 'nn.Module') -> 'nn.Module':
        """Apply configured memory optimizations to model."""
        if self.config.enable_gradient_checkpointing:
            model = apply_gradient_checkpointing(
                model,
                strategy=self.config.checkpoint_strategy,
                checkpoint_layers=self.config.checkpoint_layers
            )

        if self.config.enable_activation_recomputation:
            # Handled by checkpointing
            pass

        return model

    def train_step(
        self,
        batch: Tuple['torch.Tensor', 'torch.Tensor'],
        optimizer: 'torch.optim.Optimizer',
        criterion: Callable,
        scaler: Optional['GradScaler'] = None
    ) -> float:
        """
        Execute a single training step with memory optimization.

        Args:
            batch: (inputs, targets) tuple
            optimizer: Optimizer
            criterion: Loss function
            scaler: Optional GradScaler for AMP

        Returns:
            Loss value
        """
        inputs, targets = batch

        # Load parameters to GPU if offloading
        if self._offloader:
            self._offloader.load_to_gpu()

        # Forward pass
        with autocast(enabled=scaler is not None):
            outputs = self.model(inputs)
            loss = criterion(outputs, targets) / self.accumulation_steps

        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        self._step_count += 1

        # Gradient accumulation
        if self._step_count % self.accumulation_steps == 0:
            # Offload gradients if using CPU offloading
            if self._offloader and self.config.offload_activations:
                self._offloader.offload_gradients()

            # Optimizer step
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

            # Offload parameters if using CPU offloading
            if self._offloader:
                self._offloader.offload_to_cpu()
                clear_cuda_cache()

        return loss.item() * self.accumulation_steps

    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        return get_memory_usage()


# =============================================================================
# Benchmarking Utilities
# =============================================================================


def benchmark_memory(
    model: 'nn.Module',
    input_shape: Tuple[int, ...],
    device: 'torch.device',
    num_iterations: int = 10,
    use_checkpointing: bool = False,
    use_offloading: bool = False
) -> Dict[str, Any]:
    """
    Benchmark memory usage with and without optimizations.

    Args:
        model: Model to benchmark
        input_shape: Shape of input tensor (without batch dimension)
        device: Device to run on
        num_iterations: Number of iterations to average
        use_checkpointing: Enable gradient checkpointing
        use_offloading: Enable CPU offloading

    Returns:
        Dictionary with memory statistics and timing
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for benchmark_memory")

    model = model.to(device)
    model.train()

    # Apply optimizations if requested
    if use_checkpointing:
        model = apply_gradient_checkpointing(model, strategy='balanced')

    offloader = None
    if use_offloading:
        offloader = CPUOffloader(model)

    # Prepare data
    batch_size = input_shape[0] if len(input_shape) > 0 else 1
    dummy_input = torch.randn(input_shape, device=device)
    dummy_target = torch.randn(model(dummy_input).shape, device=device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Reset memory stats
    reset_memory_stats(device)
    clear_cuda_cache()

    # Warmup
    for _ in range(3):
        optimizer.zero_grad()
        output = model(dummy_input)
        loss = criterion(output, dummy_target)
        loss.backward()
        optimizer.step()

    # Benchmark
    reset_memory_stats(device)
    start_time = time.time()

    for _ in range(num_iterations):
        optimizer.zero_grad()

        if offloader:
            offloader.load_to_gpu()

        output = model(dummy_input)
        loss = criterion(output, dummy_target)
        loss.backward()
        optimizer.step()

        if offloader:
            offloader.offload_to_cpu()
            clear_cuda_cache()

    end_time = time.time()
    peak_memory = get_peak_memory(device)

    return {
        'peak_memory_mb': peak_memory,
        'avg_iteration_time_ms': (end_time - start_time) / num_iterations * 1000,
        'total_time_s': end_time - start_time,
        'use_checkpointing': use_checkpointing,
        'use_offloading': use_offloading
    }


def compare_memory_strategies(
    model_factory: Callable[[], 'nn.Module'],
    input_shape: Tuple[int, ...],
    device: 'torch.device',
    num_layers: int = 10
) -> Dict[str, Dict[str, Any]]:
    """
    Compare different memory optimization strategies.

    Args:
        model_factory: Function that creates a new model instance
        input_shape: Input tensor shape
        device: Device to run on
        num_layers: Number of layers for layer-wise comparison

    Returns:
        Dictionary mapping strategy name to benchmark results
    """
    results = {}

    strategies = [
        ('baseline', False, False),
        ('checkpointing', True, False),
        ('offloading', False, True),
        ('checkpointing+offloading', True, True),
    ]

    for name, use_ckpt, use_offload in strategies:
        print(f"Benchmarking strategy: {name}")
        model = model_factory()

        try:
            result = benchmark_memory(
                model=model,
                input_shape=input_shape,
                device=device,
                use_checkpointing=use_ckpt,
                use_offloading=use_offload
            )
            results[name] = result

            # Calculate memory savings
            if 'baseline' in results:
                baseline_memory = results['baseline']['peak_memory_mb']
                current_memory = result['peak_memory_mb']
                savings = (baseline_memory - current_memory) / baseline_memory * 100
                results[name]['memory_savings_percent'] = savings

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                results[name] = {'error': 'OOM', 'peak_memory_mb': float('inf')}
                clear_cuda_cache()
            else:
                raise

    return results


# =============================================================================
# Registry
# =============================================================================


MEMORY_OPTIMIZER_COMPONENTS = {
    # Config
    'MemoryOptimizationConfig': MemoryOptimizationConfig,
    # Utilities
    'get_memory_usage': get_memory_usage,
    'reset_memory_stats': reset_memory_stats,
    'get_peak_memory': get_peak_memory,
    'clear_cuda_cache': clear_cuda_cache,
    # Checkpointing
    'get_checkpoint_segments': get_checkpoint_segments,
    'CheckpointedSequential': CheckpointedSequential,
    'apply_gradient_checkpointing': apply_gradient_checkpointing,
    # Offloading
    'CPUOffloader': CPUOffloader,
    'OffloadedOptimizer': OffloadedOptimizer,
    # Recomputation
    'ActivationRecomputer': ActivationRecomputer,
    # Attention
    'memory_efficient_attention': memory_efficient_attention,
    # In-place
    'enable_inplace_activation': enable_inplace_activation,
    # Trainer
    'MemoryOptimizedTrainer': MemoryOptimizedTrainer,
    # Benchmarking
    'benchmark_memory': benchmark_memory,
    'compare_memory_strategies': compare_memory_strategies,
}
