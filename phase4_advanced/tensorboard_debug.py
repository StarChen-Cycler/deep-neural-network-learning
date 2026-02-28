"""
TensorBoard Debug: PyTorch integration for training visualization.

This module provides:
    - TensorBoardMonitor: Easy integration with PyTorch training loops
    - Automatic gradient/weight histogram logging
    - Training curve visualization
    - Model graph visualization

Usage:
    from phase4_advanced.tensorboard_debug import TensorBoardMonitor

    monitor = TensorBoardMonitor(log_dir='runs/experiment')
    monitor.watch(model)

    for epoch in range(epochs):
        for batch in dataloader:
            loss = train_step(model, batch)
            monitor.log_step(epoch, batch_idx, loss, model)

    monitor.close()

TensorBoard Commands:
    tensorboard --logdir=runs
    tensorboard --logdir=runs --port=6007

References:
    - TensorFlow TensorBoard Documentation
    - PyTorch torch.utils.tensorboard
"""

from typing import List, Optional, Union, Dict, Any, Callable, Tuple
from dataclasses import dataclass
import numpy as np

# Import PyTorch conditionally
try:
    import torch
    import torch.nn as nn
    from torch.utils.tensorboard import SummaryWriter
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    SummaryWriter = None

from .training_monitor import (
    TrainingMonitor,
    GradientReport,
    ActivationReport,
    WeightUpdateReport,
    MonitorStatus,
    detect_dead_neurons,
    compute_activation_distribution,
)


# =============================================================================
# TensorBoard Monitor Class
# =============================================================================


class TensorBoardMonitor:
    """
    TensorBoard monitor for PyTorch training.

    Provides easy integration with PyTorch models for automatic
    logging of gradients, weights, activations, and training metrics.

    Features:
        - Automatic gradient histogram logging
        - Weight distribution tracking
        - Activation monitoring via forward hooks
        - Dead neuron detection
        - Learning rate tracking
        - Loss curve visualization

    Usage:
        # Basic usage
        monitor = TensorBoardMonitor('runs/my_experiment')
        monitor.watch(model, log_freq=100)

        for epoch in range(epochs):
            for batch_idx, (x, y) in enumerate(dataloader):
                loss = train_step(model, x, y)
                monitor.log_training_step(epoch, batch_idx, loss)

        monitor.close()

        # With gradient checking
        monitor = TensorBoardMonitor('runs/experiment')
        monitor.watch(model)

        # In training loop
        monitor.log_training_step(epoch, batch_idx, loss, model)

        # Check for issues
        if monitor.has_issues():
            print(monitor.get_warnings())

    Attributes:
        writer: TensorBoard SummaryWriter
        log_dir: Directory for TensorBoard logs
        global_step: Current training step
    """

    def __init__(
        self,
        log_dir: str = "runs/experiment",
        comment: str = "",
        purge_step: Optional[int] = None,
        max_queue: int = 10,
        flush_secs: int = 120,
        filename_suffix: str = "",
    ):
        """
        Initialize TensorBoard monitor.

        Args:
            log_dir: Directory for TensorBoard logs
            comment: Comment to append to log_dir
            purge_step: Step at which to start purging
            max_queue: Max events to queue before flushing
            flush_secs: How often to flush events
            filename_suffix: Suffix for event file
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for TensorBoardMonitor. "
                "Install with: pip install torch tensorboard"
            )

        self.log_dir = log_dir
        self.writer = SummaryWriter(
            log_dir=log_dir,
            comment=comment,
            purge_step=purge_step,
            max_queue=max_queue,
            flush_secs=flush_secs,
            filename_suffix=filename_suffix,
        )

        # State
        self.global_step = 0
        self._watched_model = None
        self._hooks = []
        self._activation_cache: Dict[str, torch.Tensor] = {}
        self._warnings: List[str] = []
        self._log_freq = 100

        # Training metrics history
        self._loss_history: List[float] = []
        self._lr_history: List[float] = []

    def watch(
        self,
        model: "nn.Module",
        log_freq: int = 100,
        log_gradients: bool = True,
        log_weights: bool = True,
        log_activations: bool = False,
    ) -> None:
        """
        Watch a model for automatic logging.

        Args:
            model: PyTorch model to watch
            log_freq: Frequency (in steps) to log histograms
            log_gradients: Whether to log gradient histograms
            log_weights: Whether to log weight histograms
            log_activations: Whether to log activation histograms
        """
        self._watched_model = model
        self._log_freq = log_freq

        if log_activations:
            self._register_activation_hooks(model)

    def _register_activation_hooks(self, model: "nn.Module") -> None:
        """Register forward hooks to capture activations."""

        def make_hook(name: str):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self._activation_cache[name] = output.detach()
                elif isinstance(output, (tuple, list)) and len(output) > 0:
                    self._activation_cache[name] = output[0].detach()

            return hook

        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_forward_hook(make_hook(name))
                self._hooks.append(hook)

    def log_training_step(
        self,
        epoch: int,
        batch_idx: int,
        loss: float,
        model: Optional["nn.Module"] = None,
        optimizer: Optional["torch.optim.Optimizer"] = None,
        learning_rate: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Log a training step.

        This should be called after optimizer.step() to capture
        the updated gradients and weights.

        Args:
            epoch: Current epoch
            batch_idx: Batch index within epoch
            loss: Loss value for this step
            model: Model (uses watched model if not provided)
            optimizer: Optimizer (for learning rate extraction)
            learning_rate: Manual learning rate (overrides optimizer)
            metrics: Additional metrics to log

        Returns:
            Dictionary with logged information
        """
        model = model or self._watched_model

        # Update global step
        self.global_step += 1

        # Log loss
        self.writer.add_scalar("Loss/train", loss, self.global_step)
        self._loss_history.append(loss)

        # Log learning rate
        if learning_rate is None and optimizer is not None:
            learning_rate = optimizer.param_groups[0]["lr"]

        if learning_rate is not None:
            self.writer.add_scalar("LearningRate", learning_rate, self.global_step)
            self._lr_history.append(learning_rate)

        # Log additional metrics
        if metrics:
            for name, value in metrics.items():
                self.writer.add_scalar(f"Metrics/{name}", value, self.global_step)

        # Log histograms periodically
        should_log = self.global_step % self._log_freq == 0

        logged_info = {
            "step": self.global_step,
            "epoch": epoch,
            "batch_idx": batch_idx,
            "loss": loss,
            "learning_rate": learning_rate,
        }

        if should_log and model is not None:
            gradient_info = self._log_gradients(model)
            weight_info = self._log_weights(model)

            logged_info["gradients"] = gradient_info
            logged_info["weights"] = weight_info

            if self._activation_cache:
                activation_info = self._log_activations()
                logged_info["activations"] = activation_info

            # Check for issues
            self._check_training_health(logged_info)

        return logged_info

    def _log_gradients(self, model: "nn.Module") -> Dict[str, Dict[str, float]]:
        """Log gradient histograms and statistics."""
        gradient_info = {}

        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            grad = param.grad.detach().cpu().numpy()
            safe_name = name.replace(".", "/")

            # Log histogram
            self.writer.add_histogram(f"Gradients/{safe_name}", grad, self.global_step)

            # Log statistics
            grad_norm = float(np.linalg.norm(grad))
            grad_mean = float(np.mean(grad))
            grad_std = float(np.std(grad))
            zero_ratio = float(np.mean(np.abs(grad) < 1e-10))

            self.writer.add_scalar(f"GradientNorm/{safe_name}", grad_norm, self.global_step)
            self.writer.add_scalar(f"GradientStd/{safe_name}", grad_std, self.global_step)

            gradient_info[name] = {
                "norm": grad_norm,
                "mean": grad_mean,
                "std": grad_std,
                "zero_ratio": zero_ratio,
            }

            # Detect issues
            if grad_norm < 1e-7:
                self._warnings.append(
                    f"Step {self.global_step}: Vanishing gradient in {name} (norm={grad_norm:.2e})"
                )
            elif grad_norm > 100:
                self._warnings.append(
                    f"Step {self.global_step}: Exploding gradient in {name} (norm={grad_norm:.2e})"
                )

            if zero_ratio > 0.9:
                self._warnings.append(
                    f"Step {self.global_step}: High zero ratio ({zero_ratio:.1%}) in {name}"
                )

        return gradient_info

    def _log_weights(self, model: "nn.Module") -> Dict[str, Dict[str, float]]:
        """Log weight histograms and statistics."""
        weight_info = {}

        for name, param in model.named_parameters():
            weight = param.detach().cpu().numpy()
            safe_name = name.replace(".", "/")

            # Log histogram
            self.writer.add_histogram(f"Weights/{safe_name}", weight, self.global_step)

            # Log statistics
            weight_norm = float(np.linalg.norm(weight))
            weight_mean = float(np.mean(weight))
            weight_std = float(np.std(weight))

            self.writer.add_scalar(f"WeightNorm/{safe_name}", weight_norm, self.global_step)
            self.writer.add_scalar(f"WeightStd/{safe_name}", weight_std, self.global_step)

            weight_info[name] = {
                "norm": weight_norm,
                "mean": weight_mean,
                "std": weight_std,
            }

        return weight_info

    def _log_activations(self) -> Dict[str, Dict[str, float]]:
        """Log activation histograms from cache."""
        activation_info = {}

        for name, activation in self._activation_cache.items():
            act = activation.cpu().numpy()
            safe_name = name.replace(".", "/")

            # Log histogram
            self.writer.add_histogram(f"Activations/{safe_name}", act, self.global_step)

            # Log statistics
            act_mean = float(np.mean(act))
            act_std = float(np.std(act))
            zero_ratio = float(np.mean(np.abs(act) < 1e-10))

            self.writer.add_scalar(f"ActivationMean/{safe_name}", act_mean, self.global_step)
            self.writer.add_scalar(f"ActivationStd/{safe_name}", act_std, self.global_step)

            activation_info[name] = {
                "mean": act_mean,
                "std": act_std,
                "zero_ratio": zero_ratio,
            }

            # Check for dead neurons
            has_dead, dead_ratio = detect_dead_neurons(act)
            if has_dead:
                self._warnings.append(
                    f"Step {self.global_step}: Dead neurons detected in {name} ({dead_ratio:.1%})"
                )

        # Clear cache
        self._activation_cache.clear()

        return activation_info

    def _check_training_health(self, logged_info: Dict[str, Any]) -> None:
        """Check for training health issues."""
        if "gradients" in logged_info:
            for name, grad_info in logged_info["gradients"].items():
                if np.isnan(grad_info["norm"]):
                    self._warnings.append(
                        f"Step {self.global_step}: NaN gradient detected in {name}"
                    )

    def log_validation(
        self,
        epoch: int,
        val_loss: float,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Log validation metrics.

        Args:
            epoch: Current epoch
            val_loss: Validation loss
            metrics: Additional validation metrics
        """
        self.writer.add_scalar("Loss/validation", val_loss, epoch)

        if metrics:
            for name, value in metrics.items():
                self.writer.add_scalar(f"Validation/{name}", value, epoch)

    def log_epoch_summary(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Log epoch summary.

        Args:
            epoch: Completed epoch
            train_loss: Average training loss
            val_loss: Optional validation loss
            metrics: Additional epoch metrics
        """
        self.writer.add_scalar("Epoch/Loss/train", train_loss, epoch)

        if val_loss is not None:
            self.writer.add_scalar("Epoch/Loss/validation", val_loss, epoch)

        if metrics:
            for name, value in metrics.items():
                self.writer.add_scalar(f"Epoch/{name}", value, epoch)

    def log_graph(
        self,
        model: "nn.Module",
        input_to_model: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    ) -> None:
        """
        Log model graph.

        Args:
            model: Model to visualize
            input_to_model: Sample input for tracing
        """
        try:
            self.writer.add_graph(model, input_to_model)
        except Exception as e:
            self._warnings.append(f"Could not log model graph: {str(e)}")

    def log_histogram(
        self,
        tag: str,
        values: Union[np.ndarray, torch.Tensor],
        step: Optional[int] = None,
    ) -> None:
        """
        Log a custom histogram.

        Args:
            tag: Tag for the histogram
            values: Values to histogram
            step: Step number (uses global_step if not provided)
        """
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()

        self.writer.add_histogram(tag, values, step or self.global_step)

    def log_scalar(
        self,
        tag: str,
        value: float,
        step: Optional[int] = None,
    ) -> None:
        """
        Log a custom scalar.

        Args:
            tag: Tag for the scalar
            value: Scalar value
            step: Step number (uses global_step if not provided)
        """
        self.writer.add_scalar(tag, value, step or self.global_step)

    def log_text(
        self,
        tag: str,
        text: str,
        step: Optional[int] = None,
    ) -> None:
        """
        Log text.

        Args:
            tag: Tag for the text
            text: Text content
            step: Step number (uses global_step if not provided)
        """
        self.writer.add_text(tag, text, step or self.global_step)

    def log_hyperparams(
        self,
        hparams: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Log hyperparameters.

        Args:
            hparams: Dictionary of hyperparameters
            metrics: Optional final metrics
        """
        # Filter to allowed types
        allowed_types = (int, float, str, bool)
        filtered_hparams = {
            k: v for k, v in hparams.items()
            if isinstance(v, allowed_types) or v is None
        }

        if metrics:
            self.writer.add_hparams(filtered_hparams, metrics)
        else:
            # Just log as text
            hparam_str = "\n".join(f"- {k}: {v}" for k, v in filtered_hparams.items())
            self.log_text("Hyperparameters", hparam_str, 0)

    def has_issues(self) -> bool:
        """Check if any issues were detected."""
        return len(self._warnings) > 0

    def get_warnings(self) -> List[str]:
        """Get list of warnings."""
        return self._warnings.copy()

    def clear_warnings(self) -> None:
        """Clear warning history."""
        self._warnings.clear()

    def flush(self) -> None:
        """Flush the writer."""
        self.writer.flush()

    def close(self) -> None:
        """Close the monitor and release resources."""
        # Remove hooks
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

        # Close writer
        self.writer.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


# =============================================================================
# Convenience Functions
# =============================================================================


def create_monitor(
    experiment_name: str,
    base_dir: str = "runs",
    **kwargs,
) -> TensorBoardMonitor:
    """
    Create a TensorBoard monitor with standard configuration.

    Args:
        experiment_name: Name of the experiment
        base_dir: Base directory for logs
        **kwargs: Additional arguments for TensorBoardMonitor

    Returns:
        Configured TensorBoardMonitor
    """
    import os
    log_dir = os.path.join(base_dir, experiment_name)
    return TensorBoardMonitor(log_dir=log_dir, **kwargs)


def quick_visualize(
    model: "nn.Module",
    dataloader: "torch.utils.data.DataLoader",
    log_dir: str = "runs/quick_viz",
    num_batches: int = 10,
) -> TensorBoardMonitor:
    """
    Quick visualization of model gradients and activations.

    Runs a few batches through the model and logs to TensorBoard.

    Args:
        model: Model to visualize
        dataloader: DataLoader with sample data
        log_dir: Directory for logs
        num_batches: Number of batches to run

    Returns:
        TensorBoardMonitor with logged data
    """
    monitor = TensorBoardMonitor(log_dir=log_dir)
    monitor.watch(model, log_freq=1, log_activations=True)

    model.train()
    criterion = nn.CrossEntropyLoss()

    for batch_idx, (x, y) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        model.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()

        monitor.log_training_step(0, batch_idx, loss.item(), model)

    monitor.close()
    return monitor


# =============================================================================
# Weights & Biases Integration
# =============================================================================


class WandBMonitor:
    """
    WandB monitor for PyTorch training.

    Provides Weights & Biases integration for experiment tracking
    with automatic gradient/weight histogram logging.

    Features:
        - Automatic model watching with wandb.watch()
        - Gradient and parameter histogram logging
        - System metrics tracking (GPU, memory, etc.)
        - Model checkpoint saving
        - Config/hyperparameter logging

    Usage:
        monitor = WandBMonitor(
            project="my-project",
            name="experiment-1",
            config={"lr": 0.001, "epochs": 10}
        )
        monitor.watch(model)

        for epoch in range(epochs):
            for batch_idx, (x, y) in enumerate(dataloader):
                loss = train_step(model, x, y)
                monitor.log_training_step(epoch, batch_idx, loss, model)

        monitor.finish()

    Environment Variables:
        WANDB_API_KEY: Your WandB API key
        WANDB_PROJECT: Default project name
        WANDB_ENTITY: Default entity/team
        WANDB_MODE: "online", "offline", or "disabled"
    """

    def __init__(
        self,
        project: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        dir: Optional[str] = None,
        mode: Optional[str] = None,
        anonymous: Optional[str] = None,
    ):
        """
        Initialize WandB monitor.

        Args:
            project: WandB project name
            name: Run name
            config: Configuration dictionary
            tags: List of tags
            notes: Run notes
            dir: Directory for local run files
            mode: "online", "offline", or "disabled"
            anonymous: "allow", "must", or "never"
        """
        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            raise ImportError(
                "wandb is required for WandBMonitor. "
                "Install with: pip install wandb"
            )

        # Initialize wandb run
        self.run = wandb.init(
            project=project,
            name=name,
            config=config,
            tags=tags,
            notes=notes,
            dir=dir,
            mode=mode,
            anonymous=anonymous,
            reinit=True,
        )

        # State
        self.global_step = 0
        self._watched_model = None
        self._warnings: List[str] = []

    def watch(
        self,
        model: "nn.Module",
        log_freq: int = 100,
        log: str = "all",
        log_graph: bool = False,
    ) -> None:
        """
        Watch a model for automatic logging.

        Args:
            model: PyTorch model to watch
            log_freq: Frequency (in steps) to log
            log: What to log: "all", "gradients", "parameters", "None"
            log_graph: Whether to log the model graph
        """
        self._watched_model = model
        self._wandb.watch(model, log=log, log_freq=log_freq, log_graph=log_graph)

    def log_training_step(
        self,
        epoch: int,
        batch_idx: int,
        loss: float,
        model: Optional["nn.Module"] = None,
        optimizer: Optional["torch.optim.Optimizer"] = None,
        learning_rate: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None,
        log_histograms: bool = False,
    ) -> Dict[str, Any]:
        """
        Log a training step.

        Args:
            epoch: Current epoch
            batch_idx: Batch index within epoch
            loss: Loss value for this step
            model: Model (uses watched model if not provided)
            optimizer: Optimizer (for learning rate extraction)
            learning_rate: Manual learning rate (overrides optimizer)
            metrics: Additional metrics to log
            log_histograms: Whether to log gradient histograms manually

        Returns:
            Dictionary with logged information
        """
        model = model or self._watched_model
        self.global_step += 1

        # Build log dictionary
        log_dict = {
            "loss": loss,
            "epoch": epoch,
            "batch_idx": batch_idx,
        }

        # Get learning rate
        if learning_rate is None and optimizer is not None:
            learning_rate = optimizer.param_groups[0]["lr"]

        if learning_rate is not None:
            log_dict["learning_rate"] = learning_rate

        # Add metrics
        if metrics:
            log_dict.update(metrics)

        # Log gradients manually if requested
        if log_histograms and model is not None:
            grad_histograms = self._build_gradient_histograms(model)
            log_dict.update(grad_histograms)

        # Log to wandb
        self._wandb.log(log_dict, step=self.global_step)

        return log_dict

    def _build_gradient_histograms(
        self, model: "nn.Module"
    ) -> Dict[str, "wandb.Histogram"]:
        """Build gradient histograms for wandb logging."""
        histograms = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach().cpu().numpy()
                safe_name = name.replace(".", "/")
                histograms[f"Gradients/{safe_name}"] = self._wandb.Histogram(grad)

        return histograms

    def log_validation(
        self,
        epoch: int,
        val_loss: float,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Log validation metrics.

        Args:
            epoch: Current epoch
            val_loss: Validation loss
            metrics: Additional validation metrics
        """
        log_dict = {"val_loss": val_loss}

        if metrics:
            for name, value in metrics.items():
                log_dict[f"val_{name}"] = value

        self._wandb.log(log_dict, step=self.global_step)

    def log_epoch_summary(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Log epoch summary.

        Args:
            epoch: Completed epoch
            train_loss: Average training loss
            val_loss: Optional validation loss
            metrics: Additional epoch metrics
        """
        log_dict = {
            "epoch_loss/train": train_loss,
        }

        if val_loss is not None:
            log_dict["epoch_loss/val"] = val_loss

        if metrics:
            for name, value in metrics.items():
                log_dict[f"epoch_{name}"] = value

        self._wandb.log(log_dict, step=self.global_step)

    def log_artifact(
        self,
        artifact_path: str,
        name: str,
        type: str = "model",
        description: Optional[str] = None,
    ) -> None:
        """
        Log an artifact (e.g., model checkpoint).

        Args:
            artifact_path: Path to artifact file
            name: Artifact name
            type: Artifact type (model, dataset, etc.)
            description: Optional description
        """
        artifact = self._wandb.Artifact(name, type=type, description=description)
        artifact.add_file(artifact_path)
        self.run.log_artifact(artifact)

    def log_model(
        self,
        model: "nn.Module",
        name: str = "model",
        description: Optional[str] = None,
    ) -> None:
        """
        Log a PyTorch model.

        Args:
            model: Model to log
            name: Model name
            description: Optional description
        """
        self._wandb.log_artifact(
            self._wandb.Artifact(name, type="model", description=description)
        )
        # Also save model state
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, f"{name}.pt")
            torch.save(model.state_dict(), path)

            artifact = self._wandb.Artifact(name, type="model")
            artifact.add_file(path)
            self.run.log_artifact(artifact)

    def log_figure(
        self,
        figure: Any,
        name: str,
        step: Optional[int] = None,
    ) -> None:
        """
        Log a matplotlib figure.

        Args:
            figure: Matplotlib figure
            name: Figure name
            step: Optional step number
        """
        self._wandb.log({name: self._wandb.Image(figure)}, step=step or self.global_step)

    def log_table(
        self,
        name: str,
        data: List[List[Any]],
        columns: Optional[List[str]] = None,
    ) -> None:
        """
        Log a table.

        Args:
            name: Table name
            data: Table data (list of rows)
            columns: Column names
        """
        table = self._wandb.Table(data=data, columns=columns)
        self._wandb.log({name: table})

    def has_issues(self) -> bool:
        """Check if any issues were detected."""
        return len(self._warnings) > 0

    def get_warnings(self) -> List[str]:
        """Get list of warnings."""
        return self._warnings.copy()

    @property
    def config(self) -> Dict[str, Any]:
        """Get the run config."""
        return dict(self._wandb.config)

    @property
    def url(self) -> str:
        """Get the run URL."""
        return self.run.url

    def finish(self, quiet: bool = False) -> None:
        """
        Finish the run.

        Args:
            quiet: If True, suppress output
        """
        self._wandb.finish(quiet=quiet)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()
        return False


# =============================================================================
# Dual Monitor (TensorBoard + WandB)
# =============================================================================


class DualMonitor:
    """
    Dual monitor that logs to both TensorBoard and WandB.

    Usage:
        monitor = DualMonitor(
            tensorboard_dir="runs/experiment",
            wandb_project="my-project",
            wandb_name="experiment-1"
        )
        monitor.watch(model)

        for epoch in range(epochs):
            for batch_idx, (x, y) in enumerate(dataloader):
                loss = train_step(model, x, y)
                monitor.log_training_step(epoch, batch_idx, loss, model)

        monitor.close()
    """

    def __init__(
        self,
        tensorboard_dir: str = "runs/experiment",
        wandb_project: Optional[str] = None,
        wandb_name: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize dual monitor.

        Args:
            tensorboard_dir: TensorBoard log directory
            wandb_project: WandB project name (None to disable)
            wandb_name: WandB run name
            wandb_config: WandB config
            **kwargs: Additional arguments for TensorBoardMonitor
        """
        # Initialize TensorBoard
        self.tb_monitor = TensorBoardMonitor(log_dir=tensorboard_dir, **kwargs)

        # Initialize WandB if project specified
        self.wb_monitor = None
        if wandb_project is not None:
            self.wb_monitor = WandBMonitor(
                project=wandb_project,
                name=wandb_name,
                config=wandb_config,
            )

        self.global_step = 0

    def watch(
        self,
        model: "nn.Module",
        log_freq: int = 100,
        log_activations: bool = True,
    ) -> None:
        """Watch model for both monitors."""
        self.tb_monitor.watch(model, log_freq=log_freq, log_activations=log_activations)

        if self.wb_monitor is not None:
            self.wb_monitor.watch(model, log_freq=log_freq)

    def log_training_step(
        self,
        epoch: int,
        batch_idx: int,
        loss: float,
        model: Optional["nn.Module"] = None,
        optimizer: Optional["torch.optim.Optimizer"] = None,
        learning_rate: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Log training step to both monitors."""
        self.global_step += 1

        # Log to TensorBoard
        tb_info = self.tb_monitor.log_training_step(
            epoch, batch_idx, loss, model, optimizer, learning_rate, metrics
        )

        # Log to WandB
        if self.wb_monitor is not None:
            self.wb_monitor.global_step = self.global_step
            self.wb_monitor.log_training_step(
                epoch, batch_idx, loss, model, optimizer, learning_rate, metrics
            )

        return tb_info

    def log_validation(
        self,
        epoch: int,
        val_loss: float,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log validation metrics to both monitors."""
        self.tb_monitor.log_validation(epoch, val_loss, metrics)

        if self.wb_monitor is not None:
            self.wb_monitor.log_validation(epoch, val_loss, metrics)

    def close(self) -> None:
        """Close both monitors."""
        self.tb_monitor.close()

        if self.wb_monitor is not None:
            self.wb_monitor.finish()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# =============================================================================
# Module Constants
# =============================================================================


TENSORBOARD_DEBUG_COMPONENTS = [
    "TensorBoardMonitor",
    "WandBMonitor",
    "DualMonitor",
    "create_monitor",
    "quick_visualize",
]
