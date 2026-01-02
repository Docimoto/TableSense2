"""
Experiment logging infrastructure with Weights & Biases integration.

This module provides experiment tracking capabilities for training runs,
including loss curves, validation metrics, hyperparameters, and model checkpoints.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import wandb
import torch.nn as nn


class ExperimentLogger:
    """
    Logger for experiment tracking with W&B integration.
    
    Handles:
    - W&B run initialization and logging
    - Loss curves and metrics
    - Hyperparameter logging
    - Model checkpoint saving
    - Config file saving
    """
    
    def __init__(
        self,
        experiment_name: str,
        project_name: str = "tablesense2",
        run_dir: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
        resume: Optional[str] = None,
        mode: str = "online",
    ):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Name for this experiment run
            project_name: W&B project name
            run_dir: Directory to save checkpoints and configs (default: runs/{experiment_name})
            config: Dictionary of hyperparameters/config to log
            tags: List of tags for W&B run
            resume: W&B run ID to resume (optional)
            mode: W&B mode ("online", "offline", "disabled")
        """
        # Add timestamp suffix to experiment name for unique run tracking
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        experiment_name_with_timestamp = f"{experiment_name} {timestamp}"
        
        self.experiment_name = experiment_name_with_timestamp
        self.project_name = project_name
        
        # Set up run directory
        if run_dir is None:
            run_dir = Path("runs") / experiment_name_with_timestamp
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.config_dir = self.run_dir / "configs"
        self.config_dir.mkdir(exist_ok=True)
        
        # Initialize W&B
        self.wandb_run = wandb.init(
            project=project_name,
            name=experiment_name_with_timestamp,
            config=config or {},
            tags=tags or [],
            resume=resume,
            mode=mode,
            dir=str(self.run_dir),
        )
        
        # Save config to file
        if config:
            self.save_config(config, "config.yaml")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to W&B.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Optional step number (epoch/iteration)
        """
        # DO NOT add step to metrics dict - wandb.log() takes step as a separate parameter
        # Adding step to metrics dict can cause W&B to ignore or mishandle the metrics
        try:
            # Ensure step is not None and is a valid integer
            if step is None:
                step = 0
            # Use the run's log method directly to ensure metrics are logged to the correct run
            if hasattr(self, 'wandb_run') and self.wandb_run:
                result = self.wandb_run.log(metrics, step=step, commit=True)
            else:
                # Fallback to global wandb.log()
                result = wandb.log(metrics, step=step, commit=True)
        except Exception as e:
            raise
    
    def log_losses(
        self,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        cls_loss: Optional[float] = None,
        l1_loss: Optional[float] = None,
        iou_loss: Optional[float] = None,
        det_loss: Optional[float] = None,
        step: Optional[int] = None,
    ):
        """
        Log training losses for Deformable DETR.
        
        Args:
            train_loss: Total training loss
            val_loss: Total validation loss
            cls_loss: Classification loss component
            l1_loss: L1 bounding box regression loss component
            iou_loss: IoU/GIoU loss component
            det_loss: Total detection loss (cls + l1 + iou)
            step: Optional step number
        """
        losses = {}
        if train_loss is not None:
            losses["train/loss"] = train_loss
        if val_loss is not None:
            losses["val/loss"] = val_loss
        if det_loss is not None:
            losses["train/det_loss"] = det_loss
        if cls_loss is not None:
            losses["train/cls_loss"] = cls_loss
        if l1_loss is not None:
            losses["train/l1_loss"] = l1_loss
        if iou_loss is not None:
            losses["train/iou_loss"] = iou_loss
        
        if losses:
            self.log_metrics(losses, step=step)
    
    def log_validation_metrics(
        self,
        precision: float,
        recall: float,
        f1: float,
        eob_mean: Optional[float] = None,
        eob_std: Optional[float] = None,
        tp: Optional[int] = None,
        fp: Optional[int] = None,
        fn: Optional[int] = None,
        step: Optional[int] = None,
    ):
        """
        Log validation metrics.
        
        Args:
            precision: Precision score
            recall: Recall score
            f1: F1 score
            eob_mean: Mean EoB (optional)
            eob_std: Standard deviation of EoB (optional)
            tp: Number of true positives (optional)
            fp: Number of false positives (optional)
            fn: Number of false negatives (optional)
            step: Optional step number
        """
        metrics = {
            "val/precision": precision,
            "val/recall": recall,
            "val/f1": f1,
        }
        if eob_mean is not None:
            metrics["val/eob_mean"] = eob_mean
        if eob_std is not None:
            metrics["val/eob_std"] = eob_std
        if tp is not None:
            metrics["val/tp"] = tp
        if fp is not None:
            metrics["val/fp"] = fp
        if fn is not None:
            metrics["val/fn"] = fn
        
        self.log_metrics(metrics, step=step)
    
    def log_system_metrics(
        self,
        gpu_memory_mb: Optional[float] = None,
        cpu_memory_mb: Optional[float] = None,
        epoch_time_seconds: Optional[float] = None,
        step: Optional[int] = None,
    ):
        """
        Log system metrics (GPU memory, CPU memory, timing).
        
        Args:
            gpu_memory_mb: GPU memory usage in MB (optional)
            cpu_memory_mb: CPU memory usage in MB (optional)
            epoch_time_seconds: Time taken for epoch in seconds (optional)
            step: Optional step number
        """
        metrics = {}
        if gpu_memory_mb is not None:
            metrics["system/gpu_memory_mb"] = gpu_memory_mb
        if cpu_memory_mb is not None:
            metrics["system/cpu_memory_mb"] = cpu_memory_mb
        if epoch_time_seconds is not None:
            metrics["system/epoch_time_seconds"] = epoch_time_seconds
        
        if metrics:
            self.log_metrics(metrics, step=step)
    
    def log_gradient_norms(
        self,
        max_grad_norm: Optional[float] = None,
        mean_grad_norm: Optional[float] = None,
        step: Optional[int] = None,
    ):
        """
        Log gradient norms for debugging training stability.
        
        Args:
            max_grad_norm: Maximum gradient norm across all parameters
            mean_grad_norm: Mean gradient norm across all parameters
            step: Optional step number
        """
        metrics = {}
        if max_grad_norm is not None:
            metrics["train/max_grad_norm"] = max_grad_norm
        if mean_grad_norm is not None:
            metrics["train/mean_grad_norm"] = mean_grad_norm
        
        if metrics:
            self.log_metrics(metrics, step=step)
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """
        Log hyperparameters explicitly for sweeps.
        
        Args:
            hyperparams: Dictionary of hyperparameters to log
        """
        wandb.config.update(hyperparams, allow_val_change=True)
    
    def log_model_summary(self, model: nn.Module):
        """
        Log model architecture summary and parameter counts.
        
        Args:
            model: PyTorch model
        """
        # Robust parameter counting
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        wandb.config.update({
            'model/total_params': total_params,
            'model/trainable_params': trainable_params,
            'model/total_params_m': total_params / 1e6,
        })
        
        # Log textual representation as a persistent artifact
        # This is safer than torchsummary for complex/custom architectures
        model_arch_path = self.run_dir / "model_arch.txt"
        with open(model_arch_path, "w") as f:
            f.write(str(model))
        
        # Sanitize experiment name for artifact (wandb only allows alphanumeric, dashes, underscores, dots)
        sanitized_name = self.experiment_name.replace(" ", "_").replace(":", "-")
        artifact = wandb.Artifact(
            name=f"{sanitized_name}_architecture",
            type="model_architecture"
        )
        artifact.add_file(str(model_arch_path))
        self.wandb_run.log_artifact(artifact)
        
        # Optional: Watch gradients
        wandb.watch(model, log="gradients", log_freq=100)
    
    def log_dataset_stats(self, train_size: int, val_size: int, dataset_names: List[str]):
        """
        Log dataset statistics context.
        
        Args:
            train_size: Number of training samples
            val_size: Number of validation samples
            dataset_names: List of dataset names used
        """
        wandb.config.update({
            'data/train_size': train_size,
            'data/val_size': val_size,
            'data/datasets': dataset_names,
        })
    
    def log_model_artifacts(
        self,
        model_path: Path,
        artifact_name: Optional[str] = None,
        artifact_type: str = "model",
        aliases: Optional[list] = None,
    ):
        """
        Log model artifacts to W&B.
        
        Args:
            model_path: Path to model checkpoint file
            artifact_name: Name for the artifact (defaults to experiment name)
            artifact_type: Type of artifact ("model", "checkpoint", etc.)
            aliases: List of aliases for the artifact (e.g., ["latest", "best"])
        """
        if artifact_name is None:
            artifact_name = f"{self.experiment_name}_{artifact_type}"
        
        artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
        )
        artifact.add_file(str(model_path))
        
        if aliases:
            self.wandb_run.log_artifact(artifact, aliases=aliases)
        else:
            self.wandb_run.log_artifact(artifact)
    
    def save_config(self, config: Dict[str, Any], filename: str = "config.yaml"):
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary
            filename: Name of config file
        """
        import yaml
        
        config_path = self.config_dir / filename
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    def save_checkpoint(
        self,
        model_state: Dict[str, Any],
        optimizer_state: Optional[Dict[str, Any]] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        filename: Optional[str] = None,
    ):
        """
        Save model checkpoint.
        
        Args:
            model_state: Model state dict
            optimizer_state: Optimizer state dict (optional)
            epoch: Epoch number (optional)
            metrics: Dictionary of metrics to save (optional)
            is_best: Whether this is the best model so far
            filename: Custom filename (optional, defaults to checkpoint_epoch{epoch}.pt or best.pt)
        """
        checkpoint = {
            "model_state_dict": model_state,
            "epoch": epoch,
        }
        
        if optimizer_state is not None:
            checkpoint["optimizer_state_dict"] = optimizer_state
        
        if metrics is not None:
            checkpoint["metrics"] = metrics
        
        # Determine filename
        if filename is None:
            if is_best:
                filename = "best.pt"
            elif epoch is not None:
                filename = f"checkpoint_epoch_{epoch}.pt"
            else:
                filename = "checkpoint.pt"
        
        checkpoint_path = self.checkpoint_dir / filename
        import torch
        torch.save(checkpoint, checkpoint_path)
        
        # Log checkpoint to W&B
        if is_best:
            wandb.run.summary["best_epoch"] = epoch
            if metrics:
                for key, value in metrics.items():
                    wandb.run.summary[f"best_{key}"] = value
    
    def finish(self):
        """Finish the W&B run."""
        wandb.finish()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()

