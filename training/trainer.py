"""
Training loop for table detection models.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm
import time

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from training.experiment_logger import ExperimentLogger


class Trainer:
    """
    Generic trainer for table detection models.
    
    Supports:
    - Variable batch sizes (batch_size=1 for variable sheet sizes)
    - Gradient accumulation
    - Cosine learning rate schedule with warmup
    - Model checkpointing
    - Experiment logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: Any,
        val_loader: Optional[Any] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        logger: Optional[ExperimentLogger] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        gradient_accumulation_steps: int = 8,
        max_epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 3,
        checkpoint_dir: Optional[Path] = None,
        save_every_n_epochs: int = 10,
        log_gradient_norms: bool = False,
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Optional validation data loader
            optimizer: Optional optimizer (defaults to AdamW)
            criterion: Optional loss function (model-specific)
            logger: Optional experiment logger
            device: Device to train on
            gradient_accumulation_steps: Number of steps to accumulate gradients
            max_epochs: Maximum number of training epochs
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            warmup_epochs: Number of warmup epochs
            checkpoint_dir: Directory to save checkpoints
            save_every_n_epochs: Save checkpoint every N epochs
            log_gradient_norms: Whether to log gradient norms (for debugging)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_epochs = max_epochs
        self.logger = logger
        self.save_every_n_epochs = save_every_n_epochs
        self.log_gradient_norms = log_gradient_norms
        
        # Setup optimizer
        if optimizer is None:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999)
            )
        else:
            self.optimizer = optimizer
        
        # Setup learning rate scheduler
        total_steps = len(train_loader) * max_epochs
        warmup_steps = len(train_loader) * warmup_epochs
        
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=1e-6
        )
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        
        # Loss function (can be overridden by model-specific training)
        self.criterion = criterion
        
        # Checkpoint directory
        if checkpoint_dir is None and logger is not None:
            checkpoint_dir = logger.checkpoint_dir
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.best_val_f1 = 0.0
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            # Forward pass
            loss = self._compute_loss(batch)
            
            # Scale loss by accumulation steps
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Log gradient norms if enabled (before optimizer step)
                if self.log_gradient_norms and self.logger:
                    grad_norms = self._compute_gradient_norms()
                    if grad_norms:
                        self.logger.log_gradient_norms(
                            max_grad_norm=grad_norms.get('max'),
                            mean_grad_norm=grad_norms.get('mean'),
                            step=len(self.train_loader) * self.current_epoch + batch_idx
                        )
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item() * self.gradient_accumulation_steps})
        
        # Final gradient step if needed
        if num_batches % self.gradient_accumulation_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Get current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        
        metrics = {
            'loss': avg_loss,
            'lr': current_lr,
        }
        
        return metrics
    
    def _compute_gradient_norms(self) -> Optional[Dict[str, float]]:
        """
        Compute gradient norms across all model parameters.
        
        Returns:
            Dictionary with 'max' and 'mean' gradient norms, or None if no gradients
        """
        total_norm = 0.0
        param_count = 0
        max_norm = 0.0
        
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                max_norm = max(max_norm, param_norm.item())
                param_count += 1
        
        if param_count == 0:
            return None
        
        total_norm = total_norm ** (1. / 2)
        mean_norm = total_norm / param_count if param_count > 0 else 0.0
        
        return {
            'max': max_norm,
            'mean': mean_norm,
        }
    
    def _compute_loss(self, batch: Any) -> torch.Tensor:
        """
        Compute loss for a batch.
        
        This method should be overridden by model-specific trainers.
        Default implementation assumes batch is a dict with 'features' and 'targets'.
        
        Args:
            batch: Training batch
            
        Returns:
            Loss tensor
        """
        if isinstance(batch, dict):
            features = batch['features'].to(self.device)
            targets = batch['targets'].to(self.device)
            
            outputs = self.model(features)
            
            if self.criterion is None:
                # Default: binary cross-entropy for baseline model
                criterion = nn.BCELoss()
            else:
                criterion = self.criterion
            
            loss = criterion(outputs, targets)
            return loss
        else:
            raise ValueError("Batch format not supported. Expected dict with 'features' and 'targets'.")
    
    def validate(self, evaluator: Optional[Any] = None) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            evaluator: Optional evaluator object with evaluate() method
            
        Returns:
            Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        val_loss = 0.0
        num_batches = 0
        
        all_gt_boxes = []
        all_pred_boxes = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                if isinstance(batch, dict):
                    features = batch['features'].to(self.device)
                    targets = batch.get('targets')
                    gt_boxes = batch.get('gt_boxes', [])
                    
                    outputs = self.model(features)
                    
                    # Compute loss if targets available
                    if targets is not None:
                        targets = targets.to(self.device)
                        if self.criterion is None:
                            criterion = nn.BCELoss()
                        else:
                            criterion = self.criterion
                        val_loss += criterion(outputs, targets).item()
                        num_batches += 1
                    
                    # Get predictions for evaluation
                    if hasattr(self.model, 'predict_boxes'):
                        pred_boxes_list = self.model.predict_boxes(outputs)
                        all_pred_boxes.extend(pred_boxes_list)
                        all_gt_boxes.extend(gt_boxes)
        
        metrics = {}
        if num_batches > 0:
            metrics['loss'] = val_loss / num_batches
        
        # Run evaluator if provided
        if evaluator is not None and len(all_gt_boxes) > 0:
            eval_results = evaluator.evaluate_batch(all_gt_boxes, all_pred_boxes)
            metrics.update(eval_results)
        
        return metrics
    
    def train(self, evaluator: Optional[Any] = None, early_stopping_patience: Optional[int] = None):
        """
        Main training loop.
        
        Args:
            evaluator: Optional evaluator for validation
            early_stopping_patience: Optional patience for early stopping (None = disabled)
        """
        best_val_f1 = 0.0
        patience_counter = 0
        
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate(evaluator=evaluator)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Log metrics
            if self.logger:
                self.logger.log_losses(
                    train_loss=train_metrics.get('loss'),
                    val_loss=val_metrics.get('loss'),
                    step=epoch
                )
                
                # Log learning rate and loss ratios for debugging DETR stability
                logs = {}
                if 'lr' in train_metrics:
                    logs['train/lr'] = train_metrics['lr']
                
                # Log detailed loss breakdown if available
                if 'cls_loss' in train_metrics and 'l1_loss' in train_metrics:
                    # Monitor if classification overpowers regression or vice versa
                    logs['train/loss_ratio_cls_l1'] = train_metrics['cls_loss'] / (train_metrics['l1_loss'] + 1e-6)
                
                if logs:
                    self.logger.log_metrics(logs, step=epoch)
                
                # Log system metrics
                try:
                    # GPU memory (if CUDA available)
                    gpu_memory_mb = None
                    if torch.cuda.is_available():
                        gpu_memory_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                        torch.cuda.reset_peak_memory_stats(self.device)
                    
                    # CPU memory (if psutil available)
                    cpu_memory_mb = None
                    if PSUTIL_AVAILABLE:
                        cpu_memory_mb = psutil.Process().memory_info().rss / (1024 ** 2)
                    
                    self.logger.log_system_metrics(
                        gpu_memory_mb=gpu_memory_mb,
                        cpu_memory_mb=cpu_memory_mb,
                        epoch_time_seconds=epoch_time,
                        step=epoch
                    )
                except Exception as e:
                    # Don't fail training if system metrics fail
                    import warnings
                    warnings.warn(f"Failed to log system metrics: {e}")
                
                if 'f1' in val_metrics:
                    self.logger.log_validation_metrics(
                        precision=val_metrics.get('precision', 0.0),
                        recall=val_metrics.get('recall', 0.0),
                        f1=val_metrics.get('f1', 0.0),
                        eob_mean=val_metrics.get('eob_mean'),
                        eob_std=val_metrics.get('eob_std'),
                        tp=val_metrics.get('tp'),
                        fp=val_metrics.get('fp'),
                        fn=val_metrics.get('fn'),
                        step=epoch
                    )
            
            # Save checkpoint
            val_f1 = val_metrics.get('f1', 0.0)
            is_best = val_f1 > best_val_f1
            
            if is_best:
                best_val_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1
            
            if self.checkpoint_dir:
                # Save periodic checkpoint
                if (epoch + 1) % self.save_every_n_epochs == 0:
                    self._save_checkpoint(epoch, train_metrics, val_metrics, is_best=False)
                
                # Save best checkpoint
                if is_best:
                    self._save_checkpoint(epoch, train_metrics, val_metrics, is_best=True)
            
            # Early stopping
            if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        print(f"Training completed. Best validation F1: {best_val_f1:.4f}")
    
    def _save_checkpoint(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        is_best: bool = False
    ):
        """Save model checkpoint."""
        if self.checkpoint_dir is None:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'best_val_f1': self.best_val_f1,
        }
        
        if is_best:
            filename = "best.pt"
            self.best_val_f1 = val_metrics.get('f1', 0.0)
        else:
            filename = f"checkpoint_epoch_{epoch + 1}.pt"
        
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        
        if self.logger:
            self.logger.save_checkpoint(
                model_state=self.model.state_dict(),
                optimizer_state=self.optimizer.state_dict(),
                epoch=epoch,
                metrics=val_metrics,
                is_best=is_best
            )

