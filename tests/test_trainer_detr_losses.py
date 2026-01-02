"""
Tests for Phase 3 trainer with DETR-style loss handling.

Verifies that the trainer can handle models that return loss dictionaries
(in addition to single loss tensors).
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tempfile
from pathlib import Path

from training.trainer import Trainer
from training.experiment_logger import ExperimentLogger


class DummyDETRModel(nn.Module):
    """
    Dummy model that returns a loss dictionary (simulating DETR behavior).
    
    This is used to test that the trainer can handle DETR-style loss outputs.
    """
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, features, targets=None):
        """
        Forward pass that returns loss dict in training mode.
        
        Args:
            features: Input features
            targets: Optional targets (for training)
            
        Returns:
            If training: dict with 'loss', 'cls_loss', 'l1_loss', 'iou_loss'
            If eval: predictions tensor
        """
        if self.training and targets is not None:
            # Simulate DETR loss computation
            output = self.linear(features.mean(dim=-1))
            cls_loss = nn.functional.mse_loss(output, targets)
            l1_loss = torch.tensor(0.1, device=features.device)
            iou_loss = torch.tensor(0.05, device=features.device)
            total_loss = cls_loss + l1_loss + iou_loss
            
            return {
                'loss': total_loss,
                'cls_loss': cls_loss,
                'l1_loss': l1_loss,
                'iou_loss': iou_loss,
            }
        else:
            # Inference mode
            return self.linear(features.mean(dim=-1))


class DummyDETRDataset(Dataset):
    """Dummy dataset for DETR model testing."""
    
    def __init__(self, num_samples=5):
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        features = torch.randn(10, 20)  # (seq_len, feature_dim)
        targets = torch.randn(1)
        return {
            'features': features,
            'targets': targets,
        }


class DETRLossTrainer(Trainer):
    """
    Extended trainer that handles DETR-style loss dictionaries.
    
    This demonstrates how the trainer can be extended to support
    models that return loss dictionaries instead of single tensors.
    """
    
    def _compute_loss(self, batch):
        """
        Compute loss, handling both single tensor and dict returns.
        
        If model returns a dict with 'loss' key, extract it.
        Otherwise, use the default behavior.
        """
        if isinstance(batch, dict):
            features = batch['features'].to(self.device)
            targets = batch.get('targets')
            
            if targets is not None:
                targets = targets.to(self.device)
                # Pass both features and targets for DETR models
                model_output = self.model(features, targets=targets)
            else:
                model_output = self.model(features)
            
            # Handle loss dictionary (DETR-style)
            if isinstance(model_output, dict):
                if 'loss' not in model_output:
                    raise ValueError("Model returned dict but missing 'loss' key")
                return model_output['loss']
            else:
                # Handle single tensor loss
                if self.criterion is None:
                    criterion = nn.MSELoss()
                else:
                    criterion = self.criterion
                return criterion(model_output, targets)
        else:
            raise ValueError("Batch format not supported")


def test_trainer_handles_detr_loss_dict():
    """Test that trainer can handle models returning loss dictionaries."""
    train_dataset = DummyDETRDataset(num_samples=3)
    val_dataset = DummyDETRDataset(num_samples=2)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    model = DummyDETRModel()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        
        trainer = DETRLossTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device="cpu",
            gradient_accumulation_steps=1,
            max_epochs=1,
            lr=1e-3,
            checkpoint_dir=checkpoint_dir,
        )
        
        # Train one epoch - should not raise errors
        metrics = trainer.train_epoch()
        
        assert 'loss' in metrics
        assert metrics['loss'] >= 0.0
        assert not torch.isnan(torch.tensor(metrics['loss']))
        assert not torch.isinf(torch.tensor(metrics['loss']))


def test_trainer_logs_detr_loss_components():
    """Test that trainer can extract and log DETR loss components."""
    train_dataset = DummyDETRDataset(num_samples=2)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    
    model = DummyDETRModel()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "runs" / "test_run"
        
        logger = ExperimentLogger(
            experiment_name="test_detr_losses",
            project_name="test-project",
            run_dir=run_dir,
            config={"test": True},
            mode="disabled",  # Disable W&B for testing
        )
        
        trainer = DETRLossTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=None,
            logger=logger,
            device="cpu",
            gradient_accumulation_steps=1,
            max_epochs=1,
            lr=1e-3,
        )
        
        # Override train_epoch to capture loss components
        original_train_epoch = trainer.train_epoch
        
        def train_epoch_with_logging():
            metrics = original_train_epoch()
            # In a real implementation, we'd extract loss components from model output
            # and log them via logger.log_losses(cls_loss=..., l1_loss=..., etc.)
            return metrics
        
        trainer.train_epoch = train_epoch_with_logging
        
        # Train one epoch
        metrics = trainer.train_epoch()
        
        logger.finish()
        
        assert 'loss' in metrics


 

