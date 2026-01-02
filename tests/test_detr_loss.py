"""
Tests for DETR loss computation.
"""

import torch
import pytest
from models.losses.detr_loss import DeformableDETRLoss
from models.losses.hungarian_matcher import HungarianMatcher


def test_detr_loss():
    """Test DETR loss computation."""
    loss_fn = DeformableDETRLoss(
        num_classes=2,
        weight_dict={
            'loss_ce': 1.0,
            'loss_bbox': 5.0,
            'loss_giou': 2.0,
        },
    )
    
    B = 1
    num_queries = 10
    
    outputs = {
        'pred_logits': torch.randn(B, num_queries, 2),
        'pred_boxes': torch.rand(B, num_queries, 4),  # Normalized [0, 1]
    }
    
    targets = [
        {
            'labels': torch.tensor([1]),  # One table
            'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2]]),  # Normalized (cx, cy, w, h)
        }
    ]
    
    losses = loss_fn(outputs, targets)
    
    assert 'loss_ce' in losses
    assert 'loss_bbox' in losses
    assert 'loss_giou' in losses
    
    # Losses should be scalars
    assert losses['loss_ce'].dim() == 0
    assert losses['loss_bbox'].dim() == 0
    assert losses['loss_giou'].dim() == 0
    
    # Losses should be non-negative
    assert losses['loss_ce'] >= 0
    assert losses['loss_bbox'] >= 0
    assert losses['loss_giou'] >= 0


def test_detr_loss_empty_targets():
    """Test DETR loss with empty targets (all background)."""
    loss_fn = DeformableDETRLoss(num_classes=2)
    
    B = 1
    num_queries = 10
    
    outputs = {
        'pred_logits': torch.randn(B, num_queries, 2),
        'pred_boxes': torch.rand(B, num_queries, 4),
    }
    
    targets = [
        {
            'labels': torch.tensor([]),  # No tables
            'boxes': torch.tensor([]).reshape(0, 4),
        }
    ]
    
    losses = loss_fn(outputs, targets)
    
    # Should still compute classification loss (background)
    assert 'loss_ce' in losses

