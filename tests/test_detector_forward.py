"""
Smoke tests for full detector forward pass.
"""

import torch
import pytest
from models.table_detector import TableDetector


def test_detector_forward_minimal_config():
    """Test detector forward pass with minimal config."""
    detector = TableDetector(
        in_channels=43,
        backbone_base_width=32,  # Minimal
        backbone_depths=[2, 2, 4, 2],  # Minimal
        num_queries=10,  # Minimal
        num_encoder_layers=2,  # Minimal
        num_decoder_layers=2,  # Minimal
        hidden_dim=128,  # Minimal
        n_heads=4,  # Minimal
        n_points=2,  # Minimal
    )
    
    # Dummy input: small sheet
    B, H, W = 1, 100, 100
    x = torch.randn(B, 43, H, W)
    
    # Forward pass (inference)
    output = detector(x)
    
    assert 'pred_logits' in output
    assert 'pred_boxes' in output
    assert output['pred_logits'].shape == (B, 10, 2)
    assert output['pred_boxes'].shape == (B, 10, 4)
    
    # Boxes should be in [0, 1]
    assert output['pred_boxes'].min() >= 0.0
    assert output['pred_boxes'].max() <= 1.0


def test_detector_forward_with_targets():
    """Test detector forward pass with targets (training mode)."""
    detector = TableDetector(
        in_channels=43,
        backbone_base_width=32,
        backbone_depths=[2, 2, 4, 2],
        num_queries=10,
        num_encoder_layers=2,
        num_decoder_layers=2,
        hidden_dim=128,
        n_heads=4,
        n_points=2,
    )
    
    B, H, W = 1, 100, 100
    x = torch.randn(B, 43, H, W)
    
    targets = [
        {
            'labels': torch.tensor([1]),  # One table
            'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2]]),  # Normalized (cx, cy, w, h)
        }
    ]
    
    # Forward pass (training)
    output = detector(x, targets=targets)
    
    assert 'pred_logits' in output
    assert 'pred_boxes' in output
    assert 'losses' in output
    
    # Check losses
    losses = output['losses']
    assert 'loss_ce' in losses
    assert 'loss_bbox' in losses
    assert 'loss_giou' in losses


def test_detector_forward_with_mask():
    """Test detector forward pass with padding mask."""
    detector = TableDetector(
        in_channels=43,
        backbone_base_width=32,
        backbone_depths=[2, 2, 4, 2],
        num_queries=10,
        num_encoder_layers=2,
        num_decoder_layers=2,
        hidden_dim=128,
        n_heads=4,
        n_points=2,
    )
    
    B, H, W = 1, 100, 100
    x = torch.randn(B, 43, H, W)
    mask = torch.ones(B, H, W, dtype=torch.bool)
    
    output = detector(x, mask=mask)
    
    assert 'pred_logits' in output
    assert 'pred_boxes' in output

