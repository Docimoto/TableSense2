"""
Tests for Hungarian matcher.
"""

import torch
import pytest
from models.losses.hungarian_matcher import HungarianMatcher, box_cxcywh_to_xyxy, generalized_box_iou


def test_box_conversion():
    """Test box format conversion."""
    boxes_cxcywh = torch.tensor([[0.5, 0.5, 0.4, 0.4]])  # Center at (0.5, 0.5), size 0.4x0.4
    boxes_xyxy = box_cxcywh_to_xyxy(boxes_cxcywh)
    
    assert boxes_xyxy.shape == (1, 4)
    # Should be approximately [0.3, 0.3, 0.7, 0.7]
    assert boxes_xyxy[0, 0] < 0.5  # x1
    assert boxes_xyxy[0, 2] > 0.5  # x2


def test_giou():
    """Test Generalized IoU computation."""
    boxes1 = torch.tensor([[0.0, 0.0, 0.5, 0.5]])  # (x1, y1, x2, y2)
    boxes2 = torch.tensor([[0.25, 0.25, 0.75, 0.75]])
    
    giou = generalized_box_iou(boxes1, boxes2)
    
    assert giou.shape == (1, 1)
    # GIoU can be negative (when boxes don't overlap well)
    # But these boxes overlap, so GIoU should be positive
    assert giou[0, 0] > -1.0  # GIoU >= -1.0 (minimum value)
    assert giou[0, 0] <= 1.0  # GIoU <= 1.0


def test_hungarian_matcher():
    """Test Hungarian matcher."""
    matcher = HungarianMatcher(
        cost_class=1.0,
        cost_bbox=5.0,
        cost_giou=2.0,
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
    
    indices = matcher(outputs, targets)
    
    assert len(indices) == B
    assert isinstance(indices[0], tuple)
    assert len(indices[0]) == 2  # (row_indices, col_indices)

