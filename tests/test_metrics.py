"""
Tests for evaluation metrics.
"""

import pytest
from evaluation.metrics import compute_eob, compute_iou, compute_giou


def test_compute_eob_exact_match():
    """Test EoB for exact match."""
    gt_box = (1, 1, 10, 10)
    pred_box = (1, 1, 10, 10)
    assert compute_eob(gt_box, pred_box) == 0.0


def test_compute_eob_offset():
    """Test EoB for offset boxes."""
    gt_box = (1, 1, 10, 10)
    pred_box = (2, 1, 10, 10)  # Offset by 1 column
    assert compute_eob(gt_box, pred_box) == 1.0
    
    pred_box = (1, 2, 11, 10)  # Offset by 1 row and 1 column
    assert compute_eob(gt_box, pred_box) == 1.0


def test_compute_iou_exact_match():
    """Test IoU for exact match."""
    box1 = (1, 1, 10, 10)
    box2 = (1, 1, 10, 10)
    assert compute_iou(box1, box2) == 1.0


def test_compute_iou_no_overlap():
    """Test IoU for non-overlapping boxes."""
    box1 = (1, 1, 10, 10)
    box2 = (11, 11, 20, 20)
    assert compute_iou(box1, box2) == 0.0


def test_compute_iou_partial_overlap():
    """Test IoU for partially overlapping boxes."""
    box1 = (1, 1, 10, 10)  # Area = 100
    box2 = (6, 6, 15, 15)  # Area = 100, overlap = 25
    # Intersection = 25, Union = 100 + 100 - 25 = 175
    expected_iou = 25 / 175
    assert abs(compute_iou(box1, box2) - expected_iou) < 1e-6


def test_compute_giou():
    """Test GIoU computation."""
    box1 = (1, 1, 10, 10)
    box2 = (1, 1, 10, 10)
    giou = compute_giou(box1, box2)
    assert abs(giou - 1.0) < 1e-6  # Should be close to 1.0 for exact match

