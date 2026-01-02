"""
Tests for table evaluator.
"""

import pytest
from evaluation.evaluator import TableEvaluator


def test_evaluator_perfect_match():
    """Test evaluator with perfect matches."""
    evaluator = TableEvaluator(eob_threshold=2.0)
    gt_boxes = [(1, 1, 10, 10), (15, 15, 20, 20)]
    pred_boxes = [(1, 1, 10, 10), (15, 15, 20, 20)]
    
    result = evaluator.evaluate(gt_boxes, pred_boxes)
    assert result['precision'] == 1.0
    assert result['recall'] == 1.0
    assert result['f1'] == 1.0
    assert result['tp'] == 2
    assert result['fp'] == 0
    assert result['fn'] == 0


def test_evaluator_false_positive():
    """Test evaluator with false positives."""
    evaluator = TableEvaluator(eob_threshold=2.0)
    gt_boxes = [(1, 1, 10, 10)]
    pred_boxes = [(1, 1, 10, 10), (15, 15, 20, 20)]  # One extra prediction
    
    result = evaluator.evaluate(gt_boxes, pred_boxes)
    assert result['precision'] == 0.5  # 1 TP / 2 predictions
    assert result['recall'] == 1.0  # 1 TP / 1 GT
    assert result['tp'] == 1
    assert result['fp'] == 1
    assert result['fn'] == 0


def test_evaluator_false_negative():
    """Test evaluator with false negatives."""
    evaluator = TableEvaluator(eob_threshold=2.0)
    gt_boxes = [(1, 1, 10, 10), (15, 15, 20, 20)]
    pred_boxes = [(1, 1, 10, 10)]  # Missing one prediction
    
    result = evaluator.evaluate(gt_boxes, pred_boxes)
    assert result['precision'] == 1.0  # 1 TP / 1 prediction
    assert result['recall'] == 0.5  # 1 TP / 2 GT
    assert result['tp'] == 1
    assert result['fp'] == 0
    assert result['fn'] == 1


def test_evaluator_eob_threshold():
    """Test evaluator with EoB threshold."""
    evaluator = TableEvaluator(eob_threshold=2.0)
    gt_boxes = [(1, 1, 10, 10)]
    pred_boxes = [(2, 2, 11, 11)]  # EoB = 1.0 (within threshold)
    
    result = evaluator.evaluate(gt_boxes, pred_boxes)
    assert result['tp'] == 1
    
    # Now with EoB > threshold
    pred_boxes = [(5, 5, 15, 15)]  # EoB = 4.0 (outside threshold)
    result = evaluator.evaluate(gt_boxes, pred_boxes)
    assert result['tp'] == 0
    assert result['fp'] == 1


def test_evaluator_empty():
    """Test evaluator with empty inputs."""
    evaluator = TableEvaluator(eob_threshold=2.0)
    
    # Both empty
    result = evaluator.evaluate([], [])
    assert result['precision'] == 1.0
    assert result['recall'] == 1.0
    assert result['f1'] == 1.0
    
    # Empty predictions
    result = evaluator.evaluate([(1, 1, 10, 10)], [])
    assert result['precision'] == 0.0
    assert result['recall'] == 0.0
    assert result['fn'] == 1
    
    # Empty ground truth
    result = evaluator.evaluate([], [(1, 1, 10, 10)])
    assert result['precision'] == 0.0
    assert result['recall'] == 0.0
    assert result['fp'] == 1

