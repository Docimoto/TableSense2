"""
Evaluation metrics for table detection.

Implements EoB (Error of Boundaries) and IoU (Intersection over Union) metrics.
"""

from typing import Tuple, List
import numpy as np


def compute_eob(gt_box: Tuple[int, int, int, int], pred_box: Tuple[int, int, int, int]) -> float:
    """
    Compute Error of Boundaries (EoB) between ground truth and predicted boxes.
    
    EoB = max(|x1-x1'|, |y1-y1'|, |x2-x2'|, |y2-y2'|)
    
    Args:
        gt_box: Ground truth box as (col_left, row_top, col_right, row_bottom) in 1-based indices
        pred_box: Predicted box as (col_left, row_top, col_right, row_bottom) in 1-based indices
        
    Returns:
        EoB value (non-negative float)
        
    Examples:
        >>> compute_eob((1, 1, 10, 10), (1, 1, 10, 10))
        0.0
        >>> compute_eob((1, 1, 10, 10), (2, 1, 10, 10))
        1.0
        >>> compute_eob((1, 1, 10, 10), (1, 2, 11, 10))
        1.0
    """
    gt_col_left, gt_row_top, gt_col_right, gt_row_bottom = gt_box
    pred_col_left, pred_row_top, pred_col_right, pred_row_bottom = pred_box
    
    eob = max(
        abs(gt_col_left - pred_col_left),
        abs(gt_row_top - pred_row_top),
        abs(gt_col_right - pred_col_right),
        abs(gt_row_bottom - pred_row_bottom)
    )
    
    return float(eob)


def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """
    Compute Intersection over Union (IoU) between two boxes.
    
    Args:
        box1: First box as (col_left, row_top, col_right, row_bottom) in 1-based indices
        box2: Second box as (col_left, row_top, col_right, row_bottom) in 1-based indices
        
    Returns:
        IoU value in [0, 1]
        
    Examples:
        >>> compute_iou((1, 1, 10, 10), (1, 1, 10, 10))
        1.0
        >>> compute_iou((1, 1, 10, 10), (11, 11, 20, 20))
        0.0
    """
    col_left1, row_top1, col_right1, row_bottom1 = box1
    col_left2, row_top2, col_right2, row_bottom2 = box2
    
    # Compute intersection
    inter_col_left = max(col_left1, col_left2)
    inter_row_top = max(row_top1, row_top2)
    inter_col_right = min(col_right1, col_right2)
    inter_row_bottom = min(row_bottom1, row_bottom2)
    
    if inter_col_left > inter_col_right or inter_row_top > inter_row_bottom:
        # No intersection
        return 0.0
    
    inter_area = (inter_col_right - inter_col_left + 1) * (inter_row_bottom - inter_row_top + 1)
    
    # Compute union
    area1 = (col_right1 - col_left1 + 1) * (row_bottom1 - row_top1 + 1)
    area2 = (col_right2 - col_left2 + 1) * (row_bottom2 - row_top2 + 1)
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def compute_giou(
    box1: Tuple[int, int, int, int],
    box2: Tuple[int, int, int, int]
) -> float:
    """
    Compute Generalized IoU (GIoU) between two boxes.
    
    GIoU = IoU - |C \\ (A ∪ B)| / |C|
    where C is the smallest enclosing box.
    
    Args:
        box1: First box as (col_left, row_top, col_right, row_bottom) in 1-based indices
        box2: Second box as (col_left, row_top, col_right, row_bottom) in 1-based indices
        
    Returns:
        GIoU value in [-1, 1] (typically in [0, 1] for overlapping boxes)
    """
    col_left1, row_top1, col_right1, row_bottom1 = box1
    col_left2, row_top2, col_right2, row_bottom2 = box2
    
    # Compute intersection
    inter_col_left = max(col_left1, col_left2)
    inter_row_top = max(row_top1, row_top2)
    inter_col_right = min(col_right1, col_right2)
    inter_row_bottom = min(row_bottom1, row_bottom2)
    
    if inter_col_left > inter_col_right or inter_row_top > inter_row_bottom:
        inter_area = 0
    else:
        inter_area = (inter_col_right - inter_col_left + 1) * (inter_row_bottom - inter_row_top + 1)
    
    # Compute union
    area1 = (col_right1 - col_left1 + 1) * (row_bottom1 - row_top1 + 1)
    area2 = (col_right2 - col_left2 + 1) * (row_bottom2 - row_top2 + 1)
    union_area = area1 + area2 - inter_area
    
    # Compute smallest enclosing box C
    c_col_left = min(col_left1, col_left2)
    c_row_top = min(row_top1, row_top2)
    c_col_right = max(col_right1, col_right2)
    c_row_bottom = max(row_bottom1, row_bottom2)
    c_area = (c_col_right - c_col_left + 1) * (c_row_bottom - c_row_top + 1)
    
    if union_area == 0:
        return 0.0
    
    iou = inter_area / union_area
    
    # Compute GIoU
    giou = iou - (c_area - union_area) / c_area if c_area > 0 else iou
    
    return giou


def compute_normalized_l1(
    gt_box: Tuple[int, int, int, int],
    pred_box: Tuple[int, int, int, int],
    sheet_width: int,
    sheet_height: int,
) -> float:
    """
    Compute normalized L1 distance between boxes, normalized by sheet diagonal.
    
    Normalized L1 = (|x1p-x1g| + |y1p-y1g| + |x2p-x2g| + |y2p-y2g|) / D
    where D = sqrt(W^2 + H^2) is the sheet diagonal.
    
    Args:
        gt_box: Ground truth box as (col_left, row_top, col_right, row_bottom) in 1-based indices
        pred_box: Predicted box as (col_left, row_top, col_right, row_bottom) in 1-based indices
        sheet_width: Sheet width in cells
        sheet_height: Sheet height in cells
        
    Returns:
        Normalized L1 distance (non-negative float)
    """
    gt_col_left, gt_row_top, gt_col_right, gt_row_bottom = gt_box
    pred_col_left, pred_row_top, pred_col_right, pred_row_bottom = pred_box
    
    # Compute L1 distance
    l1 = (
        abs(pred_col_left - gt_col_left) +
        abs(pred_row_top - gt_row_top) +
        abs(pred_col_right - gt_col_right) +
        abs(pred_row_bottom - gt_row_bottom)
    )
    
    # Normalize by diagonal
    diagonal = np.sqrt(sheet_width ** 2 + sheet_height ** 2)
    if diagonal == 0:
        return 0.0
    
    return l1 / diagonal


def compute_percentile_iou(iou_values: List[float], percentile: float) -> float:
    """
    Compute percentile IoU from a list of IoU values.
    
    Args:
        iou_values: List of IoU values
        percentile: Percentile to compute (e.g., 10 for 10th percentile)
        
    Returns:
        Percentile IoU value, or 0.0 if list is empty
    """
    if not iou_values:
        return 0.0
    
    return float(np.percentile(iou_values, percentile))

