"""
Hungarian evaluator for table detection with IoU/GIoU-based matching.

Uses Hungarian matching (optimal assignment) with IoU/GIoU cost matrix
to compute per-image statistics for bucket-based visualization.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment

from evaluation.metrics import compute_iou, compute_giou, compute_normalized_l1, compute_percentile_iou


class HungarianEvaluator:
    """
    Evaluator using Hungarian matching with IoU/GIoU for optimal assignment.
    
    Computes per-image statistics needed for bucket classification:
    - T, P, M, U_t, U_p, miss_rate, extra_rate
    - mean_iou, p10_iou, mean_nL1
    """
    
    def __init__(self, use_giou: bool = True):
        """
        Initialize Hungarian evaluator.
        
        Args:
            use_giou: If True, use GIoU for matching; if False, use IoU
        """
        self.use_giou = use_giou
    
    def evaluate_image(
        self,
        gt_boxes: List[Tuple[int, int, int, int]],
        pred_boxes: List[Tuple[int, int, int, int]],
        sheet_width: int,
        sheet_height: int,
    ) -> Dict[str, float]:
        """
        Evaluate a single image and return per-image statistics.
        
        Args:
            gt_boxes: List of GT boxes as (col_left, row_top, col_right, row_bottom)
            pred_boxes: List of predicted boxes as (col_left, row_top, col_right, row_bottom)
            sheet_width: Sheet width in cells
            sheet_height: Sheet height in cells
            
        Returns:
            Dictionary with per-image statistics:
            - 'T': Number of targets (GT boxes)
            - 'P': Number of predictions
            - 'M': Number of matched pairs
            - 'U_t': Number of unmatched targets (misses)
            - 'U_p': Number of unmatched predictions (extras)
            - 'miss_rate': U_t / max(T, 1)
            - 'extra_rate': U_p / max(T, 1)
            - 'mean_iou': Mean IoU over matched pairs (0 if M=0)
            - 'p10_iou': 10th percentile IoU over matched pairs (0 if M=0)
            - 'mean_nL1': Mean normalized L1 over matched pairs (large sentinel if M=0)
            - 'matched_pairs': List of (gt_idx, pred_idx, iou, nL1) tuples
            - 'unmatched_gt_indices': List of unmatched GT indices
            - 'unmatched_pred_indices': List of unmatched prediction indices
        """
        T = len(gt_boxes)
        P = len(pred_boxes)
        
        # Handle edge cases
        if T == 0 and P == 0:
            return {
                'T': 0,
                'P': 0,
                'M': 0,
                'U_t': 0,
                'U_p': 0,
                'miss_rate': 0.0,
                'extra_rate': 0.0,
                'mean_iou': 0.0,
                'p10_iou': 0.0,
                'mean_nL1': 0.0,
                'matched_pairs': [],
                'unmatched_gt_indices': [],
                'unmatched_pred_indices': [],
            }
        
        if T == 0:
            # All predictions are extras
            return {
                'T': 0,
                'P': P,
                'M': 0,
                'U_t': 0,
                'U_p': P,
                'miss_rate': 0.0,
                'extra_rate': float('inf') if T == 0 else (P / max(T, 1)),
                'mean_iou': 0.0,
                'p10_iou': 0.0,
                'mean_nL1': 1000.0,  # Large sentinel
                'matched_pairs': [],
                'unmatched_gt_indices': [],
                'unmatched_pred_indices': list(range(P)),
            }
        
        if P == 0:
            # All GT are misses
            return {
                'T': T,
                'P': 0,
                'M': 0,
                'U_t': T,
                'U_p': 0,
                'miss_rate': 1.0,
                'extra_rate': 0.0,
                'mean_iou': 0.0,
                'p10_iou': 0.0,
                'mean_nL1': 1000.0,  # Large sentinel
                'matched_pairs': [],
                'unmatched_gt_indices': list(range(T)),
                'unmatched_pred_indices': [],
            }
        
        # Compute cost matrix using IoU or GIoU
        cost_matrix = np.zeros((T, P))
        for gt_idx, gt_box in enumerate(gt_boxes):
            for pred_idx, pred_box in enumerate(pred_boxes):
                if self.use_giou:
                    overlap = compute_giou(gt_box, pred_box)
                else:
                    overlap = compute_iou(gt_box, pred_box)
                # Use negative overlap as cost (we want to maximize overlap)
                cost_matrix[gt_idx, pred_idx] = -overlap
        
        # Hungarian matching
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Extract matched pairs with their IoU and nL1 values
        matched_pairs = []
        matched_gt_indices = set()
        matched_pred_indices = set()
        iou_values = []
        nL1_values = []
        
        for gt_idx, pred_idx in zip(row_indices, col_indices):
            gt_box = gt_boxes[gt_idx]
            pred_box = pred_boxes[pred_idx]
            
            # Compute IoU (always use IoU for metrics, even if matching used GIoU)
            iou = compute_iou(gt_box, pred_box)
            nL1 = compute_normalized_l1(gt_box, pred_box, sheet_width, sheet_height)
            
            matched_pairs.append((gt_idx, pred_idx, iou, nL1))
            matched_gt_indices.add(gt_idx)
            matched_pred_indices.add(pred_idx)
            iou_values.append(iou)
            nL1_values.append(nL1)
        
        M = len(matched_pairs)
        U_t = T - M
        U_p = P - M
        
        # Compute rates
        miss_rate = U_t / max(T, 1)
        extra_rate = U_p / max(T, 1)
        
        # Compute matched statistics
        if M > 0:
            mean_iou = float(np.mean(iou_values))
            p10_iou = compute_percentile_iou(iou_values, 10.0)
            mean_nL1 = float(np.mean(nL1_values))
        else:
            mean_iou = 0.0
            p10_iou = 0.0
            mean_nL1 = 1000.0  # Large sentinel
        
        # Find unmatched indices
        unmatched_gt_indices = [i for i in range(T) if i not in matched_gt_indices]
        unmatched_pred_indices = [i for i in range(P) if i not in matched_pred_indices]
        
        return {
            'T': T,
            'P': P,
            'M': M,
            'U_t': U_t,
            'U_p': U_p,
            'miss_rate': miss_rate,
            'extra_rate': extra_rate,
            'mean_iou': mean_iou,
            'p10_iou': p10_iou,
            'mean_nL1': mean_nL1,
            'matched_pairs': matched_pairs,
            'unmatched_gt_indices': unmatched_gt_indices,
            'unmatched_pred_indices': unmatched_pred_indices,
        }
    
    def evaluate_batch(
        self,
        gt_batch: List[List[Tuple[int, int, int, int]]],
        pred_batch: List[List[Tuple[int, int, int, int]]],
        sheet_widths: List[int],
        sheet_heights: List[int],
    ) -> List[Dict[str, float]]:
        """
        Evaluate a batch of images and return per-image statistics.
        
        Args:
            gt_batch: List of GT box lists (one per sheet)
            pred_batch: List of predicted box lists (one per sheet)
            sheet_widths: List of sheet widths (one per sheet)
            sheet_heights: List of sheet heights (one per sheet)
            
        Returns:
            List of per-image statistics dictionaries
        """
        results = []
        for gt_boxes, pred_boxes, width, height in zip(gt_batch, pred_batch, sheet_widths, sheet_heights):
            result = self.evaluate_image(gt_boxes, pred_boxes, width, height)
            results.append(result)
        return results
