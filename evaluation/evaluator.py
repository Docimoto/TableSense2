"""
Table detection evaluator.

Implements evaluation metrics (Precision, Recall, F1) using EoB-based matching.
Supports three evaluation modes: CNN-only, Excel-only, and Hybrid.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np

from evaluation.metrics import compute_eob


class TableEvaluator:
    """
    Evaluates table detection predictions against ground truth.
    
    Uses greedy one-to-one matching based on EoB (Error of Boundaries).
    A prediction is considered a True Positive (TP) if matched with EoB <= threshold.
    """
    
    def __init__(self, eob_threshold: float = 2.0):
        """
        Initialize evaluator.
        
        Args:
            eob_threshold: EoB threshold for considering a match as TP (default: 2.0)
        """
        self.eob_threshold = eob_threshold
    
    def evaluate(
        self,
        gt_boxes: List[Tuple[int, int, int, int]],
        pred_boxes: List[Tuple[int, int, int, int]],
        pred_scores: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate predictions against ground truth.
        
        Args:
            gt_boxes: List of ground truth boxes as (col_left, row_top, col_right, row_bottom)
            pred_boxes: List of predicted boxes as (col_left, row_top, col_right, row_bottom)
            pred_scores: Optional list of prediction confidence scores
            
        Returns:
            Dictionary with metrics:
            - 'precision': Precision score
            - 'recall': Recall score
            - 'f1': F1 score
            - 'tp': Number of true positives
            - 'fp': Number of false positives
            - 'fn': Number of false negatives
            - 'eob_mean': Mean EoB for matched pairs
            - 'eob_std': Standard deviation of EoB for matched pairs
        """
        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            # Perfect match (no tables)
            return {
                'precision': 1.0,
                'recall': 1.0,
                'f1': 1.0,
                'tp': 0,
                'fp': 0,
                'fn': 0,
                'eob_mean': 0.0,
                'eob_std': 0.0,
            }
        
        if len(gt_boxes) == 0:
            # All predictions are false positives
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'tp': 0,
                'fp': len(pred_boxes),
                'fn': 0,
                'eob_mean': 0.0,
                'eob_std': 0.0,
            }
        
        if len(pred_boxes) == 0:
            # All ground truths are false negatives
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'tp': 0,
                'fp': 0,
                'fn': len(gt_boxes),
                'eob_mean': 0.0,
                'eob_std': 0.0,
            }
        
        # Greedy one-to-one matching by increasing EoB
        matches = self._greedy_match(gt_boxes, pred_boxes)
        
        # Count TP, FP, FN
        tp = 0
        matched_gt_indices = set()
        matched_pred_indices = set()
        eob_values = []
        
        for gt_idx, pred_idx, eob in matches:
            if eob <= self.eob_threshold:
                tp += 1
                matched_gt_indices.add(gt_idx)
                matched_pred_indices.add(pred_idx)
                eob_values.append(eob)
        
        fp = len(pred_boxes) - len(matched_pred_indices)
        fn = len(gt_boxes) - len(matched_gt_indices)
        
        # Compute metrics
        precision = tp / len(pred_boxes) if len(pred_boxes) > 0 else 0.0
        recall = tp / len(gt_boxes) if len(gt_boxes) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        eob_mean = np.mean(eob_values) if eob_values else 0.0
        eob_std = np.std(eob_values) if eob_values else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'eob_mean': eob_mean,
            'eob_std': eob_std,
        }
    
    def _greedy_match(
        self,
        gt_boxes: List[Tuple[int, int, int, int]],
        pred_boxes: List[Tuple[int, int, int, int]],
    ) -> List[Tuple[int, int, float]]:
        """
        Perform greedy one-to-one matching between GT and predictions.
        
        Matches are made by increasing EoB (best matches first).
        
        Args:
            gt_boxes: List of ground truth boxes
            pred_boxes: List of predicted boxes
            
        Returns:
            List of tuples (gt_idx, pred_idx, eob) for matched pairs
        """
        # Compute all pairwise EoB values
        eob_matrix = []
        for gt_box in gt_boxes:
            row = []
            for pred_box in pred_boxes:
                eob = compute_eob(gt_box, pred_box)
                row.append(eob)
            eob_matrix.append(row)
        
        eob_matrix = np.array(eob_matrix)
        
        # Greedy matching: repeatedly find the best unmatched pair
        matches = []
        used_gt = set()
        used_pred = set()
        
        # Create list of all possible matches sorted by EoB
        all_matches = []
        for gt_idx in range(len(gt_boxes)):
            for pred_idx in range(len(pred_boxes)):
                eob = eob_matrix[gt_idx, pred_idx]
                all_matches.append((eob, gt_idx, pred_idx))
        
        all_matches.sort()  # Sort by EoB (ascending)
        
        # Greedily assign matches
        for eob, gt_idx, pred_idx in all_matches:
            if gt_idx not in used_gt and pred_idx not in used_pred:
                matches.append((gt_idx, pred_idx, eob))
                used_gt.add(gt_idx)
                used_pred.add(pred_idx)
        
        return matches
    
    def evaluate_batch(
        self,
        gt_batch: List[List[Tuple[int, int, int, int]]],
        pred_batch: List[List[Tuple[int, int, int, int]]],
        pred_scores_batch: Optional[List[List[float]]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate a batch of predictions.
        
        Args:
            gt_batch: List of ground truth box lists (one per sheet)
            pred_batch: List of predicted box lists (one per sheet)
            pred_scores_batch: Optional list of prediction score lists
            
        Returns:
            Aggregated metrics across all sheets
        """
        all_tp = 0
        all_fp = 0
        all_fn = 0
        all_eob_values = []
        
        for gt_boxes, pred_boxes in zip(gt_batch, pred_batch):
            result = self.evaluate(gt_boxes, pred_boxes)
            all_tp += result['tp']
            all_fp += result['fp']
            all_fn += result['fn']
            
            # Collect EoB values from matched pairs
            matches = self._greedy_match(gt_boxes, pred_boxes)
            for gt_idx, pred_idx, eob in matches:
                if eob <= self.eob_threshold:
                    all_eob_values.append(eob)
        
        total_pred = all_tp + all_fp
        total_gt = all_tp + all_fn
        
        precision = all_tp / total_pred if total_pred > 0 else 0.0
        recall = all_tp / total_gt if total_gt > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        eob_mean = np.mean(all_eob_values) if all_eob_values else 0.0
        eob_std = np.std(all_eob_values) if all_eob_values else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': all_tp,
            'fp': all_fp,
            'fn': all_fn,
            'eob_mean': eob_mean,
            'eob_std': eob_std,
        }

