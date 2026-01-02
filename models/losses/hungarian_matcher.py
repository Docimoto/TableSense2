"""
Hungarian Matcher for DETR.

Implements optimal one-to-one matching between predictions and ground truth
using the Hungarian algorithm.
"""

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from typing import Tuple


class HungarianMatcher(nn.Module):
    """
    Hungarian matcher for optimal assignment between predictions and ground truth.
    
    Uses scipy's linear_sum_assignment (Hungarian algorithm) to find the optimal
    one-to-one matching that minimizes the total cost.
    """
    
    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
    ):
        """
        Initialize Hungarian matcher.
        
        Args:
            cost_class: Weight for classification cost (default: 1.0)
            cost_bbox: Weight for L1 bbox cost (default: 5.0)
            cost_giou: Weight for GIoU cost (default: 2.0)
        """
        super().__init__()
        
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
    
    @torch.no_grad()
    def forward(
        self,
        outputs: dict,
        targets: list,
    ) -> list:
        """
        Compute optimal assignment between predictions and ground truth.
        
        Args:
            outputs: Dictionary with keys:
                - 'pred_logits': (B, num_queries, num_classes) classification logits
                - 'pred_boxes': (B, num_queries, 4) predicted boxes in normalized (cx, cy, w, h)
            targets: List of dicts, one per batch item, each with keys:
                - 'labels': (num_gt,) class labels
                - 'boxes': (num_gt, 4) ground truth boxes in normalized (cx, cy, w, h)
        
        Returns:
            List of tuples (row_indices, col_indices) for each batch item,
            where row_indices are GT indices and col_indices are prediction indices
        """
        B = outputs['pred_logits'].shape[0]
        num_queries = outputs['pred_logits'].shape[1]
        
        # Compute cost matrix for each batch item
        indices = []
        
        for b in range(B):
            out_prob = outputs['pred_logits'][b].softmax(-1)  # (num_queries, num_classes)
            out_bbox = outputs['pred_boxes'][b]  # (num_queries, 4)
            
            tgt_ids = targets[b]['labels']  # (num_gt,)
            tgt_bbox = targets[b]['boxes']  # (num_gt, 4)
            
            # Handle empty targets
            if len(tgt_ids) == 0:
                # No ground truth - all queries match to background
                indices.append((torch.tensor([], dtype=torch.long, device=out_bbox.device),
                               torch.tensor([], dtype=torch.long, device=out_bbox.device)))
                continue
            
            # Classification cost: -log(p(class))
            cost_class = -out_prob[:, tgt_ids.long()]  # (num_queries, num_gt)
            
            # L1 cost
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)  # (num_queries, num_gt)
            
            # GIoU cost
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox),
                box_cxcywh_to_xyxy(tgt_bbox)
            )  # (num_queries, num_gt)
            
            # Total cost
            C = (
                self.cost_class * cost_class
                + self.cost_bbox * cost_bbox
                + self.cost_giou * cost_giou
            )
            
            # Hungarian algorithm
            C = C.cpu().numpy()
            row_indices, col_indices = linear_sum_assignment(C)
            
            # Convert back to torch tensors
            row_indices = torch.as_tensor(row_indices, dtype=torch.long, device=out_bbox.device)
            col_indices = torch.as_tensor(col_indices, dtype=torch.long, device=out_bbox.device)
            
            indices.append((row_indices, col_indices))
        
        return indices


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format."""
    cx, cy, w, h = x.unbind(-1)
    b = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    """Convert boxes from (x1, y1, x2, y2) to (cx, cy, w, h) format."""
    x1, y1, x2, y2 = x.unbind(-1)
    b = [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)]
    return torch.stack(b, dim=-1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """Compute area of boxes."""
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise IoU between two sets of boxes.
    
    Args:
        boxes1: (N, 4) boxes in (x1, y1, x2, y2) format
        boxes2: (M, 4) boxes in (x1, y1, x2, y2) format
        
    Returns:
        (N, M) IoU matrix
    """
    area1 = box_area(boxes1)  # (N,)
    area2 = box_area(boxes2)  # (M,)
    
    # Compute intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)
    
    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)
    
    # Union
    union = area1[:, None] + area2 - inter
    
    iou = inter / union
    return iou


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise Generalized IoU (GIoU) between two sets of boxes.
    
    Args:
        boxes1: (N, 4) boxes in (x1, y1, x2, y2) format
        boxes2: (M, 4) boxes in (x1, y1, x2, y2) format
        
    Returns:
        (N, M) GIoU matrix
    """
    # IoU
    iou = box_iou(boxes1, boxes2)
    
    # Area of boxes
    area1 = box_area(boxes1)  # (N,)
    area2 = box_area(boxes2)  # (M,)
    
    # Enclosing box
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)
    
    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    area_enclosing = wh[:, :, 0] * wh[:, :, 1]  # (N, M)
    
    # GIoU
    giou = iou - (area_enclosing - (area1[:, None] + area2 - iou * area_enclosing)) / area_enclosing
    
    return giou

