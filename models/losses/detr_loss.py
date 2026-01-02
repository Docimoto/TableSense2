"""
Deformable DETR Loss.

Implements classification, L1, and IoU/GIoU losses for DETR training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple

from .hungarian_matcher import HungarianMatcher, box_cxcywh_to_xyxy, generalized_box_iou


class DeformableDETRLoss(nn.Module):
    """
    Loss function for Deformable DETR.
    
    Computes classification loss, L1 loss, and GIoU loss for matched predictions,
    and classification loss for unmatched predictions (background class).
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        matcher: HungarianMatcher = None,
        weight_dict: Dict[str, float] = None,
        eos_coef: float = 0.1,
    ):
        """
        Initialize DETR loss.
        
        Args:
            num_classes: Number of classes including background (default: 2)
            matcher: Hungarian matcher instance (default: None, creates new one)
            weight_dict: Dictionary of loss weights (default: None, uses defaults)
            eos_coef: Weight for background class in classification loss (default: 0.1)
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.matcher = matcher if matcher is not None else HungarianMatcher()
        
        # Default loss weights
        if weight_dict is None:
            weight_dict = {
                'loss_ce': 1.0,
                'loss_bbox': 5.0,
                'loss_giou': 2.0,
            }
        self.weight_dict = weight_dict
        
        self.eos_coef = eos_coef
        
        # Empty class weight for focal loss (if needed)
        empty_weight = torch.ones(self.num_classes)
        empty_weight[0] = self.eos_coef  # Background class
        self.register_buffer('empty_weight', empty_weight)
    
    def loss_labels(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        num_boxes: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute classification loss.
        
        Args:
            outputs: Dictionary with 'pred_logits' key
            targets: List of target dictionaries
            indices: List of (row_indices, col_indices) from matcher
            num_boxes: Total number of boxes (for normalization)
            
        Returns:
            Dictionary with 'loss_ce' key
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # (B, num_queries, num_classes)
        
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([
            t['labels'][J] for t, (_, J) in zip(targets, indices)
        ])
        target_classes = torch.full(
            src_logits.shape[:2],
            0,  # Background class
            dtype=torch.int64,
            device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        
        # Cross-entropy loss
        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2),
            target_classes,
            self.empty_weight
        )
        
        losses = {'loss_ce': loss_ce}
        return losses
    
    def loss_boxes(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        num_boxes: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute L1 and GIoU losses.
        
        Args:
            outputs: Dictionary with 'pred_boxes' key
            targets: List of target dictionaries
            indices: List of (row_indices, col_indices) from matcher
            num_boxes: Total number of boxes (for normalization)
            
        Returns:
            Dictionary with 'loss_bbox' and 'loss_giou' keys
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]  # (num_matched, 4)
        target_boxes = torch.cat([
            t['boxes'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)  # (num_matched, 4)
        
        # L1 loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = loss_bbox.sum() / num_boxes
        if loss_bbox.dim() > 0:
            loss_bbox = loss_bbox.squeeze()
        
        # GIoU loss
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)
        ))
        loss_giou = loss_giou.sum() / num_boxes
        if loss_giou.dim() > 0:
            loss_giou = loss_giou.squeeze()
        
        losses = {'loss_bbox': loss_bbox, 'loss_giou': loss_giou}
        return losses
    
    def _get_src_permutation_idx(self, indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get permutation indices for source (predictions)."""
        batch_idx = torch.cat([
            torch.full_like(src, i) for i, (src, _) in enumerate(indices)
        ])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.
        
        Args:
            outputs: Dictionary with keys:
                - 'pred_logits': (B, num_queries, num_classes)
                - 'pred_boxes': (B, num_queries, 4)
            targets: List of dicts, one per batch item, each with keys:
                - 'labels': (num_gt,) class labels
                - 'boxes': (num_gt, 4) ground truth boxes
        
        Returns:
            Dictionary with loss values
        """
        # Compute optimal matching
        indices = self.matcher(outputs, targets)
        
        # Compute number of boxes (for normalization)
        num_boxes = sum(len(t['labels']) for t in targets)
        if num_boxes == 0:
            # Handle empty targets case - only classification loss
            num_boxes = torch.as_tensor([1.0], dtype=torch.float, device=next(iter(outputs.values())).device)
            losses = {}
            # Classification loss for background
            src_logits = outputs['pred_logits']  # (B, num_queries, num_classes)
            target_classes = torch.full(
                src_logits.shape[:2],
                0,  # Background class
                dtype=torch.int64,
                device=src_logits.device
            )
            loss_ce = F.cross_entropy(
                src_logits.transpose(1, 2),
                target_classes,
                self.empty_weight
            )
            losses['loss_ce'] = loss_ce * self.weight_dict.get('loss_ce', 1.0)
            losses['loss_bbox'] = torch.tensor(0.0, device=src_logits.device)
            losses['loss_giou'] = torch.tensor(0.0, device=src_logits.device)
            return losses
        
        num_boxes = torch.as_tensor(
            [num_boxes],
            dtype=torch.float,
            device=next(iter(outputs.values())).device
        )
        
        # Compute losses
        losses = {}
        
        # Classification loss
        losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
        
        # Box losses (only for matched predictions)
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))
        
        # Apply loss weights
        for k in losses.keys():
            if k in self.weight_dict:
                losses[k] *= self.weight_dict[k]
        
        return losses

