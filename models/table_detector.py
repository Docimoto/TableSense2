"""
Table Detector using Deformable DETR.

Main model wrapper that composes backbone, encoder, decoder, and head.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional

from .backbone import TableBackbone
from .detection.detr_encoder import DeformableDETREncoder
from .detection.detr_decoder import DeformableDETRDecoder
from .detection.detr_head import DeformableDETRHead
from .detection.positional_encoding import PositionalEncoding2D
from .losses.detr_loss import DeformableDETRLoss


class TableDetector(nn.Module):
    """
    Table detector using Deformable DETR architecture.
    
    Composes:
    - Backbone: ConvNeXt V2 backbone for feature extraction
    - Encoder: Deformable DETR encoder for multi-scale feature processing
    - Decoder: Deformable DETR decoder with object queries
    - Head: Classification and bounding box regression heads
    """
    
    def __init__(
        self,
        # Backbone config
        in_channels: int = 43,
        backbone_base_width: int = 64,
        backbone_depths: List[int] = [3, 3, 9, 3],
        # DETR config
        num_queries: int = 20,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        n_heads: int = 8,
        n_points: int = 4,
        # Head config
        num_classes: int = 2,  # table vs background
        # Loss config
        loss_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize table detector.
        
        Args:
            in_channels: Number of input channels (cell features, default: 43)
            backbone_base_width: Base channel width for backbone (default: 64)
            backbone_depths: Number of blocks per backbone stage (default: [3, 3, 9, 3])
            num_queries: Number of object queries (default: 20)
            num_encoder_layers: Number of encoder layers (default: 6)
            num_decoder_layers: Number of decoder layers (default: 6)
            hidden_dim: Hidden dimension (default: 256)
            dropout: Dropout rate (default: 0.1)
            n_heads: Number of attention heads (default: 8)
            n_points: Number of sampling points per head (default: 4)
            num_classes: Number of classes including background (default: 2)
            loss_weights: Optional dictionary of loss weights (default: None)
        """
        super().__init__()
        
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Backbone
        self.backbone = TableBackbone(
            in_channels=in_channels,
            base_width=backbone_base_width,
            depths=backbone_depths,
            hidden_dim=hidden_dim,
        )
        
        # Number of feature levels (C2, C3, C4, C5 = 4 levels)
        n_levels = 4
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding2D(d_model=hidden_dim)
        
        # Encoder
        self.encoder = DeformableDETREncoder(
            d_model=hidden_dim,
            n_layers=num_encoder_layers,
            d_ffn=hidden_dim * 4,
            dropout=dropout,
            n_levels=n_levels,
            n_heads=n_heads,
            n_points=n_points,
        )
        
        # Decoder
        self.decoder = DeformableDETRDecoder(
            d_model=hidden_dim,
            n_layers=num_decoder_layers,
            d_ffn=hidden_dim * 4,
            dropout=dropout,
            n_levels=n_levels,
            n_heads=n_heads,
            n_points=n_points,
            num_queries=num_queries,
        )
        
        # Detection head
        self.head = DeformableDETRHead(
            d_model=hidden_dim,
            num_classes=num_classes,
        )
        
        # Loss function
        if loss_weights is None:
            loss_weights = {
                'loss_ce': 1.0,
                'loss_bbox': 5.0,
                'loss_giou': 2.0,
            }
        
        self.criterion = DeformableDETRLoss(
            num_classes=num_classes,
            weight_dict=loss_weights,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C_in, H, W) - cell features
            mask: Optional padding mask of shape (B, H, W), True = valid, False = padding
            targets: Optional list of target dictionaries for training, each with keys:
                - 'labels': (num_gt,) class labels
                - 'boxes': (num_gt, 4) ground truth boxes in normalized (cx, cy, w, h)
        
        Returns:
            Dictionary with keys:
            - 'pred_logits': (B, num_queries, num_classes) classification logits
            - 'pred_boxes': (B, num_queries, 4) predicted boxes in normalized (cx, cy, w, h)
            - 'losses': (if targets provided) dictionary of loss values
        """
        # Backbone: extract multi-scale features
        features, masks = self.backbone(x, mask=mask)
        # features: List of (B, C, H, W) tensors [C2, C3, C4, C5] with uniform channels
        # masks: List of (B, H, W) boolean tensors, True = valid, False = padding
        
        # Get spatial shapes
        spatial_shapes = [(f.shape[2], f.shape[3]) for f in features]
        
        # Compute valid_ratios per level from masks for coordinate normalization
        # valid_ratios[i] = (valid_h / H, valid_w / W) for level i
        valid_ratios = []
        if masks is not None:
            for mask_level, (H, W) in zip(masks, spatial_shapes):
                # Count valid positions per row and column
                valid_h = mask_level.sum(dim=2).float()  # (B, H) - valid positions per row
                valid_w = mask_level.sum(dim=1).float()  # (B, W) - valid positions per column
                # Normalize by spatial dimensions
                valid_ratio_h = valid_h / W  # (B, H)
                valid_ratio_w = valid_w / H  # (B, W)
                # Get max valid ratio (for normalization)
                max_valid_h = valid_ratio_h.max(dim=1)[0]  # (B,)
                max_valid_w = valid_ratio_w.max(dim=1)[0]  # (B,)
                valid_ratios.append(torch.stack([max_valid_w, max_valid_h], dim=1))  # (B, 2)
        else:
            # If no masks, assume all regions are valid
            B = features[0].shape[0]
            for H, W in spatial_shapes:
                valid_ratios.append(torch.ones(B, 2, device=x.device))  # (B, 2), all 1.0
        
        # Generate positional encodings
        pos_embeds = self.pos_encoding(spatial_shapes)
        # List of (H*W, hidden_dim) tensors
        # Move to same device as input
        pos_embeds = [pos.to(x.device) for pos in pos_embeds]
        
        # Create reference points for encoder (center of each spatial location)
        # For multi-scale, we need reference points for each level
        # Reference points are normalized using valid_ratios to account for padding
        B = features[0].shape[0]
        reference_points_list = []
        for level_idx, (H, W) in enumerate(spatial_shapes):
            # Get valid ratios for this level
            valid_ratio = valid_ratios[level_idx]  # (B, 2) = (valid_w, valid_h)
            
            # Create grid of reference points in normalized coordinates [0, 1]
            y_coords = torch.arange(H, dtype=torch.float32, device=x.device)
            x_coords = torch.arange(W, dtype=torch.float32, device=x.device)
            y_coords = (y_coords + 0.5) / H  # Normalize to [0, 1]
            x_coords = (x_coords + 0.5) / W  # Normalize to [0, 1]
            
            # Create meshgrid
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            ref_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)  # (H*W, 2)
            ref_points = ref_points.unsqueeze(0).expand(B, -1, -1)  # (B, H*W, 2)
            
            # Normalize by valid_ratios to account for padding
            # ref_points are in [0, 1] relative to full feature map
            # We scale them by valid_ratio to get coordinates relative to valid region
            valid_ratio_expanded = valid_ratio.unsqueeze(1)  # (B, 1, 2)
            ref_points = ref_points * valid_ratio_expanded  # (B, H*W, 2)
            
            # Expand for multi-level attention (each point attends to all levels)
            ref_points = ref_points.unsqueeze(2).expand(-1, -1, len(spatial_shapes), -1)
            # (B, H*W, n_levels, 2)
            
            reference_points_list.append(ref_points)
        
        # Concatenate reference points
        reference_points_encoder = torch.cat(reference_points_list, dim=1)
        # (B, num_keys, n_levels, 2)
        
        # Encoder
        memory = self.encoder(
            srcs=features,
            pos_embeds=pos_embeds,
            reference_points=reference_points_encoder,
            spatial_shapes=spatial_shapes,
            padding_masks=masks,
        )
        # memory: (B, num_keys, hidden_dim)
        
        # Decoder
        hidden_states, reference_points_decoder = self.decoder(
            memory=memory,
            spatial_shapes=spatial_shapes,
            padding_mask=None,  # Can add if needed
        )
        # hidden_states: (B, num_queries, hidden_dim)
        # reference_points_decoder: (B, num_queries, n_levels, 2)
        
        # Detection head
        pred_logits, pred_boxes = self.head(hidden_states)
        # pred_logits: (B, num_queries, num_classes)
        # pred_boxes: (B, num_queries, 4)
        
        # Prepare output
        output = {
            'pred_logits': pred_logits,
            'pred_boxes': pred_boxes,
        }
        
        # Compute losses if targets provided
        if targets is not None:
            losses = self.criterion(output, targets)
            output['losses'] = losses
        
        return output

