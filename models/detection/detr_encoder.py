"""
Deformable DETR Encoder.

Multi-scale transformer encoder using deformable attention.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional

from .deformable_attention import MultiScaleDeformableAttention


class DeformableDETREncoderLayer(nn.Module):
    """Single encoder layer with deformable attention and FFN."""
    
    def __init__(
        self,
        d_model: int = 256,
        d_ffn: int = 1024,
        dropout: float = 0.1,
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
    ):
        """
        Initialize encoder layer.
        
        Args:
            d_model: Model dimension (default: 256)
            d_ffn: Feed-forward network dimension (default: 1024)
            dropout: Dropout rate (default: 0.1)
            n_levels: Number of feature levels (default: 4)
            n_heads: Number of attention heads (default: 8)
            n_points: Number of sampling points per head (default: 4)
        """
        super().__init__()
        
        # Multi-scale deformable attention
        self.self_attn = MultiScaleDeformableAttention(
            d_model=d_model,
            n_levels=n_levels,
            n_heads=n_heads,
            n_points=n_points,
        )
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(
        self,
        src: torch.Tensor,
        pos: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: List[Tuple[int, int]],
        level_start_index: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            src: Input features of shape (B, num_keys, d_model)
            pos: Positional encodings of shape (num_keys, d_model)
            reference_points: Reference points of shape (B, num_keys, n_levels, 2)
            spatial_shapes: List of (H, W) tuples for each level
            level_start_index: Optional start indices for each level
            padding_mask: Optional padding mask of shape (B, num_keys)
            
        Returns:
            Output features of shape (B, num_keys, d_model)
        """
        # Self-attention with residual
        src2 = self.self_attn(
            query=src,
            reference_points=reference_points,
            input_flatten=src,
            input_spatial_shapes=spatial_shapes,
            input_level_start_index=level_start_index,
            input_padding_mask=padding_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # FFN with residual
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class DeformableDETREncoder(nn.Module):
    """
    Deformable DETR Encoder.
    
    Multi-scale transformer encoder that processes multi-scale feature maps
    from the backbone using deformable attention.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 6,
        d_ffn: int = 1024,
        dropout: float = 0.1,
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
    ):
        """
        Initialize encoder.
        
        Args:
            d_model: Model dimension (default: 256)
            n_layers: Number of encoder layers (default: 6)
            d_ffn: Feed-forward network dimension (default: 1024)
            dropout: Dropout rate (default: 0.1)
            n_levels: Number of feature levels (default: 4)
            n_heads: Number of attention heads (default: 8)
            n_points: Number of sampling points per head (default: 4)
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_levels = n_levels
        
        # Encoder layers
        self.layers = nn.ModuleList([
            DeformableDETREncoderLayer(
                d_model=d_model,
                d_ffn=d_ffn,
                dropout=dropout,
                n_levels=n_levels,
                n_heads=n_heads,
                n_points=n_points,
            )
            for _ in range(n_layers)
        ])
        
        # Note: Feature projection is handled by adapters in the backbone.
        # Features passed to this encoder should already have uniform channels (d_model).
    
    def forward(
        self,
        srcs: List[torch.Tensor],
        pos_embeds: List[torch.Tensor],
        reference_points: torch.Tensor,
        spatial_shapes: List[Tuple[int, int]],
        level_start_index: Optional[torch.Tensor] = None,
        padding_masks: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            srcs: List of multi-scale feature maps, each of shape (B, C, H, W)
                  Features should already have uniform channels (C == d_model) from adapters
            pos_embeds: List of positional encodings, each of shape (H*W, d_model)
            reference_points: Reference points of shape (B, num_keys, n_levels, 2)
            spatial_shapes: List of (H, W) tuples for each level
            level_start_index: Optional start indices for each level
            padding_masks: Optional list of padding masks, each of shape (B, H, W)
            
        Returns:
            Encoded features of shape (B, num_keys, d_model)
        """
        # Flatten and concatenate multi-scale features
        src_flatten = []
        spatial_shapes_flat = []
        
        for i, src in enumerate(srcs):
            B, C, H, W = src.shape
            spatial_shapes_flat.append((H, W))
            
            # Verify channels match d_model (features should be pre-projected by adapters)
            assert C == self.d_model, (
                f"Feature level {i} has {C} channels but expected {self.d_model}. "
                "Features should be pre-projected by adapters in the backbone."
            )
            
            # Flatten: (B, C, H, W) -> (B, H*W, C)
            src = src.flatten(2).transpose(1, 2)
            
            src_flatten.append(src)
        
        # Concatenate all levels
        src_flatten = torch.cat(src_flatten, dim=1)  # (B, num_keys, d_model)
        
        # Concatenate positional encodings
        pos_embed_flatten = torch.cat(pos_embeds, dim=0)  # (num_keys, d_model)
        pos_embed_flatten = pos_embed_flatten.unsqueeze(0).expand(
            src_flatten.shape[0], -1, -1
        )  # (B, num_keys, d_model)
        
        # Add positional encoding
        src_flatten = src_flatten + pos_embed_flatten
        
        # Create padding mask if provided
        padding_mask = None
        if padding_masks is not None:
            mask_flatten = []
            for mask in padding_masks:
                mask_flatten.append(mask.flatten(1))  # (B, H*W)
            padding_mask = torch.cat(mask_flatten, dim=1)  # (B, num_keys)
            padding_mask = ~padding_mask  # Invert: True = padding, False = valid
        
        # Process through encoder layers
        output = src_flatten
        for layer in self.layers:
            output = layer(
                src=output,
                pos=pos_embed_flatten,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes_flat,
                level_start_index=level_start_index,
                padding_mask=padding_mask,
            )
        
        return output

