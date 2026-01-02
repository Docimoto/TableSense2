"""
Deformable DETR Decoder.

Transformer decoder with object queries and deformable cross-attention.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict

from .deformable_attention import MultiScaleDeformableAttention


class DeformableDETRDecoderLayer(nn.Module):
    """Single decoder layer with self-attention, cross-attention, and FFN."""
    
    def __init__(
        self,
        d_model: int = 256,
        d_ffn: int = 1024,
        dropout: float = 0.1,
        n_levels: int = 3,
        n_heads: int = 8,
        n_points: int = 4,
    ):
        """
        Initialize decoder layer.
        
        Args:
            d_model: Model dimension (default: 256)
            d_ffn: Feed-forward network dimension (default: 1024)
            dropout: Dropout rate (default: 0.1)
            n_levels: Number of feature levels (default: 4)
            n_heads: Number of attention heads (default: 8)
            n_points: Number of sampling points per head (default: 4)
        """
        super().__init__()
        
        # Self-attention (standard multi-head attention)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # Cross-attention (deformable attention to encoder features)
        self.cross_attn = MultiScaleDeformableAttention(
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
        self.dropout4 = nn.Dropout(dropout)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
    
    def forward(
        self,
        tgt: torch.Tensor,
        query_pos: torch.Tensor,
        memory: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: List[Tuple[int, int]],
        level_start_index: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            tgt: Target queries of shape (B, num_queries, d_model)
            query_pos: Query positional encodings of shape (B, num_queries, d_model)
            memory: Encoder output of shape (B, num_keys, d_model)
            reference_points: Reference points of shape (B, num_queries, n_levels, 2)
            spatial_shapes: List of (H, W) tuples for each level
            level_start_index: Optional start indices for each level
            padding_mask: Optional padding mask of shape (B, num_keys)
            
        Returns:
            Decoded queries of shape (B, num_queries, d_model)
        """
        # Self-attention with residual
        q = k = tgt + query_pos
        tgt2 = self.self_attn(q, k, tgt, need_weights=False)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention with residual
        tgt2 = self.cross_attn(
            query=tgt + query_pos,
            reference_points=reference_points,
            input_flatten=memory,
            input_spatial_shapes=spatial_shapes,
            input_level_start_index=level_start_index,
            input_padding_mask=padding_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # FFN with residual
        tgt2 = self.linear2(self.dropout1(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


class DeformableDETRDecoder(nn.Module):
    """
    Deformable DETR Decoder.
    
    Transformer decoder with learned object queries that attend to encoder features
    using deformable attention.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 6,
        d_ffn: int = 1024,
        dropout: float = 0.1,
        n_levels: int = 3,
        n_heads: int = 8,
        n_points: int = 4,
        num_queries: int = 100,
    ):
        """
        Initialize decoder.
        
        Args:
            d_model: Model dimension (default: 256)
            n_layers: Number of decoder layers (default: 6)
            d_ffn: Feed-forward network dimension (default: 1024)
            dropout: Dropout rate (default: 0.1)
            n_levels: Number of feature levels (default: 3)
            n_heads: Number of attention heads (default: 8)
            n_points: Number of sampling points per head (default: 4)
            num_queries: Number of object queries (default: 100)
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_queries = num_queries
        
        # Learned object queries
        self.query_embed = nn.Embedding(num_queries, d_model)
        
        # Learned reference points (normalized coordinates [0, 1])
        self.reference_points = nn.Linear(d_model, 2)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            DeformableDETRDecoderLayer(
                d_model=d_model,
                d_ffn=d_ffn,
                dropout=dropout,
                n_levels=n_levels,
                n_heads=n_heads,
                n_points=n_points,
            )
            for _ in range(n_layers)
        ])
        
        # Layer norm for output
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        memory: torch.Tensor,
        spatial_shapes: List[Tuple[int, int]],
        level_start_index: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            memory: Encoder output of shape (B, num_keys, d_model)
            spatial_shapes: List of (H, W) tuples for each level
            level_start_index: Optional start indices for each level
            padding_mask: Optional padding mask of shape (B, num_keys)
            
        Returns:
            Tuple of:
            - decoded queries: (B, num_queries, d_model)
            - reference points: (B, num_queries, n_levels, 2)
        """
        B = memory.shape[0]
        
        # Initialize object queries
        query_embed = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        # (B, num_queries, d_model)
        
        # Initialize reference points from query embeddings
        reference_points = self.reference_points(query_embed).sigmoid()
        # (B, num_queries, 2)
        
        # Expand reference points for multi-level attention
        # Each query attends to all levels
        reference_points_multi_level = reference_points.unsqueeze(2).expand(
            -1, -1, len(spatial_shapes), -1
        )  # (B, num_queries, n_levels, 2)
        
        # Process through decoder layers
        tgt = query_embed
        for layer in self.layers:
            tgt = layer(
                tgt=tgt,
                query_pos=query_embed,
                memory=memory,
                reference_points=reference_points_multi_level,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                padding_mask=padding_mask,
            )
        
        # Final layer norm
        tgt = self.norm(tgt)
        
        return tgt, reference_points_multi_level

