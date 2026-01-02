"""
Multi-Scale Deformable Attention Module.

Ported and adapted from the official Deformable DETR implementation:
https://github.com/fundamentalvision/Deformable-DETR

This module implements multi-scale deformable attention, which allows the model
to attend to features at multiple scales with learnable sampling offsets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple


def ms_deform_attn_core_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: List[Tuple[int, int]],
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    level_start_index: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Core computation for multi-scale deformable attention (PyTorch implementation).
    
    This is a pure PyTorch implementation. For better performance, the official
    repo uses CUDA kernels, but this version is sufficient for v0.
    
    Args:
        value: Value tensor of shape (B, num_keys, num_heads, C // num_heads)
        value_spatial_shapes: List of (H, W) tuples for each feature level
        sampling_locations: Sampling locations of shape (B, num_queries, num_heads, num_levels, num_points, 2)
        attention_weights: Attention weights of shape (B, num_queries, num_heads, num_levels, num_points)
        level_start_index: Optional start indices for each level (default: None)
        
    Returns:
        Output tensor of shape (B, num_queries, num_heads, C // num_heads)
    """
    B, num_keys, num_heads, C_head = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape
    
    # Split value by levels
    value_list = value.split([H * W for H, W in value_spatial_shapes], dim=1)
    
    sampling_grids = 2 * sampling_locations - 1  # Normalize to [-1, 1]
    sampling_value_list = []
    
    for level_id, (H, W) in enumerate(value_spatial_shapes):
        # Reshape value for this level: (B, H*W, num_heads, C_head) -> (B*num_heads, C_head, H, W)
        value_l_ = value_list[level_id]  # (B, H*W, num_heads, C_head)
        value_l_ = value_l_.reshape(B, H, W, num_heads, C_head)  # (B, H, W, num_heads, C_head)
        value_l_ = value_l_.permute(0, 3, 4, 1, 2).contiguous()  # (B, num_heads, C_head, H, W)
        # Flatten batch and heads for grid_sample: (B*num_heads, C_head, H, W)
        # Use view instead of flatten to ensure proper reshaping (requires contiguous)
        value_l_ = value_l_.view(B * num_heads, C_head, H, W)  # (B*num_heads, C_head, H, W)
        
        # Extract sampling locations for this level
        # sampling_grids: (B, num_queries, n_heads, n_levels, n_points, 2)
        # Extract for this level: (B, num_queries, n_heads, n_points, 2)
        # Note: In encoder self-attention, num_queries == num_keys (all levels combined)
        # So we extract the grid for all queries, but only use the values for this level
        sampling_grid_l = sampling_grids[:, :, :, level_id, :, :]  # (B, num_queries, n_heads, n_points, 2)
        # Reshape to (B*n_heads, num_queries, n_points, 2)
        sampling_grid_l = sampling_grid_l.permute(0, 2, 1, 3, 4).contiguous()  # (B, n_heads, num_queries, n_points, 2)
        sampling_grid_l = sampling_grid_l.view(B * num_heads, num_queries, num_points, 2)
        
        # Extract attention weights for this level
        attention_weight_l = attention_weights[:, :, :, level_id]  # (B, num_queries, n_heads, n_points)
        attention_weight_l = attention_weight_l.permute(0, 2, 1, 3).contiguous()  # (B, n_heads, num_queries, n_points)
        attention_weight_l = attention_weight_l.view(B * num_heads, num_queries, num_points)
        
        # Sample using grid_sample
        # grid_sample expects:
        #   input: (N, C, H_in, W_in) - 4D spatial
        #   grid: (N, H_out, W_out, 2) - 4D where N matches input batch
        # We have:
        #   value_l_: (B*n_heads, C_head, H, W) ✓
        #   sampling_grid_l: (B*n_heads, num_queries, n_points, 2) ✓
        # This should work - grid_sample will sample num_queries * n_points locations
        sampling_value_l = F.grid_sample(
            value_l_,  # (B*n_heads, C_head, H, W)
            sampling_grid_l,  # (B*n_heads, num_queries, n_points, 2) - treated as (N, H_out=num_queries, W_out=n_points, 2)
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        )  # (B*n_heads, C_head, num_queries, num_points)
        
        sampling_value_l = sampling_value_l.transpose(1, 2)  # (B*num_heads, num_queries, C_head, num_points)
        
        # Weighted sum over sampling points
        attention_weight_l = attention_weight_l.unsqueeze(2)  # (B*num_heads, num_queries, 1, num_points)
        sampling_value_l = (sampling_value_l * attention_weight_l).sum(dim=-1)
        # (B*num_heads, num_queries, C_head)
        
        sampling_value_list.append(sampling_value_l)
    
    # Concatenate across levels
    output = torch.stack(sampling_value_list, dim=-2)  # (B*num_heads, num_queries, num_levels, C_head)
    output = output.sum(dim=2)  # (B*num_heads, num_queries, C_head)
    
    # Reshape back
    output = output.reshape(B, num_heads, num_queries, C_head).transpose(1, 2)
    # (B, num_queries, num_heads, C_head)
    
    return output


class MultiScaleDeformableAttention(nn.Module):
    """
    Multi-Scale Deformable Attention Module.
    
    Adapted from: https://github.com/fundamentalvision/Deformable-DETR
    
    This module performs attention over multi-scale feature maps with learnable
    sampling offsets, allowing the model to focus on relevant regions at different scales.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
    ):
        """
        Initialize multi-scale deformable attention.
        
        Args:
            d_model: Model dimension (default: 256)
            n_levels: Number of feature levels (default: 4)
            n_heads: Number of attention heads (default: 8)
            n_points: Number of sampling points per attention head (default: 4)
        """
        super().__init__()
        
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        
        # Linear projections
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters."""
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        # Initialize offsets to sample around reference points
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.n_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        
        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)
        
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)
        
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.0)
    
    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        input_flatten: torch.Tensor,
        input_spatial_shapes: List[Tuple[int, int]],
        input_level_start_index: Optional[torch.Tensor] = None,
        input_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            query: Query tensor of shape (B, num_queries, d_model)
            reference_points: Reference points of shape (B, num_queries, n_levels, 2)
                             in normalized coordinates [0, 1]
            input_flatten: Flattened input features of shape (B, num_keys, d_model)
            input_spatial_shapes: List of (H, W) tuples for each feature level
            input_level_start_index: Optional start indices for each level (default: None)
            input_padding_mask: Optional padding mask of shape (B, num_keys) (default: None)
            
        Returns:
            Output tensor of shape (B, num_queries, d_model)
        """
        B, num_queries, _ = query.shape
        B, num_keys, _ = input_flatten.shape
        num_levels = len(input_spatial_shapes)
        
        # Value projection
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        
        value = value.view(B, num_keys, self.n_heads, self.d_model // self.n_heads)
        
        # Compute sampling offsets and attention weights
        sampling_offsets = self.sampling_offsets(query).view(
            B, num_queries, self.n_heads, num_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            B, num_queries, self.n_heads, num_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            B, num_queries, self.n_heads, num_levels, self.n_points
        )
        
        # Normalize reference points to [0, 1] if needed
        if reference_points.shape[-1] == 2:
            # Reference points shape: (B, num_queries, n_levels, 2)
            # Expand to match sampling_offsets: (B, num_queries, n_heads, n_levels, n_points, 2)
            offset_normalizer = torch.stack(
                [torch.tensor([H, W], device=query.device, dtype=torch.float32) for H, W in input_spatial_shapes],
                dim=0
            )
            # reference_points: (B, num_queries, n_levels, 2)
            # sampling_offsets: (B, num_queries, n_heads, n_levels, n_points, 2)
            # We need to expand reference_points to match sampling_offsets shape
            # Add dimensions for n_heads and n_points
            ref_points_expanded = reference_points.unsqueeze(2).unsqueeze(4)  # (B, num_queries, 1, n_levels, 1, 2)
            # Expand to match sampling_offsets: (B, num_queries, n_heads, n_levels, n_points, 2)
            ref_points_expanded = ref_points_expanded.expand(-1, -1, self.n_heads, -1, self.n_points, -1)
            
            # offset_normalizer: (n_levels, 2) -> expand to match sampling_offsets
            # Shape should be (1, 1, 1, n_levels, 1, 2)
            offset_norm_expanded = offset_normalizer.view(1, 1, 1, num_levels, 1, 2)
            
            # sampling_offsets: (B, num_queries, n_heads, n_levels, n_points, 2)
            # Divide by spatial dimensions to normalize offsets
            sampling_locations = (
                ref_points_expanded
                + sampling_offsets / offset_norm_expanded.clamp(min=1.0)
            )
        else:
            raise ValueError("Reference points must have shape (B, num_queries, n_levels, 2)")
        
        # Apply deformable attention
        output = ms_deform_attn_core_pytorch(
            value, input_spatial_shapes, sampling_locations, attention_weights,
            level_start_index=input_level_start_index
        )
        # output shape: (B, num_queries, num_heads, C_head)
        
        # Reshape to combine heads: (B, num_queries, d_model)
        output = output.reshape(B, num_queries, self.d_model)
        
        # Output projection
        output = self.output_proj(output)
        
        return output

