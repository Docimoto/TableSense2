"""
2D Positional Encoding for multi-scale feature maps.

Provides positional encodings for Deformable DETR encoder/decoder.
"""

import torch
import torch.nn as nn
import math
from typing import List, Tuple


class PositionalEncoding2D(nn.Module):
    """
    2D Positional encoding using sinusoidal functions.
    
    Generates positional encodings for multi-scale feature maps to provide
    spatial information to the transformer.
    """
    
    def __init__(self, d_model: int = 256, temperature: int = 10000):
        """
        Initialize 2D positional encoding.
        
        Args:
            d_model: Model dimension (default: 256)
            temperature: Temperature parameter for sinusoidal encoding (default: 10000)
        """
        super().__init__()
        
        if d_model % 4 != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by 4 for 2D positional encoding")
        
        self.d_model = d_model
        self.temperature = temperature
    
    def forward(
        self, spatial_shapes: List[Tuple[int, int]]
    ) -> List[torch.Tensor]:
        """
        Generate positional encodings for each feature level.
        
        Args:
            spatial_shapes: List of (H, W) tuples for each feature level
            
        Returns:
            List of positional encoding tensors, each of shape (H*W, d_model)
        """
        encodings = []
        
        for H, W in spatial_shapes:
            # Create coordinate grids
            y_embed = torch.arange(H, dtype=torch.float32).unsqueeze(1).repeat(1, W)
            x_embed = torch.arange(W, dtype=torch.float32).unsqueeze(0).repeat(H, 1)
            
            # Normalize coordinates to [0, 1]
            y_embed = y_embed / (H - 1) if H > 1 else y_embed
            x_embed = x_embed / (W - 1) if W > 1 else x_embed
            
            # Scale to [0, 2*pi] for sinusoidal encoding
            y_embed = y_embed * 2 * math.pi
            x_embed = x_embed * 2 * math.pi
            
            # Create positional encoding
            # We need d_model features total, split between x and y
            # Each coordinate (x or y) gets d_model // 2 features
            # Use d_model // 4 different frequencies, each with sin and cos = d_model // 2 per coordinate
            dim_t = torch.arange(self.d_model // 4, dtype=torch.float32)
            dim_t = self.temperature ** (2 * (dim_t // 2) / (self.d_model // 4))
            
            pos_x = x_embed[:, :, None] / dim_t  # (H, W, d_model//4)
            pos_y = y_embed[:, :, None] / dim_t  # (H, W, d_model//4)
            
            # Apply sin and cos to each frequency to get d_model//2 features per coordinate
            # Stack along a new dimension, then flatten
            pos_x_encoded = []
            for i in range(self.d_model // 4):
                pos_x_encoded.append(pos_x[:, :, i:i+1].sin())
                pos_x_encoded.append(pos_x[:, :, i:i+1].cos())
            pos_x = torch.cat(pos_x_encoded, dim=2)  # (H, W, d_model//2)
            
            pos_y_encoded = []
            for i in range(self.d_model // 4):
                pos_y_encoded.append(pos_y[:, :, i:i+1].sin())
                pos_y_encoded.append(pos_y[:, :, i:i+1].cos())
            pos_y = torch.cat(pos_y_encoded, dim=2)  # (H, W, d_model//2)
            
            # Concatenate y and x encodings to get d_model total
            pos = torch.cat((pos_y, pos_x), dim=2)  # (H, W, d_model)
            assert pos.shape[2] == self.d_model, f"Expected d_model={self.d_model}, got {pos.shape[2]}"
            pos = pos.flatten(0, 1)  # (H*W, d_model)
            encodings.append(pos)
        
        return encodings


class LearnedPositionalEncoding2D(nn.Module):
    """
    Learned 2D positional encoding.
    
    Uses learned embeddings instead of sinusoidal functions.
    """
    
    def __init__(self, d_model: int = 256, max_h: int = 1000, max_w: int = 1000):
        """
        Initialize learned positional encoding.
        
        Args:
            d_model: Model dimension (default: 256)
            max_h: Maximum height for positional embeddings (default: 1000)
            max_w: Maximum width for positional embeddings (default: 1000)
        """
        super().__init__()
        
        self.d_model = d_model
        self.max_h = max_h
        self.max_w = max_w
        
        # Learned positional embeddings
        self.pos_embed_h = nn.Embedding(max_h, d_model // 2)
        self.pos_embed_w = nn.Embedding(max_w, d_model // 2)
    
    def forward(
        self, spatial_shapes: List[Tuple[int, int]]
    ) -> List[torch.Tensor]:
        """
        Generate learned positional encodings for each feature level.
        
        Args:
            spatial_shapes: List of (H, W) tuples for each feature level
            
        Returns:
            List of positional encoding tensors, each of shape (H*W, d_model)
        """
        encodings = []
        
        for H, W in spatial_shapes:
            # Create coordinate indices
            y_indices = torch.arange(H).unsqueeze(1).repeat(1, W)
            x_indices = torch.arange(W).unsqueeze(0).repeat(H, 1)
            
            # Clamp to valid range
            y_indices = torch.clamp(y_indices, 0, self.max_h - 1)
            x_indices = torch.clamp(x_indices, 0, self.max_w - 1)
            
            # Get embeddings
            pos_h = self.pos_embed_h(y_indices)  # (H, W, d_model//2)
            pos_w = self.pos_embed_w(x_indices)  # (H, W, d_model//2)
            
            # Concatenate
            pos = torch.cat([pos_h, pos_w], dim=2)  # (H, W, d_model)
            pos = pos.flatten(0, 1)  # (H*W, d_model)
            
            encodings.append(pos)
        
        return encodings

