"""
ConvNeXt V2 building blocks.

Implements depthwise convolution blocks with LayerNorm, channel-wise MLP, and GELU activation.
Based on ConvNeXt V2 architecture for efficient feature extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt V2 block with depthwise convolution, LayerNorm, and channel-wise MLP.
    
    Architecture:
    - Depthwise convolution (3×3)
    - LayerNorm
    - Channel-wise MLP (expansion ratio configurable)
    - GELU activation
    - Optional Global Response Normalization (GRN)
    - Residual connection
    """
    
    def __init__(
        self,
        dim: int,
        expansion_ratio: int = 4,
        use_grn: bool = False,
        drop_path: float = 0.0,
    ):
        """
        Initialize ConvNeXt block.
        
        Args:
            dim: Input/output channel dimension
            expansion_ratio: Expansion ratio for channel-wise MLP (default: 4)
            use_grn: Whether to use Global Response Normalization (default: False)
            drop_path: Drop path rate for stochastic depth (default: 0.0)
        """
        super().__init__()
        
        # Depthwise convolution (3×3)
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, padding=1, groups=dim
        )
        
        # LayerNorm
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        # Channel-wise MLP
        expanded_dim = dim * expansion_ratio
        self.pwconv1 = nn.Linear(dim, expanded_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expanded_dim, dim)
        
        # Optional Global Response Normalization
        self.use_grn = use_grn
        if use_grn:
            self.grn = GlobalResponseNorm(dim)
        
        # Drop path (stochastic depth)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        residual = x
        
        # Depthwise convolution
        x = self.dwconv(x)
        
        # Permute to (B, H, W, C) for LayerNorm
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)
        
        # Channel-wise MLP
        x = self.pwconv1(x)
        x = self.act(x)
        if self.use_grn:
            x = self.grn(x)
        x = self.pwconv2(x)
        
        # Permute back to (B, C, H, W)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # Residual connection + drop path
        x = residual + self.drop_path(x)
        
        return x


class GlobalResponseNorm(nn.Module):
    """
    Global Response Normalization (GRN) layer.
    
    Normalizes features across spatial dimensions to improve training stability.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize GRN layer.
        
        Args:
            dim: Channel dimension
            eps: Small epsilon for numerical stability (default: 1e-6)
        """
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, H, W, C)
            
        Returns:
            Normalized tensor of shape (B, H, W, C)
        """
        # Compute global response: L2 norm across spatial dimensions
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)  # (B, 1, 1, C)
        
        # Normalize
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + self.eps)  # (B, 1, 1, C)
        
        # Apply learnable scaling and bias
        x = self.gamma * (x * Nx) + self.beta + x
        
        return x


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.
    
    When applied to main path, randomly zeroes some samples during training.
    """
    
    def __init__(self, drop_prob: float = 0.0):
        """
        Initialize drop path.
        
        Args:
            drop_prob: Drop probability (default: 0.0)
        """
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor (possibly zeroed during training)
        """
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        random_tensor = keep_prob + torch.rand(
            x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype
        )
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

