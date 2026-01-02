"""
Patch stem for ConvNeXt V2 backbone.

Projects cell features (43 channels) to base_width channels using a patch embedding.
"""

import torch
import torch.nn as nn


class PatchStem(nn.Module):
    """
    Patch embedding stem using 4×4 convolution with stride 4.
    
    Projects input cell features (C_cell_features = 43) to base_width channels.
    """
    
    def __init__(self, in_channels: int = 43, out_channels: int = 64):
        """
        Initialize patch stem.
        
        Args:
            in_channels: Number of input channels (cell features, default: 43)
            out_channels: Number of output channels (base_width, default: 64)
        """
        super().__init__()
        
        # 4×4 convolution with stride 4 (patch embedding)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=4, stride=4, padding=1
        )
        
        # LayerNorm
        self.norm = nn.LayerNorm(out_channels, eps=1e-6)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C_in, H, W)
            
        Returns:
            Output tensor of shape (B, C_out, H//4, W//4)
        """
        # Convolution
        x = self.conv(x)  # (B, C_out, H//4, W//4)
        
        # LayerNorm (requires permute to (B, H, W, C))
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        return x

