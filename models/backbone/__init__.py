"""
Backbone module for table detection.

Provides ConvNeXt V2 backbone that produces multi-scale feature maps
for Deformable DETR encoder.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ConvNeXtV2Encoder


class FeatureAdapter(nn.Module):
    """
    Multi-scale feature adapter.
    
    Applies 1×1 convolution followed by normalization to each feature level
    to ensure uniform channel dimensionality across levels for DETR.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        """
        Initialize feature adapter.
        
        Args:
            in_channels: Input channel dimension
            out_channels: Output channel dimension (typically hidden_dim for DETR)
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.LayerNorm(out_channels, eps=1e-6)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C_in, H, W)
            
        Returns:
            Output tensor of shape (B, C_out, H, W)
        """
        # 1×1 convolution
        x = self.conv(x)  # (B, C_out, H, W)
        
        # Normalization (requires permute to (B, H, W, C))
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        return x


class TableBackbone(nn.Module):
    """
    Table detection backbone wrapper.
    
    Composes stem + encoder to produce multi-scale feature maps for DETR.
    Applies feature adapters (1×1 Conv → Normalization) to each level.
    backbone stages connect directly to Deformable encoder.
    """
    
    def __init__(
        self,
        in_channels: int = 43,
        base_width: int = 64,
        depths: List[int] = [3, 3, 9, 3],
        expansion_ratio: int = 4,
        use_grn: bool = False,
        hidden_dim: int = 256,
    ):
        """
        Initialize table backbone.
        
        Args:
            in_channels: Number of input channels (cell features, default: 43)
            base_width: Base channel width (default: 64)
            depths: Number of blocks per stage (default: [3, 3, 9, 3])
            expansion_ratio: Expansion ratio for MLP in blocks (default: 4)
            use_grn: Whether to use Global Response Normalization (default: False)
            hidden_dim: Output channel dimension for adapters (default: 256)
        """
        super().__init__()
        
        self.encoder = ConvNeXtV2Encoder(
            in_channels=in_channels,
            base_width=base_width,
            depths=depths,
            expansion_ratio=expansion_ratio,
            use_grn=use_grn,
        )
        
        # Stride information for each output feature map
        # C2: stride 4 (stage 0), C3: stride 8 (stage 1), C4: stride 16 (stage 2), C5: stride 32 (stage 3)
        self.strides = [4, 8, 16, 32]
        
        # Channel dimensions for each stage: [C, 2C, 4C, 8C]
        stage_channels = [
            base_width,
            base_width * 2,
            base_width * 4,
            base_width * 8,
        ]
        
        # Feature adapters for each level (C2, C3, C4, C5)
        # Each adapter: 1×1 Conv → Normalization
        self.adapters = nn.ModuleList([
            FeatureAdapter(in_channels=stage_channels[i], out_channels=hidden_dim)
            for i in range(4)
        ])
    
    def forward(
        self, x: torch.Tensor, mask: torch.Tensor = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C_in, H, W)
            mask: Optional padding mask of shape (B, H, W), True = valid, False = padding
                  If None, assumes all regions are valid
            
        Returns:
            Tuple of:
            - feature_maps: List of multi-scale feature maps [C2, C3, C4, C5] with uniform channels
            - masks: List of padding masks for each feature map
        """
        # Get multi-scale features from encoder
        features, masks = self.encoder(x)
        
        # Apply feature adapters to each level (1×1 Conv → Normalization)
        adapted_features = []
        for i, (feat, adapter) in enumerate(zip(features, self.adapters)):
            adapted_feat = adapter(feat)
            adapted_features.append(adapted_feat)
        
        # If input mask provided, propagate it through downsampling
        if mask is not None:
            spatial_shapes = [(feat.shape[2], feat.shape[3]) for feat in features]
            masks = self._propagate_mask(mask, spatial_shapes)
        
        return adapted_features, masks
    
    def _propagate_mask(
        self, mask: torch.Tensor, target_shapes: List[Tuple[int, int]]
    ) -> List[torch.Tensor]:
        """
        Resize input mask to align with each feature map shape using nearest neighbor.
        
        Args:
            mask: Input mask of shape (B, H, W), True = valid, False = padding
            target_shapes: List of (H, W) tuples for each feature level
            
        Returns:
            List of masks matched to feature map spatial dimensions.
        """
        mask_float = mask.float().unsqueeze(1)  # (B, 1, H, W)
        propagated_masks = []
        
        for H, W in target_shapes:
            down = F.interpolate(
                mask_float, size=(H, W), mode='nearest'
            )  # (B, 1, H, W)
            propagated_masks.append(down.squeeze(1).bool())
        
        return propagated_masks

