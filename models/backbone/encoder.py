"""
ConvNeXt V2 encoder with multi-stage architecture.

Produces multi-scale feature maps for Deformable DETR encoder.
"""

import torch
import torch.nn as nn
from typing import List, Tuple
from .blocks import ConvNeXtBlock
from .stem import PatchStem


class ConvNeXtV2Encoder(nn.Module):
    """
    ConvNeXt V2 encoder with 4 stages producing multi-scale features.
    
    Architecture:
    - Patch stem (4×4 conv, stride 4)
    - 4 stages with channels [C, 2C, 4C, 8C]
    - Each stage: multiple ConvNeXt blocks + stride-2 downsampling
    - Outputs: C2 (stride 4), C3 (stride 8), C4 (stride 16), C5 (stride 32)
    """
    
    def __init__(
        self,
        in_channels: int = 43,
        base_width: int = 64,
        depths: List[int] = [3, 3, 9, 3],
        expansion_ratio: int = 4,
        use_grn: bool = False,
    ):
        """
        Initialize ConvNeXt V2 encoder.
        
        Args:
            in_channels: Number of input channels (cell features, default: 43)
            base_width: Base channel width (default: 64)
            depths: Number of blocks per stage (default: [3, 3, 9, 3])
            expansion_ratio: Expansion ratio for MLP in blocks (default: 4)
            use_grn: Whether to use Global Response Normalization (default: False)
        """
        super().__init__()
        
        self.base_width = base_width
        self.depths = depths
        
        # Patch stem: projects 43 channels → base_width channels
        self.stem = PatchStem(in_channels=in_channels, out_channels=base_width)
        
        # Stage channels: [C, 2C, 4C, 8C]
        stage_channels = [
            base_width,
            base_width * 2,
            base_width * 4,
            base_width * 8,
        ]
        
        # Build stages
        self.stages = nn.ModuleList()
        for i in range(4):
            # Stage 0: input is base_width from stem, output is base_width (no downsampling)
            # Stage 1+: downsample from previous stage
            in_ch = stage_channels[i - 1] if i > 0 else base_width
            stage = self._make_stage(
                in_channels=in_ch,
                out_channels=stage_channels[i],
                num_blocks=depths[i],
                expansion_ratio=expansion_ratio,
                use_grn=use_grn,
                downsample=i > 0,  # Downsample between stages (not before stage 0)
            )
            self.stages.append(stage)
        
        # Output feature map indices for multi-scale detection
        # C2: stage 0 (stride 4), C3: stage 1 (stride 8), C4: stage 2 (stride 16), C5: stage 3 (stride 32)
        self.out_indices = [0, 1, 2, 3]  # Stages 0, 1, 2, 3 (0-indexed: stages[0], stages[1], stages[2], stages[3])
    
    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        expansion_ratio: int,
        use_grn: bool,
        downsample: bool,
    ) -> nn.Module:
        """
        Create a single encoder stage.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            num_blocks: Number of ConvNeXt blocks
            expansion_ratio: MLP expansion ratio
            use_grn: Whether to use GRN
            downsample: Whether to downsample at the start
            
        Returns:
            Stage module
        """
        layers = []
        
        # Downsampling layer (stride-2 conv) if needed
        if downsample:
            layers.append(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=2, padding=1
                )
            )
            # Note: LayerNorm will be applied in ConvNeXtBlock, not here
        
        # ConvNeXt blocks
        for _ in range(num_blocks):
            layers.append(
                ConvNeXtBlock(
                    dim=out_channels,
                    expansion_ratio=expansion_ratio,
                    use_grn=use_grn,
                )
            )
        
        return nn.Sequential(*layers)
    
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C_in, H, W)
            
        Returns:
            Tuple of:
            - feature_maps: List of multi-scale feature maps [C2, C3, C4, C5]
            - masks: List of padding masks for each feature map
        """
        # Patch stem
        x = self.stem(x)  # (B, base_width, H//4, W//4)
        
        # Process through stages
        features = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            # Store features from stages 0, 1, 2, 3 (C2, C3, C4, C5)
            if i in self.out_indices:
                features.append(x)
        
        # Generate padding masks for each feature map
        # Masks indicate valid (non-padded) regions: True = valid, False = padding
        masks = []
        for feat in features:
            B, C, H, W = feat.shape
            # For now, assume all regions are valid (no padding)
            # In practice, this should be computed from input padding
            mask = torch.ones(B, H, W, dtype=torch.bool, device=feat.device)
            masks.append(mask)
        
        return features, masks

