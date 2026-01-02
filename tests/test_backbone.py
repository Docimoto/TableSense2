"""
Tests for ConvNeXt V2 backbone components.
"""

import torch
import pytest
from models.backbone.blocks import ConvNeXtBlock
from models.backbone.stem import PatchStem
from models.backbone.encoder import ConvNeXtV2Encoder
from models.backbone import TableBackbone


def test_convnext_block():
    """Test ConvNeXt block forward pass."""
    block = ConvNeXtBlock(dim=64, expansion_ratio=4)
    x = torch.randn(2, 64, 32, 32)
    out = block(x)
    assert out.shape == (2, 64, 32, 32)


def test_patch_stem():
    """Test patch stem forward pass."""
    stem = PatchStem(in_channels=43, out_channels=64)
    x = torch.randn(2, 43, 100, 100)
    out = stem(x)
    assert out.shape == (2, 64, 25, 25)  # 100/4 = 25


def test_convnext_encoder():
    """Test ConvNeXt encoder forward pass."""
    encoder = ConvNeXtV2Encoder(
        in_channels=43,
        base_width=32,  # Minimal config
        depths=[2, 2, 4, 2],  # Minimal config
    )
    x = torch.randn(1, 43, 100, 100)
    features, masks = encoder(x)
    
    # Should output 4 feature maps (C2, C3, C4, C5)
    assert len(features) == 4
    assert len(masks) == 4
    
    # Check shapes (approximate, depends on exact downsampling)
    # After stem: 100/4 = 25
    # Stage 0: 25 (no downsampling)
    # Stage 1: 25/2 = 12 (C3)
    # Stage 2: 12/2 = 6 (C4)
    # Stage 3: 6/2 = 3 (C5)
    assert features[0].shape[2] > 0  # C2 height
    assert features[1].shape[2] > 0  # C3 height
    assert features[2].shape[2] > 0  # C4 height
    assert features[3].shape[2] > 0  # C5 height


def test_table_backbone():
    """Test TableBackbone wrapper."""
    backbone = TableBackbone(
        in_channels=43,
        base_width=32,  # Minimal config
        depths=[2, 2, 4, 2],  # Minimal config
    )
    x = torch.randn(1, 43, 100, 100)
    features, masks = backbone(x)
    
    assert len(features) == 4
    assert len(masks) == 4
    assert len(backbone.strides) == 4
    for feat, mask in zip(features, masks):
        assert mask.shape[1:] == feat.shape[2:]


def test_table_backbone_with_mask():
    """Test TableBackbone with input mask."""
    backbone = TableBackbone(
        in_channels=43,
        base_width=32,
        depths=[2, 2, 4, 2],
    )
    x = torch.randn(1, 43, 100, 100)
    mask = torch.ones(1, 100, 100, dtype=torch.bool)
    features, masks = backbone(x, mask=mask)
    
    assert len(features) == 4
    assert len(masks) == 4
    # Masks should be propagated and match feature shapes
    for feat, m in zip(features, masks):
        assert m.shape[0] == 1  # Batch size
        assert m.shape[1:] == feat.shape[2:]

