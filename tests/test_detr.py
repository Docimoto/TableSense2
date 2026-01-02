"""
Tests for Deformable DETR components.
"""

import torch
import pytest
from models.detection.deformable_attention import MultiScaleDeformableAttention
from models.detection.positional_encoding import PositionalEncoding2D
from models.detection.detr_encoder import DeformableDETREncoder
from models.detection.detr_decoder import DeformableDETRDecoder
from models.detection.detr_head import DeformableDETRHead


def test_deformable_attention():
    """Test multi-scale deformable attention."""
    attn = MultiScaleDeformableAttention(
        d_model=128,  # Minimal config
        n_levels=3,
        n_heads=4,
        n_points=2,
    )
    
    B, num_queries = 2, 10
    spatial_shapes = [(10, 10), (5, 5), (3, 3)]
    num_keys = sum(H * W for H, W in spatial_shapes)  # 100 + 25 + 9 = 134
    
    query = torch.randn(B, num_queries, 128)
    reference_points = torch.rand(2, num_queries, 3, 2)  # (B, num_queries, n_levels, 2)
    input_flatten = torch.randn(B, num_keys, 128)
    
    out = attn(
        query=query,
        reference_points=reference_points,
        input_flatten=input_flatten,
        input_spatial_shapes=spatial_shapes,
    )
    
    assert out.shape == (B, num_queries, 128)


def test_positional_encoding():
    """Test 2D positional encoding."""
    pos_enc = PositionalEncoding2D(d_model=128)
    spatial_shapes = [(10, 10), (5, 5), (3, 3)]
    encodings = pos_enc(spatial_shapes)
    
    assert len(encodings) == 3
    assert encodings[0].shape == (100, 128)  # 10*10 = 100
    assert encodings[1].shape == (25, 128)   # 5*5 = 25
    assert encodings[2].shape == (9, 128)    # 3*3 = 9


def test_detr_encoder():
    """Test Deformable DETR encoder."""
    encoder = DeformableDETREncoder(
        d_model=128,  # Minimal config
        n_layers=2,   # Minimal config
        n_levels=3,
        n_heads=4,
        n_points=2,
    )
    
    B = 1
    srcs = [
        torch.randn(B, 128, 10, 10),
        torch.randn(B, 128, 5, 5),
        torch.randn(B, 128, 3, 3),
    ]
    
    pos_enc = PositionalEncoding2D(d_model=128)
    spatial_shapes = [(10, 10), (5, 5), (3, 3)]
    pos_embeds = pos_enc(spatial_shapes)
    
    # Create reference points
    num_keys = sum(H * W for H, W in spatial_shapes)
    reference_points = torch.rand(B, num_keys, 3, 2)
    
    memory = encoder(
        srcs=srcs,
        pos_embeds=pos_embeds,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
    )
    
    assert memory.shape == (B, num_keys, 128)


def test_detr_decoder():
    """Test Deformable DETR decoder."""
    decoder = DeformableDETRDecoder(
        d_model=128,  # Minimal config
        n_layers=2,   # Minimal config
        num_queries=10,  # Minimal config
        n_levels=3,
        n_heads=4,
        n_points=2,
    )
    
    B = 1
    spatial_shapes = [(10, 10), (5, 5), (3, 3)]
    num_keys = sum(H * W for H, W in spatial_shapes)  # 100 + 25 + 9 = 134
    memory = torch.randn(B, num_keys, 128)
    
    hidden_states, reference_points = decoder(
        memory=memory,
        spatial_shapes=spatial_shapes,
    )
    
    assert hidden_states.shape == (B, 10, 128)
    assert reference_points.shape == (B, 10, 3, 2)


def test_detr_head():
    """Test detection head."""
    head = DeformableDETRHead(
        d_model=128,  # Minimal config
        num_classes=2,
    )
    
    B, num_queries = 2, 10
    hidden_states = torch.randn(B, num_queries, 128)
    
    logits, boxes = head(hidden_states)
    
    assert logits.shape == (B, num_queries, 2)
    assert boxes.shape == (B, num_queries, 4)
    # Boxes should be in [0, 1] (sigmoid applied)
    assert boxes.min() >= 0.0
    assert boxes.max() <= 1.0

