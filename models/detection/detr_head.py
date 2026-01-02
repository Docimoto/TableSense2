"""
Deformable DETR Detection Head.

Classification and bounding box regression heads for object detection.
"""

import torch
import torch.nn as nn
from typing import Tuple


class DeformableDETRHead(nn.Module):
    """
    Detection head for Deformable DETR.
    
    Predicts classification scores and bounding boxes from decoded query embeddings.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_classes: int = 2,  # table vs background
    ):
        """
        Initialize detection head.
        
        Args:
            d_model: Model dimension (default: 256)
            num_classes: Number of classes including background (default: 2)
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # Classification head
        self.class_embed = nn.Linear(d_model, num_classes)
        
        # Bounding box head (predicts normalized center coordinates and size)
        # Output: (cx, cy, w, h) in normalized coordinates [0, 1]
        self.bbox_embed = MLP(d_model, d_model, 4, 3)
    
    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            hidden_states: Decoded query embeddings of shape (B, num_queries, d_model)
            
        Returns:
            Tuple of:
            - classification_logits: (B, num_queries, num_classes)
            - bbox_preds: (B, num_queries, 4) in normalized coordinates (cx, cy, w, h)
        """
        # Classification
        outputs_class = self.class_embed(hidden_states)
        
        # Bounding box regression
        outputs_coord = self.bbox_embed(hidden_states).sigmoid()
        # Sigmoid ensures coordinates are in [0, 1]
        
        return outputs_class, outputs_coord


class MLP(nn.Module):
    """Multi-layer perceptron with ReLU activations."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        """
        Initialize MLP.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            num_layers: Number of layers
        """
        super().__init__()
        
        num_layers = max(1, num_layers)
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x

