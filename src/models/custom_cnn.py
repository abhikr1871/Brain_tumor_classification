"""
Custom CNN Architecture for Brain Tumor Classification.
Lightweight, efficient, and designed to run smoothly on edge GPUs (e.g., GTX 1650 4GB) or CPU.
"""

import torch
import torch.nn as nn
from typing import Tuple

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class BrainTumorCNN(nn.Module):
    """
    4-Stage Convolutional Neural Network with Batch Normalization and Dropout.
    Input shape: (B, 3, H, W) -> Output shape: (B, num_classes)
    """
    def __init__(self, num_classes: int = 4, in_channels: int = 3):
        super().__init__()
        
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32, dropout_rate=0.1),    # Output: 32 x H/2 x W/2
            ConvBlock(32, 64, dropout_rate=0.2),            # Output: 64 x H/4 x W/4
            ConvBlock(64, 128, dropout_rate=0.25),          # Output: 128 x H/8 x W/8
            ConvBlock(128, 256, dropout_rate=0.3)           # Output: 256 x H/16 x W/16
        )
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        logits = self.classifier(x)
        return logits
