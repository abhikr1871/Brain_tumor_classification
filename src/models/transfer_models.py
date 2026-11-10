"""
Transfer Learning Model Architectures for Brain Tumor Classification.
Wraps torchvision and timm models (ResNet, EfficientNet, DenseNet) with custom classification heads.
"""

import torch
import torch.nn as nn
from torchvision import models

def build_transfer_model(
    model_name: str = "resnet50",
    num_classes: int = 4,
    pretrained: bool = True,
    freeze_backbone: bool = False
) -> nn.Module:
    """
    Factory function for Transfer Learning architectures.
    
    Supported models:
    - 'resnet18', 'resnet50'
    - 'efficientnet_b0', 'efficientnet_b2'
    - 'densenet121'
    - 'mobilenet_v3_small'
    """
    model_name = model_name.lower()
    
    if "resnet50" in model_name:
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        )
        target_layer = model.layer4[-1]

    elif "resnet18" in model_name:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
        target_layer = model.layer4[-1]

    elif "efficientnet_b0" in model_name:
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
        target_layer = model.features[-1]

    elif "densenet121" in model_name:
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
        target_layer = model.features.denseblock4

    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "fc" not in name and "classifier" not in name:
                param.requires_grad = False

    # Attach metadata attribute for Grad-CAM convenience
    model.target_gradcam_layer = target_layer
    return model
