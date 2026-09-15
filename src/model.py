"""
Modular Neural Network Model Factory for Federated Learning.
Supports multiple computer vision architectures with dynamic N-class classification heads
and automated layer hooks for Explainable AI (Grad-CAM).
"""

from collections import OrderedDict
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import (
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    resnet18, ResNet18_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
)

SUPPORTED_ARCHITECTURES = {
    "mobilenet_v3_small": {
        "name": "MobileNetV3-Small",
        "description": "Ultra-lightweight depthwise separable CNN for resource-constrained edge CPUs",
        "category": "Edge / IoT",
        "typical_size_mb": 5.8
    },
    "resnet18": {
        "name": "ResNet-18",
        "description": "Standard residual network benchmark with deep skip connections",
        "category": "General Vision",
        "typical_size_mb": 44.7
    },
    "efficientnet_b0": {
        "name": "EfficientNet-B0",
        "description": "Compound-scaled architecture balancing accuracy, FLOPs, and parameter efficiency",
        "category": "High Efficiency",
        "typical_size_mb": 20.4
    }
}


def get_model(
    architecture: str = "mobilenet_v3_small",
    num_classes: int = 2,
    pretrained: bool = True
) -> nn.Module:
    """
    Instantiates a vision model with a customized classification head for N classes.
    
    Args:
        architecture: 'mobilenet_v3_small', 'resnet18', or 'efficientnet_b0'
        num_classes: Number of output target classes (2, 3, 5, etc.)
        pretrained: If True, initializes with ImageNet-1k pretrained weights
    """
    arch = architecture.lower().strip()

    if arch == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )
    elif arch == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )
    elif arch == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )
    else:
        raise ValueError(
            f"Unsupported architecture: '{architecture}'. "
            f"Choose from: {list(SUPPORTED_ARCHITECTURES.keys())}"
        )

    model.architecture_name = arch
    return model


def get_target_gradcam_layer(model: nn.Module, architecture: str = "mobilenet_v3_small") -> nn.Module:
    """Returns the optimal final convolutional layer for Grad-CAM feature attribution."""
    arch = architecture.lower().strip()
    if arch == "mobilenet_v3_small":
        return model.features[-1]
    elif arch == "resnet18":
        return model.layer4[-1]
    elif arch == "efficientnet_b0":
        return model.features[-1]
    else:
        # Fallback to last conv module found
        for module in reversed(list(model.modules())):
            if isinstance(module, nn.Conv2d):
                return module
        return model


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    """Extract model parameters as a list of NumPy arrays for Flower serialization."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """Load model parameters from a list of NumPy arrays."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module, precision_bytes: int = 4) -> float:
    """
    Calculate the serialized weight payload size in Megabytes (MB).
    By default FP32 (4 bytes per parameter).
    """
    total_params = sum(p.numel() for p in model.parameters())
    return (total_params * precision_bytes) / (1024 * 1024)


if __name__ == "__main__":
    print("=== Model Factory Architecture Registry ===")
    for arch_key in SUPPORTED_ARCHITECTURES:
        net = get_model(arch_key, num_classes=3, pretrained=False)
        params = count_parameters(net)
        size_mb = get_model_size_mb(net)
        cam_layer = get_target_gradcam_layer(net, arch_key)
        print(f"[{arch_key}] {params:,} parameters | Payload: {size_mb:.2f} MB | Grad-CAM Hook: {type(cam_layer).__name__}")
