import torch.nn as nn
from torchvision import models


def build_model(model_name: str, num_classes: int):
    model_name = model_name.lower()

    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        target_layer = model.layer4[-1]  # good Grad-CAM target
        return model, target_layer

    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        target_layer = model.features[-1]  # last conv-ish block
        return model, target_layer

    raise ValueError(f"Unknown model_name: {model_name}")
