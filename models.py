import torch
import torch.nn as nn
import torchvision.models as models

def initialize_model(model_name, num_classes, pretrained=True):
    model = None
    if model_name == 'resnet18':
        weights = models.ResNet18_Weights.DEFAULT  # IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    elif model_name == 'resnet34':
        weights = models.ResNet32_Weights.DEFAULT
        model = models.resnet34(weights=weights)
    elif model_name == 'resnet50':
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    else:
        raise NotImplementedError()
    return model