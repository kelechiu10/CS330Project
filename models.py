import torch
import torch.nn as nn
import torchvision.models as models
from models import pretrained_resnet_cifar as pretrained_resnets

def get_imagenet_model(model_name, pretrained=True):
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

def get_cifar_model(model_name, pretrained_dir=None):
    model = None
    if model_name == 'resnet18':
        model = pretrained_resnets.resnet18(pretrained=pretrained_dir)  # pretrained on CIFAR-10
    elif model_name == 'resnet34':
        model = pretrained_resnets.resnet34(pretrained=pretrained_dir)
    elif model_name == 'resnet50':
        model = pretrained_resnets.resnet50(pretrained=pretrained_dir)
    else:
        raise NotImplementedError()
    return model
