import torch
import torch.nn as nn
import torchvision.models as models


def get_resnet18(pretrained):
    weights = models.ResNet18_Weights.DEFAULT  # IMAGENET1K_V1
    resnet = models.resnet18(weights=weights)
    return resnet

