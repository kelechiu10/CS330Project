import torch
import torch.nn as nn
import torchvision.models as tmodels
import pretrained_resnet_cifar as pretrained_resnets


def get_imagenet_model(model_name, pretrained=True):
    model = None
    if model_name == 'resnet18':
        weights = tmodels.ResNet18_Weights.DEFAULT  # IMAGENET1K_V1
        model = tmodels.resnet18(weights=weights)
    elif model_name == 'resnet34':
        weights = tmodels.ResNet32_Weights.DEFAULT
        model = tmodels.resnet34(weights=weights)
    elif model_name == 'resnet50':
        weights = tmodels.ResNet50_Weights.DEFAULT
        model = tmodels.resnet50(weights=weights)
    else:
        raise NotImplementedError()
    return model


def get_cifar_model(model_name, pretrained_dir=None):
    model = None
    if model_name == 'resnet18':
        model = pretrained_resnets.resnet18(pretrained_dir=pretrained_dir)  # pretrained on CIFAR-10
    elif model_name == 'resnet34':
        model = pretrained_resnets.resnet34(pretrained_dir=pretrained_dir)
    elif model_name == 'resnet50':
        model = pretrained_resnets.resnet50(pretrained_dir=pretrained_dir)
    else:
        raise NotImplementedError()
    return model

def get_pretrained_resnet50(pretrained_dir=None):
    model = tmodels.resnet50(weights=tmodels.ResNet50_Weights.DEFAULT)
    layers = [model.conv1, model.layer1, model.layer2, model.layer3, model.layer4, model.fc]
    return model, layers


# res, layers = get_pretrained_resnet50()
# print(layers)
