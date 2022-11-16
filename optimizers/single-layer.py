import os

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch import autograd
from torch.utils import tensorboard

import omniglot
# from models import resnet
import models
import layered_model


class SingleLayerOptimizer:
    def __init__(self, layered_model, layer_name, lr=1e-3):
        layer_names = [x[0] for x in layered_model.layered_modules]
        layer_index = layer_names.index(layer_name)
        modules = layered_model.layered_modules[layer_index][1]

        layer_params = []
        for module in modules.modules():
            for param in module.parameters():
                layer_params.append(param)
        self.layer_parameters = layer_params
        self.lr = lr

        self.optimizer = optim.Adam(params=self.layer_parameters, lr=self.lr)

    def step(self, closure=None):
        return self.optimizer.step(closure=closure)

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)
