import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import autograd
from torch.utils import tensorboard

import omniglot
import models

class layered_model(nn.Module):
    def __init__(self, model):
        super().__init()
        self.model = model

        layers = []
        for str, module in model.named_children():
            layers.add((str, module))
        first_module = []
        for i, layer in enumerate(layers):
            j = i
            if layer[0] != "layer1":
                first_module.append(layer[1])
            else:
                break
        first_module = nn.Sequential(*first_module)
        first_layer = ("layer0", first_module)

        modified_layers = [first_layer]
        modified_layers.extend(layers[j:])
        #print(modified_layers)
        self.layered_modules = modified_layers

    def forward(self, data):
        return self.model(data)