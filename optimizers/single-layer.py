import os

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch import autograd
from torch.utils import tensorboard

import omniglot
#from models import resnet
import models
import layered_model

class single_layer_optimizer(optim.Optimizer):
    def __init__(self, layered_model, layer_name, lr=1e-3):
        super(single_layer_optimizer, self).__init__()
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

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()




