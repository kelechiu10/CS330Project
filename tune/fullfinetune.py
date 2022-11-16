import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import autograd
from torch.utils import tensorboard

import models

layers = []
resnet = models.get_model('resnet18', num_classes=10)
for str, module in resnet.named_children():
    layers.append((str, module))

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
print(modified_layers)


#DEVICE = torch.device(args.device)

# class basic_fine_tuning_model:
#     def __init__(self, model, num_outputs, lr, learn_lrs, fine_tune_mode="all"):
#         self.model = model
#         self.lr = lr
#         self.fine_tune_mode = fine_tune_mode
#         self.learn_lrs = learn_lrs
#         self._meta_parameters = model._modules()
#         self._inner_lrs = {
#             k: torch.tensor(lr, requires_grad=learn_lrs)
#             for k in self._meta_parameters.keys()
#         }
#
#     def forward(self, images):
#         return self.model(images)
#
#
#     def apply_fine_tuning(self, images, labels):
#         if fine_tune_mode == "all":
#             whil
#         elif fine_tune_mode == "last":
#
#         elif fine_tune_mode == "first":
#
#         else
#             raise Exception("Fine-tuning mode must be set to 'all', 'last', or 'first'!")