import torch.optim as optim
from typing import cast, List, Optional, Dict, Tuple
import torch

# denote blocks
# start epoch
# optimizer.reconfigure layer(k)
    # for layers in parameters
        # set lr 0
    # lrs[k] = 1e-3
# otpimzer.zero_grad
# train loop
# loss = criterion(logits, y)
# loss.backward()

# class CrossValidationOptimizer(optim.Adam):
#     def __init__(self, layered_model, lr=1e-3):
#         layer_names = [x[0] for x in layered_model.layered_modules]
#         #layer_index = layer_names.index(layer_name)
#         modules = layered_model.layered_modules[layer_index][1]
#
#         layer_params = []
#         for module in modules.modules():
#             for param in module.parameters():
#                 layer_params.append(param)
#         self.layer_parameters = layer_params
#         self.lr = lr
#
#         self.optimizer = optim.Adam(params=self.layer_parameters, lr=self.lr)

class CrossValidationOptimizer(optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, *, foreach: Optional[bool] = None,
                 maximize: bool = False, capturable: bool = False,
                 differentiable: bool = False, fused: bool = False):
        super(CrossValidationOptimizer, self).__init__(params, lr=lr, betas=betas, eps=eps,
                 weight_decay=weight_decay, amsgrad=amsgrad, foreach=foreach,
                 maximize=maximize, capturable=capturable,
                 differentiable=differentiable, fused=fused)

        self.original_params = params
        self.original_lrs = [param_group['lr'] for param_group in self.param_groups]
        for param_group in self.param_groups:
            param_group['lr'] = 0

    def use_layers(self, block_index):
        # reset previous layers learning rate and set specific block layers
        self.param_groups[block_index]['lr'] = self.original_lrs[block_index]

    def step(self, closure=None):

        for i in range(len(self.param_groups)):
            self.use_layers(i)

