import torch
import torch.nn.functional as F
import numpy as np
from torch import optim
from optimizers.layerwise import LayerWiseOptimizer


class GradNorm:
    def __init__(self, layers, lr, optimizer=optim.Adam):
        self.lr = lr
        self.optimizer = LayerWiseOptimizer(layers, lr, optimizer=optimizer)

    def get_grad_norms(self):
        # TODO: possibly change to relative gradient norm (divide layer grads' norm by layer parameters' norm)
        param_groups = self.optimizer.param_groups
        grad_norms = torch.empty(len(param_groups))

        for i, group in enumerate(param_groups):
            grad_norms[i] = np.norm([torch.norm(p.grad) ** 2 for p in group['params']])

        grad_norms = F.softmax(grad_norms, dim=0)
        return grad_norms

    def step(self, closure=None):
        grad_norms = self.get_grad_norms()
        self.optimizer.step(grad_norms, closure=closure)

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)
