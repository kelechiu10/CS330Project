import torch
import torch.nn.functional as F
from tune import Tune
import numpy as np


def sqnorm(x):
    return np.inner(x, x)


class GradNorm(Tune):
    def __init__(self, optimizer: torch.optim.Optimizer, lr):
        super().__init__(optimizer)
        self.lr = lr

    def update(self):
        # TODO: possibly change to relative gradient norm (divide layer grads' norm by layer parameters' norm)
        param_groups = self.optimizer.param_groups
        grad_norms = torch.empty(len(param_groups))

        for i, group in enumerate(param_groups):
            grad_norm = 0
            for p in group['params']:
                grad_norm += sqnorm(p.grad)  # grad does not have gradients so we can use numpy

            grad_norms[i] = np.sqrt(grad_norm)

        grad_norms = F.softmax(grad_norms, dim=0)

        for i, group in enumerate(param_groups):
            group['lr'] = grad_norms[i] * self.lr

        self.optimizer.step()
