import torch
import torch.nn.functional as F
import numpy as np
from torch import optim
from optimizers.layerwise import LayerWiseOptimizer


class GradNorm:
    def __init__(self, layers, lr, writer, optimizer=optim.Adam):
        self.lr = lr
        self.optimizer = LayerWiseOptimizer(layers, lr, optimizer=optimizer)
        self.totals = np.zeros(len(layers))
        self.writer = writer
        self.n = 0

    def get_grad_norms(self):
        param_groups = self.optimizer.param_groups
        grad_norms = torch.empty(len(param_groups))
        param_norms = torch.empty(len(param_groups))

        for i, group in enumerate(param_groups):
            grad_norms[i] = np.norm([torch.norm(p.grad) ** 2 for p in group['params']])
            param_norms[i] = np.norm([torch.norm(p) ** 2 for p in group['params']])

        grad_norms = F.softmax(grad_norms / param_norms, dim=0)
        return grad_norms

    def step(self, loss, closure=None):
        loss.backward()
        grad_norms = self.get_grad_norms()
        self.totals += grad_norms
        for i in range(len(self.totals)):
            self.writer.add_scalar(f'train/arm_{i}', self.totals[i], self.n)
        self.optimizer.step(grad_norms, closure=closure)
        self.n += 1

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)
