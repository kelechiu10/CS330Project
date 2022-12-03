from torch import optim
import numpy as np
import torch
from torch import autograd

from optimizers import LayerWiseOptimizer


class MAMLOptimizer:
    def __init__(self, layers, lr, optimizer=optim.Adam):
        self.lr = lr
        self.layer_lrs = torch.tensor(np.ones(len(layers)), requires_grad=True)
        self.outer_optimizer = optimizer(self.layer_lrs, lr=1e-3)
        self.inner_optimizer = LayerWiseOptimizer(layers, create_graph=True)

    def step(self, loss, closure=None):
        # grads = autograd.grad(loss, self.parameters, create_graph=True)
        # for i, grad in enumerate(grads):
        #     self.parameters[i] = self.parameters[i] - self.layer_lrs[i] * self.lr * grad
        loss.backwards(create_graph=True)
        self.inner_optimizer.step(self.layer_lrs)
        self.outer_optimizer.step(closure=closure)

    def zero_grad(self, set_to_none=False):
        self.inner_optimizer.zero_grad(set_to_none=set_to_none)
        self.outer_optimizer.zero_grad(set_to_none=set_to_none)
