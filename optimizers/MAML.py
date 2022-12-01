from torch import optim
import numpy as np
import torch


class MAMLOptimizer:
    def __init__(self, layers, lr, optimizer=optim.Adam):
        self.lr = lr
        self.optimizer = optimizer

    def step(self, closure=None):
        grads = autograd.grad(loss, parameters.values(), create_graph=train)
        for (name, grad) in zip(parameters.keys(), grads):
            parameters[name] = parameters[name] - self._inner_lrs[name] * grad

        self.optimizer.step(closure=closure)

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)
