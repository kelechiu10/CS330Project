import numpy as np
from torch import optim


class LayerWiseOptimizer:
    def __init__(self, layers, lr=1e-3, optimizer=optim.Adam):
        self.lr = lr
        self.n = len(layers)
        param_groups = [{"params": layer.parameters(), "lr": lr} for layer in layers]
        self.optimizer = optimizer(params=param_groups, lr=self.lr)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self, learning_rates, closure=None):
        assert len(learning_rates) == self.n

        param_groups = self.optimizer.param_groups
        for i, group in enumerate(param_groups):
            group['lr'] = learning_rates[i] * self.lr

        self.optimizer.step(closure=closure)

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)
