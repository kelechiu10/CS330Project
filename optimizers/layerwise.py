from torch import optim
import numpy as np


class LayerWiseOptimizer:
    def __init__(self, layers, lr=1e-3, optimizer=optim.Adam, **kwargs):
        self.lr = lr
        self.n = len(layers)
        print(self.n)
        param_groups = [{"params": layer.parameters(), "lr": lr} for layer in layers]
        print(kwargs)
        self.optimizer = optimizer(params=param_groups, lr=self.lr, **kwargs)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self, weights, closure=None):
        if isinstance(weights, (int, np.integer)):
            idx = weights
            weights = np.zeros(self.n)
            weights[idx] = 1
            #print(len(weights))
            #print(weights)

        assert len(weights) == self.n, f'Number of weights does not match number of layers {self.n}.'

        param_groups = self.optimizer.param_groups
        for i, group in enumerate(param_groups):
            group['lr'] = weights[i] * self.lr
            print(group['lr'])
            print(weights[i])
        self.optimizer.step(closure=closure)

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)
