from typing import List
from torch import optim


class SingleLayerOptimizer:
    def __init__(self, layers, idx, lr=1e-3):
        layer = layers[idx]

        layer_params = []
        if isinstance(layer, List):
            for l in layer:
                layer_params.append(l.parameters())
        else:
            layer_params = layer.parameters()

        self.layer_parameters = layer_params
        self.lr = lr

        self.optimizer = optim.Adam(params=self.layer_parameters, lr=self.lr)

    def step(self, closure=None):
        return self.optimizer.step(closure=closure)

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)
