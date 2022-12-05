from typing import List

import numpy as np
from torch import optim


class SingleLayerOptimizer:
    def __init__(self, layers, idx, writer, lr=1e-3):
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

        self.totals = np.zeros(len(layers))
        self.idx = idx
        self.writer = writer
        self.n = 0

    def step(self, loss, closure=None):
        loss.backward()
        self.totals[self.idx] += 1
        for i in range(len(self.totals)):
            self.writer.add_scalar(f'train/arm_{i}', self.totals[i], self.n)
        self.n += 1
        return self.optimizer.step(closure=closure)

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)
