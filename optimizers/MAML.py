from torch import optim
import numpy as np
import torch


class MAMLOptimizer:
    def __init__(self, layers, lr, optimizer=optim.Adam):
        self.lr = lr
        self.optimizer = optimizer

    def step(self, closure=None):
        self.optimizer.step(closure=closure)

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)
