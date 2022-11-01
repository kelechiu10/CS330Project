import torch


class Tune:
    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer: torch.optim.Optimizer = optimizer

    def update(self):
        self.optimizer.step()
