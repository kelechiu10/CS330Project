from SMPyBandits.Policies import BasePolicy
from torch import optim
from optimizers.layerwise import LayerWiseOptimizer
import numpy as np
import torch


class MABOptimizer:
    def __init__(self, layers, lr, mab_policy: BasePolicy, optimizer=optim.Adam):
        assert mab_policy.nbArms == len(layers), f'Number of arms not equal to number of layers {len(layers)}.'

        self.lr = lr
        self.optimizer = LayerWiseOptimizer(layers, lr, optimizer=optimizer)
        self.mab_policy = mab_policy
        self.mab_policy.startGame()
        self.last_arm = None
        self.last_loss = None

    def step(self, loss, closure=None):
        if self.last_loss is not None:
            self.mab_policy.getReward(self.last_arm, self.reward_metric(loss))#self.last_loss / loss)
        self.last_loss = loss

        arm = self.mab_policy.choice()
        self.last_arm = arm
        self.optimizer.step(arm, closure=closure)

    def reward_metric(self, loss):
        goodness = (loss - self.last_loss) / loss
        return 1. / (1 + torch.exp(-goodness))
    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)
