from SMPyBandits.Policies import BasePolicy, EpsilonGreedy, with_proba
from torch import optim
from optimizers.layerwise import LayerWiseOptimizer
import numpy as np
import numpy.random as rn
import torch


EPSILON = 0.1
class EpsilonGreedyFixed(EpsilonGreedy):
    def __init__(self, nbArms, epsilon=EPSILON, lower=0., amplitude=1.):
        super(EpsilonGreedyFixed, self).__init__(nbArms, lower=lower, amplitude=amplitude)

    def choice(self):
        """With a probability of epsilon, explore (uniform choice), otherwhise exploit based on just accumulated *rewards* (not empirical mean rewards)."""
        if with_proba(self.epsilon):  # Proba epsilon : explore
            return rn.randint(0, self.nbArms)
        else:  # Proba 1 - epsilon : exploit
            # Uniform choice among the best arms
            biased_means = self.rewards / (1 + self.pulls)
            return rn.choice(np.nonzero(biased_means == np.max(biased_means))[0])


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
            self.mab_policy.getReward(self.last_arm, self.reward_metric(loss.item()))#self.last_loss / loss)
            print(self.reward_metric(loss.item()))
        self.last_loss = loss.item()
        arm = self.mab_policy.choice()
        #print(arm)
        self.last_arm = arm

        loss.backward()
        self.optimizer.step(arm, closure=closure)
        print(self.mab_policy.rewards / (1 + self.mab_policy.pulls))
        print(self.mab_policy.pulls)

    def reward_metric(self, loss):
        goodness = (self.last_loss - loss) / self.last_loss
        # return 1. / (1 + torch.exp(-goodness))
        return goodness

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)
