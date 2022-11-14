import torch.optim as optim
from typing import cast, List, Optional, Dict, Tuple
import torch

# denote blocks
# start epoch
# optimizer.reconfigure layer(k)
    # for layers in parameters
        # set lr 0
    # lrs[k] = 1e-3
# otpimzer.zero_grad
# train loop
# loss = criterion(logits, y)
# loss.backward()

class CrossValidationOptimizer(optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, *, foreach: Optional[bool] = None,
                 maximize: bool = False, capturable: bool = False,
                 differentiable: bool = False, fused: bool = False):
        super(CrossValidationOptimizer, self).__init__(params, lr=lr, betas=betas, eps=eps,
                 weight_decay=weight_decay, amsgrad=amsgrad, foreach=foreach,
                 maximize=maximize, capturable=capturable,
                 differentiable=differentiable, fused=fused)

    def use_layers(block_index):
        # reset previous layers learning rate and set specific block layers
        pass
