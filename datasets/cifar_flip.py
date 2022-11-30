import os
import tarfile

import requests as requests
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from datasets.util import random_split

NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
te_transforms = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(*NORM)])
tr_transforms = transforms.Compose([#transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(*NORM)])

def get_dataloaders(cfg, shuffle=True, split=(0.75, 0.25)):
    num_workers = cfg.train.num_workers
    batch_size = cfg.train.batch_size

    dataset = datasets.CIFAR10(
        root=cfg.datasets.cifar.dir, train=True, transform=tr_transforms, download=True
    )
    flipped_targets = 9 - np.asarray(dataset.targets)
    dataset.targets = flipped_targets.tolist()
    train_dataset, test_dataset = random_split(dataset, split)

    dataloaders = dict()
    dataloaders['train'] = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        num_workers=num_workers, shuffle=shuffle
    )
    dataloaders['eval'] = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=num_workers, shuffle=shuffle
    )

    return dataloaders