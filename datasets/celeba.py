import os
import tarfile

import requests as requests
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from datasets.util import random_split
from torch.utils.data import DataLoader

# Root directory for the dataset
data_root = './datasets/data/'
# data_root = os.path.abspath('./datasets/data')
# Spatial size of training images, images are resized to this size.
image_size = 64

transform = transform=transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                          std=[0.5, 0.5, 0.5])
])


def get_dataloaders(cfg, shuffle=True, split=(0.75, 0.25)):
    num_workers = cfg.train.num_workers
    batch_size = cfg.train.batch_size


    #dataset = datasets.ImageFolder(data_root, transform)
    train_dataset = datasets.CelebA(root=data_root,
                    download=False, split="train",
                    target_type='attr',
                    transform=transforms.Compose([
                        transforms.Resize(image_size),
                        transforms.CenterCrop(image_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                             std=[0.5, 0.5, 0.5])
                    ]))
    test_dataset = datasets.CelebA(root=data_root,
                                    download=False, split="valid",
                                    target_type='attr',
                                    transform=transforms.Compose([
                                        transforms.Resize(image_size),
                                        transforms.CenterCrop(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                             std=[0.5, 0.5, 0.5])
                                    ]))
    #train_dataset, test_dataset = random_split(dataset, split)

    dataloaders = dict()
    dataloaders['train'] = DataLoader(
        train_dataset, batch_size=batch_size,
        num_workers=num_workers, shuffle=shuffle
    )
    dataloaders['eval'] = DataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=num_workers, shuffle=shuffle
    )
    return dataloaders



