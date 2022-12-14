import os
import tarfile

import requests as requests
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import random
import numpy as np

NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                      'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                      'snow', 'frost', 'fog', 'brightness',
                      'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
te_transforms = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(*NORM)])
tr_transforms = transforms.Compose([#transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(*NORM)])


def get_dataloaders(cfg, corrupted=True):
    base_dataset = datasets.CIFAR10(
        root=cfg.datasets.dir, train=True, transform=tr_transforms, download=True
    )

    if corrupted:
        if not os.path.exists(os.path.join(cfg.datasets.dir, 'CIFAR-10-C')):
            print('Begin downloading CIFAR10-C dataset')
            if not os.path.exists(cfg.datasets.dir):
                os.mkdir(cfg.datasets.dir)
            url = 'https://zenodo.org/record/2535967/files/CIFAR-10-C.tar'
            with requests.get(url, stream=True) as rx, tarfile.open(fileobj=rx.raw, mode="r|") as tarobj:
                tarobj.extractall(path=cfg.cifar.dir)
            print('Download successful!')
        # SIZE = 2000
        # rng = np.random.default_rng(0)
        # perm = rng.permutation(len(base_dataset))[:SIZE]
        # base_dataset.data = np.load(cfg.datasets.dir + f'/CIFAR-10-C/{cfg.datasets.cifar.corruption}.npy')[perm]
        # base_dataset.targets = np.load(cfg.datasets.dir + f'/CIFAR-10-C/labels.npy')[perm]
        SAMPLE_SIZE = 133
        data_list = []
        targets_list = []
        for corruption in common_corruptions:
            rng = np.random.default_rng(0)
            perm = rng.permutation(len(base_dataset))[:SAMPLE_SIZE]
            data = np.load(cfg.datasets.dir + f'/CIFAR-10-C/{corruption}.npy')[perm]
            targets = np.load(cfg.datasets.dir + f'/CIFAR-10-C/labels.npy')[perm]
            data_list.extend(data)
            targets_list.extend(targets)
        shuffled_inds = np.arange(len(data_list))
        np.random.permutation(shuffled_inds)
        shuffled_inds = shuffled_inds.tolist()
        data_list = np.asarray(data_list)[shuffled_inds]
        targets_list = np.asarray(targets_list)[shuffled_inds]

        base_dataset.data = data_list
        base_dataset.targets = targets_list
    return base_dataset
