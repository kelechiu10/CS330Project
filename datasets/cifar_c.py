import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                      'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                      'snow', 'frost', 'fog', 'brightness',
                      'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
te_transforms = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(*NORM)])
tr_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(*NORM)])

def get_dataloaders(cfg, shuffle=True, split=(0.75, 0.25)):
    num_workers = cfg.train.num_workers
    batch_size = cfg.train.batch_size

    base_dataset = datasets.CIFAR10(
        root=cfg.datasets.cifar.dir, train=True,
        download=True, transform=tr_transforms,
    )

    base_dataset.data = np.load(cfg.datasets.cifar.dir + f'/CIFAR-10-C/{cfg.corruption}.npy')
    train_dataset, test_dataset = torch.utils.data.random_split(base_dataset, split)

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
