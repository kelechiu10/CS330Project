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

# need args to have num_workers and batch_size even as defaults, as well as a dataroot
def get_dataloaders(args, shuffle=True):
    num_workers = args.num_workers
    batch_size = args.batch_size

    train_dataset = datasets.CIFAR10(
        root=args.dataroot, train=True,
        download=True, transform=tr_transforms,
    )
    val_dataset = datasets.CIFAR10(
        root=args.dataroot, train=True,
        download=True, transform=te_transforms,
    )

    train_dataset.data = np.load(args.dataroot + f'/CIFAR-10-C/{args.corruption}.npy')
    val_dataset.data = np.load(args.dataroot + f'/CIFAR-10-C/{args.corruption}.npy')


    dataloaders = dict()
    dataloaders['train'] = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        num_workers=num_workers, shuffle=shuffle
    )
    dataloaders['eval'] = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size,
        num_workers=num_workers, shuffle=shuffle
    )

    return dataloaders
