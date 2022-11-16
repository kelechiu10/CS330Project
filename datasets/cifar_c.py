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
def get_dataloaders(args, shuffle=False, val_split=0.25):
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
    num_train = len(train_dataset)
    train_dataset.data = np.load(args.dataroot + f'/CIFAR-10-C/{args.corruption}.npy')
    val_dataset.data = np.load(args.dataroot + f'/CIFAR-10-C/{args.corruption}.npy')

    # indices = list(range(num_train))
    # split = int(np.floor(val_split * num_train))
    #
    # if shuffle:
    #     np.random.seed(0) # can change later
    #     np.random.shuffle(indices)

    #train_idx, valid_idx = indices[split:], indices[:split]
    #train_sampler = SubsetRandomSampler(train_idx)
    #val_sampler = SubsetRandomSampler(valid_idx)

    dataloaders = dict()
    dataloaders['train'] = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        num_workers=num_workers, shuffle=True
    )
    dataloaders['eval'] = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size,
        num_workers=num_workers, shuffle=True
    )

    return dataloaders
    # train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    # datasets = {}
    # datasets['train'] = Subset(dataset, train_idx)
    # datasets['eval'] = Subset(dataset, val_idx)


# def _prepare_data(args, use_transforms=True):
#     if args.corruption in common_corruptions:
#         te_transforms_local = te_transforms_inc if use_transforms else None
#         print(f'Test on {args.corruption} level {args.level}')
#         validdir = os.path.join(args.dataroot, 'imagenet-c', args.corruption, str(args.level))
#         teset = datasets.ImageFolder(validdir, te_transforms_local)
#     if not hasattr(args, 'workers'):
#         args.workers = 8
#     collate_fn = None if use_transforms else lambda x: x
#     teloader = torch.utils.data.DataLoader(teset, batch_size=args.batch_size, shuffle=False,
#                                            num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)
#     return teloader
