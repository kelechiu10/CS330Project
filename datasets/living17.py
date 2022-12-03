import os
from robustness import datasets
from robustness.tools.breeds_helpers import make_living17
from robustness.tools.breeds_helpers import setup_breeds

# returns both the source(pretrain) and target(finetuning)
def get_dataloaders(cfg, source=True):
    if not hasattr(cfg.datasets.living17, 'info_dir'):
        cfg.datasets.living17.info_dir = "./datasets/data/imagenet_class_hierarchy/modified"
    if not (os.path.exists(cfg.datasets.living17.info_dir) and len(os.listdir(cfg.info_dir))):
        print("Downloading class hierarchy information into `info_dir`")
        setup_breeds(cfg.datasets.living17.info_dir)

    ret = make_living17(cfg.datasets.living17.info_dir, split="rand")
    superclasses, subclass_split, label_map = ret
    train_subclasses, test_subclasses = subclass_split
    dataloaders = dict()

    if source == True:
        dataset_source = datasets.CustomImageNet(cfg.datasets.living17.data_dir, train_subclasses)
        loaders_source = dataset_source.make_loaders(cfg.num_workers, cfg.batch_size)
        train_loader_source, val_loader_source = loaders_source
        dataloaders = {'train': train_loader_source, 'eval':val_loader_source}
    else:
        dataset_target = datasets.CustomImageNet(cfg.datasets.living17.data_dir, test_subclasses)
        loaders_target = dataset_target.make_loaders(cfg.train.num_workers, cfg.train.batch_size)
        train_loader_target, val_loader_target = loaders_target
        dataloaders = {'train': train_loader_target, 'eval':val_loader_target}
    return dataloaders