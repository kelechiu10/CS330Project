# attempt at creating a train/test for cifar that has a distribution shift
import torchvision
import os
from torch.utils.data import ConcatDataset, Subset
import pickle
import re
import numpy as np
from sklearn.model_selection import train_test_split

sub_to_super = dict()
label_map = dict()
distributions = []
# from https://github.com/ryanchankh/cifar100coarse/blob/master/sparse2coarse.py
def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.
    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    return coarse_labels[targets]





def _load_meta() -> None:
        path = os.path.join("datasets/data", "cifar-100-python/meta")
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            classes = data["fine_label_names"]
        class_to_idx = {_class: i for i, _class in enumerate(classes)}
        return class_to_idx



all_labels = """aquatic mammals@beaver, dolphin, otter, seal, whale
fish@aquarium fish, flatfish, ray, shark, trout
flowers@orchid, poppy, rose, sunflower, tulip
food containers@bottle, bowl, can, cup, plate
fruit and vegetables@apple, mushroom, orange, pear, sweet pepper
household electrical devices@clock, keyboard, lamp, telephone, television
household furniture@bed, chair, couch, table, wardrobe
insects@bee, beetle, butterfly, caterpillar, cockroach
large carnivores@bear, leopard, lion, tiger, wolf
large man-made outdoor things@bridge, castle, house, road, skyscraper
large natural outdoor scenes@cloud, forest, mountain, plain, sea
large omnivores and herbivores@camel, cattle, chimpanzee, elephant, kangaroo
medium-sized mammals@fox, porcupine, possum, raccoon, skunk
non-insect invertebrates@crab, lobster, snail, spider, worm
people@baby, boy, girl, man, woman
reptiles@crocodile, dinosaur, lizard, snake, turtle
small mammals@hamster, mouse, rabbit, shrew, squirrel
trees@maple_tree, oak_tree, palm_tree, pine_tree, willow_tree
vehicles 1@bicycle, bus, motorcycle, pickup truck, train
vehicles 2@lawn_mower, rocket, streetcar, tank, tractor"""

def _populate():
    class_to_idx = _load_meta()
    # print(all_labels)
    probs = [0.35, 0.35, 0.2, 0.05, 0.05]
    counts = [3000] * 5 # for a single superclass
    all_counts = [int(p * c) for p, c in zip(probs, counts)]

    main_labels = all_labels.split('\n')
    # print(class_to_idx)

    for i in range(len(main_labels)):
        cur_labels = main_labels[i].split('@')
        min_labels = cur_labels[1].split(', ')
        big_label = cur_labels[0]
        for j in range(len(min_labels)):
            sub_to_super[min_labels[j]] = big_label
            cur = class_to_idx[min_labels[j].replace(' ', "_")]
            label_map[cur] = i

            init = [cur]
            # print(f"CUR: {cur}")
            # print(init * all_counts[j])
            distributions.extend(init * all_counts[j])
        # print(f'Row {i} current count: {len(distributions)}')
        # print(sub_to_super)
        # print(label_map)
    # print(distributions)

def get_dataloaders(cfg, source=True):
    import omegaconf
    cfg = omegaconf.OmegaConf.load('conf/config.yaml')
    #cfg.datasets.dir = './datasets/data'
    SEED=42

    train_set = torchvision.datasets.CIFAR100(root=cfg.datasets.dir, train=True, transform=torchvision.transforms.ToTensor(), download=True)
    test_set = torchvision.datasets.CIFAR100(root=cfg.datasets.dir, train=False, transform=torchvision.transforms.ToTensor())

    # print(len(old_data))
    _populate()
    # targets = old_data.targets
    tr_targets = train_set.targets
    ts_targets = test_set.targets
    targets = tr_targets + ts_targets
    src_idx, tgt_idx= train_test_split(np.arange(len(targets)),
        test_size=0.5, random_state=42, shuffle=True, stratify=distributions)

    train_set.targets = sparse2coarse(tr_targets)
    test_set.targets = sparse2coarse(ts_targets)
    old_data = ConcatDataset([train_set, test_set])
    # probs = [0.35, 0.35, 0.2, 0.05, 0.05]
    # counts = [6000] * 5 # for a single superclass
    # all_counts = [int(p * c) for p, c in zip(probs, counts)]
    # for i in range(10):
    #   all_counts.append([int(p * c) for p, c in zip(probs, counts)])
    # chat gpt this

    # let source and target be the same size?
    if source:
        # pass in indicies that are sampled that are towards a distribution towards 3 labels in the dataset
        return Subset(old_data, src_idx)
    else:
        return Subset(old_data, tgt_idx)
        # pass in indicies that are sampled towards a distribution towards the remain 2
    # source: mostly the first 3 items

# sub_to_super = {'beaver': 'aquatic mammals', 'dolphin': 'aquatic mammals', 'otter': 'aquatic mammals', 'seal': 'aquatic mammals', 'whale': 'aquatic mammals', 'aquarium fish': 'fish', 'flatfish': 'fish', 'ray': 'fish', 'shark': 'fish', 'trout': 'fish', 'orchid': 'flowers', 'poppy': 'flowers', 'rose': 'flowers', 'sunflower': 'flowers', 'tulip': 'flowers', 'bottle': 'food containers', 'bowl': 'food containers', 'can': 'food containers', 'cup': 'food containers', 'plate': 'food containers', 'apple': 'fruit and vegetables', 'mushroom': 'fruit and vegetables', 'orange': 'fruit and vegetables', 'pear': 'fruit and vegetables', 'sweet pepper': 'fruit and vegetables', 'clock': 'household electrical devices', 'keyboard': 'household electrical devices', 'lamp': 'household electrical devices', 'telephone': 'household electrical devices', 'television': 'household electrical devices', 'bed': 'household furniture', 'chair': 'household furniture', 'couch': 'household furniture', 'table': 'household furniture', 'wardrobe': 'household furniture', 'bee': 'insects', 'beetle': 'insects', 'butterfly': 'insects', 'caterpillar': 'insects', 'cockroach': 'insects', 'bear': 'large carnivores', 'leopard': 'large carnivores', 'lion': 'large carnivores', 'tiger': 'large carnivores', 'wolf': 'large carnivores', 'bridge': 'large man-made outdoor things', 'castle': 'large man-made outdoor things', 'house': 'large man-made outdoor things', 'road': 'large man-made outdoor things', 'skyscraper': 'large man-made outdoor things', 'cloud': 'large natural outdoor scenes', 'forest': 'large natural outdoor scenes', 'mountain': 'large natural outdoor scenes', 'plain': 'large natural outdoor scenes', 'sea': 'large natural outdoor scenes', 'camel': 'large omnivores and herbivores', 'cattle': 'large omnivores and herbivores', 'chimpanzee': 'large omnivores and herbivores', 'elephant': 'large omnivores and herbivores', 'kangaroo': 'large omnivores and herbivores', 'fox': 'medium-sized mammals', 'porcupine': 'medium-sized mammals', 'possum': 'medium-sized mammals', 'raccoon': 'medium-sized mammals', 'skunk': 'medium-sized mammals', 'crab': 'non-insect invertebrates', 'lobster': 'non-insect invertebrates', 'snail': 'non-insect invertebrates', 'spider': 'non-insect invertebrates', 'worm': 'non-insect invertebrates', 'baby': 'people', 'boy': 'people', 'girl': 'people', 'man': 'people', 'woman': 'people', 'crocodile': 'reptiles', 'dinosaur': 'reptiles', 'lizard': 'reptiles', 'snake': 'reptiles', 'turtle': 'reptiles', 'hamster': 'small mammals', 'mouse': 'small mammals', 'rabbit': 'small mammals', 'shrew': 'small mammals', 'squirrel': 'small mammals', 'maple_tree': 'trees', 'oak_tree': 'trees', 'palm_tree': 'trees', 'pine_tree': 'trees', 'willow_tree': 'trees', 'bicycle': 'vehicles 1', 'bus': 'vehicles 1', 'motorcycle': 'vehicles 1', 'pickup truck': 'vehicles 1', 'train': 'vehicles 1', 'lawn_mower': 'vehicles 2', 'rocket': 'vehicles 2', 'streetcar': 'vehicles 2', 'tank': 'vehicles 2', 'tractor': 'vehicles 2'}

# label_map = {4: 0, 30: 0, 55: 0, 72: 0, 95: 0, 1: 1, 32: 1, 67: 1, 73: 1, 91: 1, 54: 2, 62: 2, 70: 2, 82: 2, 92: 2, 9: 3, 10: 3, 16: 3, 28: 3, 61: 3, 0: 4, 51: 4, 53: 4, 57: 4, 83: 4, 22: 5, 39: 5, 40: 5, 86: 5, 87: 5, 5: 6, 20: 6, 25: 6, 84: 6, 94: 6, 6: 7, 7: 7, 14: 7, 18: 7, 24: 7, 3: 8, 42: 8, 43: 8, 88: 8, 97: 8, 12: 9, 17: 9, 37: 9, 68: 9, 76: 9, 23: 10, 33: 10, 49: 10, 60: 10, 71: 10, 15: 11, 19: 11, 21: 11, 31: 11, 38: 11, 34: 12, 63: 12, 64: 12, 66: 12, 75: 12, 26: 13, 45: 13, 77: 13, 79: 13, 99: 13, 2: 14, 11: 14, 35: 14, 46: 14, 98: 14, 27: 15, 29: 15, 44: 15, 78: 15, 93: 15, 36: 16, 50: 16, 65: 16, 74: 16, 80: 16, 47: 17, 52: 17, 56: 17, 59: 17, 96: 17, 8: 18, 13: 18, 48: 18, 58: 18, 90: 18}
def sample():
    probs = [0.35, 0.35, 0.2, 0.05, 0.05]
    counts = [6000] * 5 # for a single superclass
    new_counts = [int(p * c) for p, c in zip(probs, counts)]
    src_idx = []
    tgt_idx = []

# get_dataloaders(None, True)
# sample()

# use torch Subset to partition the indicies for source and target
# sample in a distribution of [0.35 0.35 0.2 0.05 0.05] for source for each row