import os
import time
from functools import partial
from typing import Dict
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch import optim
from torch.utils import tensorboard
from torch.utils.data import DataLoader
import numpy as np
import datasets
from models import resnet_models
from tqdm import tqdm
from models.util import get_accuracy
import optimizers
from SMPyBandits.Policies import DiscountedThompson, BESA, SWklUCBPlus, WrapRange
from torchvision.models import resnet50, ResNet50_Weights
from datasets.util import random_split
from optimizers.MABOptimizer import EpsilonGreedyFixed


def save_model(model, epoch, cfg):
    if not os.path.isdir(cfg.models.save_dir):
        os.makedirs(cfg.models.save_dir)
    file_path = os.path.join(cfg.models.save_dir, cfg.train.model_name + '_' + cfg.datasets.name)
    if not os.path.isdir(file_path):
        os.makedirs(file_path)
    file_name = "epoch_{:03d}.mdl".format(epoch)
    file_name = os.path.join(file_path, file_name)
    torch.save(model.state_dict(), file_name)
    print('model saved in: ', file_name)


def load_model(model, filename):
    state_dict = torch.load(filename)
    model.load_state_dict(state_dict)
    model.eval()
    print('policy model ', filename, ' loaded successfully')


def train_model(model: nn.Module, dataloaders: Dict[str, DataLoader], criterion, optimizer, writer, cfg):
    since = time.time()
    model.to(cfg.train.device)
    itr = 0
    for epoch in tqdm(range(cfg.train.num_epochs), position=0, leave=False):
        model.train()
        for batch in tqdm(dataloaders['train'], position=1, leave=False):
            itr += 1
            X, Y = batch
            X = X.to(cfg.train.device)
            Y = Y.to(cfg.train.device)
            optimizer.zero_grad()
            Y_hat = model(X)
            loss = criterion(Y_hat, Y)
            optimizer.step(loss)
            writer.add_scalar('train/loss', loss.item(), itr)

        if (epoch + 1) % cfg.train.save_model_interval == 0:
            save_model(model, epoch, cfg)
        if (epoch + 1) % cfg.train.validation.interval == 0:
            model.eval()
            losses = []
            accuracies = []
            accuracies_train = []
            for i, batch in enumerate(dataloaders['test']):
                if i >= cfg.train.validation.num_iterations:
                    break
                X, Y = batch
                X = X.to(cfg.train.device)
                Y = Y.to(cfg.train.device)
                Y_hat = model(X)
                loss = criterion(Y_hat, Y)
                losses.append(loss.item())
                accuracies.append(get_accuracy(Y_hat, Y))
            for i, batch in enumerate(dataloaders['train']):
                if i >= cfg.train.validation.num_iterations:
                    break
                X, Y = batch
                X = X.to(cfg.train.device)
                Y = Y.to(cfg.train.device)
                Y_hat = model(X)
                accuracies_train.append(get_accuracy(Y_hat, Y))
            model.train()
            mean_loss = np.mean(losses)
            mean_accuracy = np.mean(accuracies)
            mean_accuracies_train = np.mean(accuracies_train)
            print(
                f'Validation: '
                f'loss: {mean_loss:.3f}, '
                f'support accuracy: {mean_accuracy:.3f}, '
                f'train accuracy: {mean_accuracies_train:.3f}, '
            )
            writer.add_scalar('val/loss', mean_loss, epoch)
            writer.add_scalar(
                'val/accuracy',
                mean_accuracy,
                epoch
            )
            writer.add_scalar(
                'train/accuracy',
                mean_accuracies_train,
                epoch
            )
    save_model(model, cfg.train.num_epochs, cfg)
    accuracies = []
    for i, batch in enumerate(dataloaders['test']):
        X, Y = batch
        X = X.to(cfg.train.device)
        Y = Y.to(cfg.train.device)
        Y_hat = model(X)
        loss = criterion(Y_hat, Y)
        accuracies.append(get_accuracy(Y_hat, Y))
    accuracy = np.mean(accuracies)
    print(f'Final Accuracy: {accuracy} ({accuracy * (1 - accuracy) / np.sqrt(len(accuracies))})')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


def get_model(cfg):
    if cfg.models.model_checkpoint == 'cifar':
        return resnet_models.get_cifar_model(cfg.models.name, cfg.train.pretrained_dir)
    elif cfg.models.model_checkpoint == 'imagenet':
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
    else:
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        load_model(model, cfg.models.model_checkpoint)
    # TODO: change these layers?
    layers = [model.conv1, model.layer1, model.layer2, model.layer3, model.layer4, model.fc]
    return model, layers


def get_dataloader(cfg):
    if cfg.datasets.name == 'cifarc':
        base_dataset = datasets.cifar(cfg, corrupted=True)
    elif cfg.datasets.name == 'cifar':
        base_dataset = datasets.cifar(cfg, corrupted=False)
    elif cfg.datasets.name == 'cifar_flip':
        return datasets.cifar_flip(cfg)
    # elif cfg.datasets.name == 'living17_source':
    #     return datasets.living17(cfg, source=True)
    # elif cfg.datasets.name == 'living17_target':
    #     return datasets.living17(cfg, source=False)
    # elif cfg.datasets.name == 'sp_cifar_100_source':
    #     return datasets.sp_cifar_100(cfg, source=True)
    # elif cfg.datasets.name == 'sp_cifar_100_target':
    #     return datasets.sp_cifar_100(cfg, source=False)
    else:
        raise f'Unknown dataset \'{cfg.datasets.name}\''

    train_dataset, test_dataset = random_split(base_dataset, cfg.datasets.split)
    dataloaders = dict()
    num_workers = cfg.train.num_workers
    batch_size = cfg.train.batch_size
    dataloaders['train'] = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        num_workers=num_workers, shuffle=cfg.datasets.shuffle
    )
    dataloaders['test'] = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=num_workers, shuffle=cfg.datasets.shuffle
    )
    return dataloaders


WrapRange = partial(WrapRange, lower=0.0, amplitude=0.2)

MAB_POLICIES = {
    'epsilon_greedy': partial(WrapRange, policy=EpsilonGreedyFixed),
    'discounted_thompson': partial(WrapRange, policy=DiscountedThompson),
    'BESA': partial(BESA, minPullsOfEachArm=3),
    'SWklUCBPlus': partial(WrapRange, policy=partial(SWklUCBPlus, tau=100))
}


def get_optimizer(cfg, opt, opt_variation, layers, model):
    num_layers = len(layers)
    if opt == 'MAB':
        policy = MAB_POLICIES[opt_variation['type']](nbArms=num_layers)
        optimizer = optimizers.MABOptimizer(layers, lr=cfg.train.lr, mab_policy=policy)
        return optimizer
    elif opt == 'layerwise':
        return optimizers.SingleLayerOptimizer(layers, opt_variation['idx'])
    elif opt == 'gradnorm':
        return optimizers.GradNorm(layers, lr=cfg.train.lr)
    elif opt == 'full':
        return optim.Adam(params=model.parameters(), lr=cfg.train.lr)
    else:
        raise f'Unknown optimizer \'{opt}\''


def get_variants(cfg, opt):
    if opt == 'MAB':
        types = cfg.optimizer.MAB.type.split(' ')
        return [{'type': t} for t in types]
    elif opt == 'layerwise':
        if cfg.layerwise.idx == -1:
            model, layers = get_model(cfg)
            return [{'idx': idx, 'type': idx} for idx in range(len(layers))]
        elif isinstance(cfg.layerwise.idx, list):
            return [{'idx': idx, 'type': idx} for idx in cfg.layerwise.idx]
        else:
            return [{'idx': cfg.layerwise.idx, 'type': cfg.layerwise.idx}]
    elif opt == 'full':
        return [{'type': ''}]
    elif opt == 'gradnorm':
        return [{'type': ''}]
    return [{'type': ''}]


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    opts = cfg.optimizer.name.split(' ')
    for opt in opts:
        for opt_variation in get_variants(cfg, opt):
            model, layers = get_model(cfg)
            dataloaders = get_dataloader(cfg)
            criterion = nn.CrossEntropyLoss()
            optimizer = get_optimizer(cfg, opt, opt_variation, layers, model)
            print(f'Starting finetuning with {opt} {opt_variation["type"]}')
            writer = tensorboard.SummaryWriter(
                log_dir=os.path.join(cfg.logging.dir, cfg.models.model_checkpoint + '_' + cfg.optimizer.name + '_' +
                                     opt_variation['type'] + '_' + cfg.datasets.name))
            train_model(model, dataloaders, criterion, optimizer, writer, cfg)

    # model, layers = resnet_models.get_cifar_model(cfg.train.model_name, cfg.train.pretrained_dir)
    # num_layers = len(layers)
    # for idx in range(num_layers):
    #     model, layers = resnet_models.get_cifar_model(cfg.train.model_name, cfg.train.pretrained_dir)
    # # model = load_pretrained_model(model_name='Standard', model_dir=cfg.train.pretrained_dir, dataset='cifar10')
    #     dataloaders = cifar_flip.get_dataloaders(cfg)
    #     criterion = nn.CrossEntropyLoss() #get_criterion(cfg)
    #     optimizer = single_layer.SingleLayerOptimizer(layers, idx)#optim.Adam(params=model.parameters(), lr=0.001) #get_optimizer(cfg)
    #     writer = tensorboard.SummaryWriter(log_dir=os.path.join(cfg.logging.dir, cfg.train.model_name + '_singlelayer_' + str(idx) + '_' + cfg.datasets.name))
    #     train_model(model, dataloaders, criterion, optimizer, writer, cfg)
    # model, layers = models.get_cifar_model(cfg.train.model_name, cfg.train.pretrained_dir)
    # num_layers = len(layers)
    # dataloaders = cifar_flip.get_dataloaders(cfg)
    # criterion = nn.CrossEntropyLoss()
    # policy = EpsilonGreedy(nbArms=num_layers)
    # optimizer = MABOptimizer.MABOptimizer(layers, lr=1e-3, mab_policy=policy)
    # writer = tensorboard.SummaryWriter(log_dir=os.path.join(cfg.logging.dir, cfg.train.model_name + '_MAB_epsilon_greedy_policy_' + '_' + cfg.train.dataset_name))
    # train_model(model, dataloaders, criterion, optimizer, writer, cfg)


if __name__ == "__main__":
    main()
