import os
import time
from typing import Dict
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch import optim
from torch.optim import Optimizer
from torch.utils import tensorboard
from torch.utils.data import DataLoader
import numpy as np
from datasets import cifar_c
from models import models
from tqdm import tqdm
from models.util import get_accuracy


def save_model(model, epoch, cfg):
    if not os.path.isdir(cfg.models.save_dir):
        os.makedirs(cfg.models.save_dir)
    file_path = os.path.join(cfg.models.save_dir, cfg.train.model_name + '_' + cfg.train.dataset_name)
    if not os.path.isdir(file_path):
        os.makedirs(file_path)
    file_name = "epoch_{:03d}.mdl".format(epoch)
    file_name = os.path.join(file_path, file_name)
    torch.save(model.state_dict(), file_name)
    print('model saved in: ', file_name)


def load_model(self, filename):
    state_dict = torch.load(filename, map_location=self.device)
    self.policy.load_state_dict(state_dict)
    self.policy.eval()
    print('policy model ', filename, ' loaded successfully')


def train_model(model: nn.Module, dataloaders: Dict[str, DataLoader], criterion, optimizer: Optimizer, writer, cfg):
    since = time.time()
    model.to(cfg.train.device)
    i = 0
    for epoch in tqdm(range(cfg.train.num_epochs), position=0, leave=False):
        model.train()
        for batch in tqdm(dataloaders['train'], position=1, leave=False):
            i += 1
            X, Y = batch
            X = X.to(cfg.train.device)
            Y = Y.to(cfg.train.device)
            Y_hat = model(X)
            loss = criterion(Y_hat, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('train/loss', loss.item(), i)
        if (epoch + 1) % cfg.train.save_model_interval == 0:
            save_model(model, epoch, cfg)
        if (epoch + 1) % cfg.train.validation.interval == 0:
            model.eval()
            losses = []
            accuracies = []
            for i, batch in enumerate(dataloaders['eval']):
                if i >= cfg.train.validation.num_iterations:
                    break
                X, Y = batch
                X = X.to(cfg.train.device)
                Y = Y.to(cfg.train.device)
                Y_hat = model(X)
                loss = criterion(Y_hat, Y)
                losses.append(loss.item())
                accuracies.append(get_accuracy(Y_hat, Y))
            mean_loss = np.mean(losses)
            mean_accuracy = np.mean(accuracies)
            print(
                f'Validation: '
                f'loss: {mean_loss:.3f}, '
                f'support accuracy: {mean_accuracy:.3f}, '
            )
            writer.add_scalar('val/loss', mean_loss, epoch)
            writer.add_scalar(
                'val/accuracy',
                mean_accuracy,
                epoch
            )

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    model, layers = models.get_cifar_model(cfg.train.model_name, cfg.train.pretrained_dir)
    dataloaders = cifar_c.get_dataloaders(cfg)
    criterion = nn.CrossEntropyLoss() #get_criterion(cfg)
    optimizer = optim.Adam(params=model.parameters(), lr=0.001) #get_optimizer(cfg)
    writer = tensorboard.SummaryWriter(log_dir=os.path.join(cfg.logging.dir, cfg.train.model_name + '_' + cfg.train.dataset_name))
    train_model(model, dataloaders, criterion, optimizer, writer, cfg)


if __name__ == "__main__":
    main()
