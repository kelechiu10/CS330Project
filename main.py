import time
from typing import Dict
import hydra
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch import optim
from torch.optim import Optimizer
from torch.utils import tensorboard
from torch.utils.data import DataLoader
import numpy as np
import models
from datasets import cifar_c

def train_model(model: nn.Module, dataloaders: Dict[str, DataLoader], criterion, optimizer: Optimizer, writer, cfg):
    since = time.time()

    for epoch in range(cfg.num_epochs):
        model.train()
        for batch in dataloaders.train:
            X, Y = batch
            X = X.to(cfg.device)
            Y = Y.to(cfg.device)
            Y_hat = model(X)
            loss = criterion(Y_hat, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('train/loss', loss.item(), epoch)
        if (epoch + 1) % cfg.save_model_interval == 0:
            model.save(epoch, cfg)
        if (epoch + 1) % cfg.validation.interval == 0:
            model.eval()
            losses = []
            accuracies = []
            for i, batch in enumerate(dataloaders.eval):
                if i >= cfg.validation.num_iterations:
                    break
                X, Y = batch
                X = X.to(cfg.device)
                Y = Y.to(cfg.device)
                Y_hat = model(X)
                loss = criterion(Y_hat, Y)
                losses.append(loss.item())
                accuracies.append(model.calc_accuracy(Y_hat, Y))
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
    model = models.get_cifar_model(cfg.train.model_name, cfg.train.pretrained_dir)
    dataloaders = cifar_c.get_dataloaders(cfg)
    criterion = nn.CrossEntropyLoss() #get_criterion(cfg)
    optimizer = optim.Adam(params=model.parameters(), lr=0.001) #get_optimizer(cfg)
    writer = tensorboard.SummaryWriter(log_dir=cfg.log_dir)
    train_model(model, dataloaders, criterion, optimizer, writer, cfg)


if __name__ == "__main__":
    main()