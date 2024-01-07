import numpy as np
import torch
import torch.nn as nn

import wandb
from wandb.keras import WandbCallback

import time

import matplotlib.pyplot as plt


class torch_trainer():

    def __init__(self, model, criterion, optimizer, epochs, dataloader, device='cuda', scheduler=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.dataloader = dataloader
        self.device = device
        self.history = train_history()
        self.metrics = None
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)
        self.wandb_log()
        self.start_time = None

    def train(self):

        self.history.clear()
        loss = 0
        val_loss = 0

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        

        for epoch in range(self.epochs):

            epoch_loss = []
            epoch_val_loss = []
            epoch_metrics = []
            items = len(self.dataloader['train'])
            self.start_time = time.time()

            self.model.train()
            item = 1
            for batch in self.dataloader['train']:
                self.optimizer.zero_grad()
                data = batch[0].to(device)
                preds = self.model(data)
                del data
                target = batch[1].to(device)
                loss = self.criterion(preds, target)
                loss.backward()
                del target
                del preds
                epoch_loss.append(loss.numpy(force=True))
                self.optimizer.step()
                self.print_log(epoch_log=False, epoch=epoch, item=item, items=items, train_loss=loss)
                item += 1

            items = len(self.dataloader['test'])
            self.model.eval()
            item = 1
            with torch.no_grad():
                for batch in self.dataloader['test']:
                    data = batch[0].to(device)
                    preds = self.model(data)
                    del data
                    target = batch[1].to(device)
                    val_loss = self.criterion(preds, target)
                    if self.metrics:
                        metric = self.metrics(preds, target)
                        epoch_metrics.append(metric.numpy(force=True))
                    del target
                    del preds
                    epoch_val_loss.append(val_loss.numpy(force=True))
                    self.print_log(epoch_log=False, epoch=epoch, item=item, items=items, train_loss=loss,
                                   val_loss=val_loss)
                    item += 1

            loss = np.array(epoch_loss).mean()
            val_loss = np.array(epoch_val_loss).mean()
            metric = np.array(epoch_metrics).mean() if self.metrics else 0

            if self.scheduler:
                self.scheduler.step(loss)

            self.history.loss.append(loss)
            self.history.val_loss.append(val_loss)
            self.history.metric.append(metric)

            self.print_log(epoch_log=True, epoch=epoch, item=item, items=items, train_loss=loss, val_loss=val_loss,
                           metric=metric)

        return self.history

    def set_metrics(self, metric_name, metric_function):
        self.metric_name = metric_name
        self.metrics = metric_function

    def print_log(self, epoch_log=True, epoch=0, item=0, items=0, train_loss=0, val_loss=0, metric=0):
        lr = self.optimizer.param_groups[0]["lr"]
        log = {'epoch': epoch + 1,
               'epoch/epochs': str(epoch + 1) + '/' + str(self.epochs),
               'train/loss': train_loss,
               'val/loss': val_loss}
        if epoch_log:
            log['lr'] = lr
            if self.metrics:
                metric_key = f'metric/{self.metric_name}'
                log[metric_key] = metric

        if epoch_log:
            log_str = f'epoch {epoch + 1}/{self.epochs}... '
        else:
            log_str = f'epoch {epoch + 1}/{self.epochs}... batch {item}/{items} '

        for item in [item for item in log if item != 'epoch' and item != 'epoch/epochs' and item != 'item']:
            if torch.is_tensor(log[item]):
                value = log[item].item()
                value = f'{value:.4f}'
            else:
                if type(log[item]) == np.float32:
                    value = f'{log[item]:.4f}'
                else:
                    value = log[item]
            log_str = log_str + ' ' + item + ': ' + str(value)
        dt = int(time.time() - self.start_time)
        log_str_time = f'; time: {dt//3600:03d}:{(dt//60)%60:02d}:{dt%60:02d}'
        log_str = log_str + log_str_time

        if epoch_log:
            print(log_str, end='\n')
        else:
            print(log_str, end='\r')

        if self.wandb and epoch_log:
            wandb.log(log, step=epoch)

    def wandb_log(self, wandb=False):
        self.wandb = wandb


class train_history():
    loss = []
    val_loss = []
    metric = []

    def init(self):
        self.loss = []
        self.val_loss = []
        self.metric = []

    def clear(self):
        self.loss.clear()
        self.val_loss.clear()
        self.metric.clear()

    def plot(self, train=True, val=True, metric=None):
        if train:
            plt.plot(self.loss, label='train loss')
        if val:
            plt.plot(self.val_loss, label='val loss')
        if metric:
            plt.plot(self.metric, label=metric)
        plt.legend()
        plt.show()       