"""
Utilities for logging especially for loss
"""
import datetime

import torch

from cyclegan.model import LossType


class Logger:
    def __init__(self):
        self.epoch = 0
        self.step = 0
        self.losses_avg = self.init_losses()
        self.losses_in_epoch = self.init_losses()
        self.start_time = datetime.datetime.now()

    @staticmethod
    def init_losses():
        losses = {
            LossType.D_A: [],
            LossType.D_B: [],
            LossType.G_AB: [],
            LossType.G_BA: [],
            LossType.cycle_A: [],
            LossType.cycle_B: []
        }

        return losses

    def add_losses(self, losses):
        """
        Add loss for logging
        :param losses: dictionary with LosType keys
        :return:
        """
        for key, loss in losses.items():
            if key in self.losses_in_epoch:
                self.losses_in_epoch[key].append(loss.data[0])

        self.step += 1

    def next_epoch(self):
        for key, losses in self.losses_in_epoch.items():
            avg_loss = torch.mean(torch.FloatTensor(losses))
            self.losses_avg[key].append(avg_loss)

        self.losses_in_epoch = self.init_losses()

        self.epoch += 1

    def print_last_loss(self):
        print('Epoch [%d], Step [%d] : D_A_loss = %.4f, D_B_loss = %.4f, G_AB_loss = %.4f, G_BA_loss = %.4f'
              % (self.epoch, self.step,
                 self.losses_in_epoch[LossType.D_A][-1],
                 self.losses_in_epoch[LossType.D_B][-1],
                 self.losses_in_epoch[LossType.G_AB][-1],
                 self.losses_in_epoch[LossType.G_BA][-1]))

    def print_avg_loss(self):
        print('Avg. Epoch [%d], Step [%d] : D_A_loss = %.4f, D_B_loss = %.4f, G_AB_loss = %.4f, G_BA_loss = %.4f'
              % (self.epoch, self.step,
                 self.losses_avg[LossType.D_A][-1],
                 self.losses_avg[LossType.D_B][-1],
                 self.losses_avg[LossType.G_AB][-1],
                 self.losses_avg[LossType.G_BA][-1]))

    def print_status(self):
        print('Epoch [%d], Step [%d], Total time : %s'
              % (self.epoch, self.step, datetime.datetime.now() - self.start_time))
