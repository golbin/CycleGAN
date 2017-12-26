"""
A script for CycleGAN training
"""
import os

import cyclegan.config as config
from cyclegan.utils import tester
from cyclegan.utils.logger import Logger
from cyclegan.model import CycleGAN
from cyclegan.utils.data import DataLoader


class Trainer:
    def __init__(self, args):
        self.args = args

        self.data_loader_A = DataLoader(args.data_A_dir, args.input_size, args.batch_size, shuffle=True)
        self.data_loader_B = DataLoader(args.data_B_dir, args.input_size, args.batch_size, shuffle=True)

        self.cycleGAN = CycleGAN(args.ngf, args.ndf, args.num_resnet,
                                 args.lrG, args.lrD, args.beta1, args.beta2,
                                 args.lambdaA, args.lambdaB, args.num_pool)

        self.logger = Logger()

    def run_epoch(self):
        for i, (real_A, real_B) in enumerate(zip(self.data_loader_A, self.data_loader_B)):
            losses = self.cycleGAN.train(real_A, real_B)

            self.logger.add_losses(losses)
            self.logger.print_last_loss()

    def run(self):
        for epoch in range(self.args.num_epochs):
            if (epoch + 1) > self.args.decay_epoch:
                self.cycleGAN.decay_optimizer(self.args.num_epochs, self.args.decay_epoch)

            self.run_epoch()

            self.logger.next_epoch()
            self.logger.print_avg_loss()

            if self.args.test_data_A_dir and self.args.test_data_B_dir:
                tester.generate_testset(epoch, self.cycleGAN, self.args)

        self.cycleGAN.save(args.model_dir)

        self.logger.print_status()


def prepare_output_dir(args):
    args.log_dir = os.path.join(args.output_dir, 'log')
    args.model_dir = os.path.join(args.output_dir, 'model')
    args.test_output_dir = os.path.join(args.output_dir, 'test')

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.test_output_dir, exist_ok=True)


if __name__ == '__main__':
    args = config.parse_args(is_training=True)

    prepare_output_dir(args)

    trainer = Trainer(args)

    trainer.run()
