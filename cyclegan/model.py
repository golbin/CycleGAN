"""
Main model of CycleGAN
Calculate loss and optimizing
"""
import os
import itertools
from enum import Enum

import torch

import cyclegan.utils.ops as ops
from cyclegan.utils.data import ImagePool
from cyclegan.network import Generator, Discriminator


class LossType(Enum):
    G = 1
    D_A = 2
    D_B = 3
    G_AB = 4
    G_BA = 5
    cycle_A = 6
    cycle_B = 7


class CycleGAN:
    def __init__(self, ngf=32, ndf=64, num_resnet=6,
                 lrG=0.0002, lrD=0.0002, beta1=0.5, beta2=0.999,
                 lambdaA=10, lambdaB=10, num_pool=50):

        self.lrG, self.lrD = lrG, lrD
        self.beta1, self.beta2 = beta1, beta2
        self.lambdaA, self.lambdaB = lambdaA, lambdaB

        self.G_AB, self.G_BA, self.D_A, self.D_B =\
            self._init_network(ngf, ndf, num_resnet)

        self.MSE_loss, self.L1_loss = self._init_loss()

        self.G_optimizer, self.D_A_optimizer, self.D_B_optimizer =\
            self._init_optimizer(lrG, lrD, beta1, beta2)

        self.fake_A_pool = ImagePool(num_pool)
        self.fake_B_pool = ImagePool(num_pool)

        self._prepare_for_gpu()

    @staticmethod
    def _init_network(ngf, ndf, num_resnet):
        G_AB = Generator(3, 3, ngf, num_resnet)
        G_BA = Generator(3, 3, ngf, num_resnet)
        D_A = Discriminator(3, 1, ndf)
        D_B = Discriminator(3, 1, ndf)

        G_AB.init_weight(mean=0.0, std=0.02)
        G_BA.init_weight(mean=0.0, std=0.02)
        D_A.init_weight(mean=0.0, std=0.02)
        D_B.init_weight(mean=0.0, std=0.02)

        return G_AB, G_BA, D_A, D_B

    def _prepare_for_gpu(self):
        if torch.cuda.device_count() > 1:
            print("{0} GPUs are detected.".format(torch.cuda.device_count()))

            self.G_AB = torch.nn.DataParallel(self.G_AB)
            self.G_BA = torch.nn.DataParallel(self.G_BA)
            self.D_A = torch.nn.DataParallel(self.D_A)
            self.D_B = torch.nn.DataParallel(self.D_B)

        if torch.cuda.is_available():
            self.G_AB.cuda()
            self.G_BA.cuda()
            self.D_A.cuda()
            self.D_B.cuda()

    @staticmethod
    def _init_loss():
        if torch.cuda.is_available():
            MSE_loss = torch.nn.MSELoss().cuda()
            L1_loss = torch.nn.L1Loss().cuda()
        else:
            MSE_loss = torch.nn.MSELoss()
            L1_loss = torch.nn.L1Loss()

        return MSE_loss, L1_loss

    def _init_optimizer(self, lrG, lrD, beta1, beta2):
        G_optimizer = torch.optim.Adam(itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()),
                                       lr=lrG, betas=(beta1, beta2))
        D_A_optimizer = torch.optim.Adam(self.D_A.parameters(), lr=lrD, betas=(beta1, beta2))
        D_B_optimizer = torch.optim.Adam(self.D_B.parameters(), lr=lrD, betas=(beta1, beta2))

        return G_optimizer, D_A_optimizer, D_B_optimizer

    def decay_optimizer(self, num_epochs, decay_epoch):
        self.G_optimizer.param_groups[0]['lr'] -= self.lrG / (num_epochs - decay_epoch)
        self.D_A_optimizer.param_groups[0]['lr'] -= self.lrD / (num_epochs - decay_epoch)
        self.D_B_optimizer.param_groups[0]['lr'] -= self.lrD / (num_epochs - decay_epoch)

    def train_generator(self, real_A, real_B):
        # Train G : A -> B
        fake_B = self.G_AB(real_A)
        D_B_result = self.D_B(fake_B)
        G_AB_loss = self.MSE_loss(D_B_result, ops.variable(torch.ones(D_B_result.size())))

        # Reconstruct and calculate cycle loss : fake B -> A
        recon_A = self.G_BA(fake_B)
        cycle_A_loss = self.L1_loss(recon_A, real_A) * self.lambdaA

        # Train G : B -> A
        fake_A = self.G_BA(real_B)
        D_A_result = self.D_A(fake_A)
        G_BA_loss = self.MSE_loss(D_A_result, ops.variable(torch.ones(D_A_result.size())))

        # Reconstruct and calculate cycle loss : fake A -> B
        recon_B = self.G_AB(fake_A)
        cycle_B_loss = self.L1_loss(recon_B, real_B) * self.lambdaB

        # Optimize G for training generator
        G_loss = G_AB_loss + G_BA_loss + cycle_A_loss + cycle_B_loss
        ops.optimize(G_loss, self.G_optimizer)

        losses = {
            LossType.G: G_loss,
            LossType.G_AB: G_AB_loss,
            LossType.G_BA: G_BA_loss,
            LossType.cycle_A: cycle_A_loss,
            LossType.cycle_B: cycle_B_loss
        }

        return losses, fake_A, fake_B

    def train_discriminator(self, real_A, real_B, fake_A, fake_B):
        # Train D for A
        fake_A = self.fake_A_pool.query(fake_A)
        D_A_real_result = self.D_A(real_A)
        D_A_fake_result = self.D_A(fake_A)
        D_A_real_loss = self.MSE_loss(D_A_real_result, ops.variable(torch.ones(D_A_real_result.size())))
        D_A_fake_loss = self.MSE_loss(D_A_fake_result, ops.variable(torch.zeros(D_A_fake_result.size())))

        # Optimize D_A for training discriminator for A
        D_A_loss = (D_A_real_loss + D_A_fake_loss) * 0.5
        ops.optimize(D_A_loss, self.D_A_optimizer)

        # Train D for B
        fake_B = self.fake_B_pool.query(fake_B)
        D_B_real_result = self.D_B(real_B)
        D_B_fake_result = self.D_B(fake_B)
        D_B_real_loss = self.MSE_loss(D_B_real_result, ops.variable(torch.ones(D_B_real_result.size())))
        D_B_fake_loss = self.MSE_loss(D_B_fake_result, ops.variable(torch.zeros(D_B_fake_result.size())))

        # Optimize D_A for training discriminator for A
        D_B_loss = (D_B_real_loss + D_B_fake_loss) * 0.5
        ops.optimize(D_B_loss, self.D_B_optimizer)

        losses = {
            LossType.D_A: D_A_loss,
            LossType.D_B: D_B_loss
        }

        return losses

    def train(self, real_A, real_B):
        real_A = ops.variable(real_A)
        real_B = ops.variable(real_B)

        losses_G, fake_A, fake_B = self.train_generator(real_A, real_B)
        losses_D = self.train_discriminator(real_A, real_B, fake_A, fake_B)

        losses = {**losses_G, **losses_D}

        return losses

    def generate_A_to_B(self, real_A, recon=False):
        fake_B = self.G_AB(ops.variable(real_A))
        recon_A = None

        if recon:
            recon_A = self.G_BA(fake_B)

        return fake_B, recon_A

    def generate_B_to_A(self, real_B, recon=False):
        fake_A = self.G_BA(ops.variable(real_B))
        recon_B = None

        if recon:
            recon_B = self.G_AB(fake_A)

        return fake_A, recon_B

    def load(self, model_dir, generator='ALL', discriminator=False):
        """
        Load pre-trained model
        :param model_dir:
        :param generator: 'ALL', 'AtoB', 'BtoA'
        :param discriminator: True/False
        :return:
        """
        print("Loading model from {0}".format(model_dir))

        if generator is ['AtoB', 'ALL']:
            self.G_AB.load_state_dict(torch.load(os.path.join(model_dir, 'generator_AB_param.pkl')))

        if generator in ['BtoA', 'ALL']:
            self.G_BA.load_state_dict(torch.load(os.path.join(model_dir, 'generator_BA_param.pkl')))

        if discriminator:
            self.D_A.load_state_dict(torch.load(os.path.join(model_dir, 'discriminator_A_param.pkl')))
            self.D_B.load_state_dict(torch.load(os.path.join(model_dir, 'discriminator_B_param.pkl')))

    def save(self, model_dir):
        print("Saving model into {0}".format(model_dir))

        torch.save(self.G_AB.state_dict(), os.path.join(model_dir, 'generator_AB_param.pkl'))
        torch.save(self.G_BA.state_dict(), os.path.join(model_dir, 'generator_BA_param.pkl'))
        torch.save(self.D_A.state_dict(), os.path.join(model_dir, 'discriminator_A_param.pkl'))
        torch.save(self.D_B.state_dict(), os.path.join(model_dir, 'discriminator_B_param.pkl'))
