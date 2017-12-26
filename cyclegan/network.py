"""
The neural network architecture of Cycle GAN
"""
import torch
import torch.nn.modules

from cyclegan.utils.ops import Activation, normal_weights_initializer


class ConvLayer(torch.nn.Module):
    def __init__(self, input_size, output_size,
                 kernel_size=3, stride=2, padding=1,
                 activation_fn=None, batch_norm=True):
        super(ConvLayer, self).__init__()

        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding)

        self.activation = Activation(activation_fn, batch_norm, output_size)

    def forward(self, x):
        output = self.conv(x)
        output = self.activation.apply(output)

        return output


class DeconvLayer(torch.nn.Module):
    def __init__(self, input_size, output_size,
                 kernel_size=3, stride=2, padding=1, output_padding=1,
                 activation_fn=None, batch_norm=True):
        super(DeconvLayer, self).__init__()

        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, output_padding)

        self.activation = Activation(activation_fn, batch_norm, output_size)

    def forward(self, x):
        output = self.deconv(x)
        output = self.activation.apply(output)

        return output


class ResnetBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=0):
        super(ResnetBlock, self).__init__()

        self.pad = torch.nn.ReflectionPad2d(padding)

        self.conv1 = ConvLayer(num_filter, num_filter,
                               kernel_size=kernel_size, stride=stride, padding=0,
                               activation_fn=Activation.ReLU, batch_norm=True)

        self.conv2 = ConvLayer(num_filter, num_filter,
                               kernel_size=kernel_size, stride=stride, padding=0,
                               activation_fn=None, batch_norm=True)

    def forward(self, x):
        output = self.pad(x)
        output = self.conv1(output)
        output = self.pad(output)
        output = self.conv2(output)

        return x + output


class Generator(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_filter=32, num_resnet=6):
        super(Generator, self).__init__()

        self.encoder = self._init_encoder_layer(in_channels, num_filter)

        self.transformer = self._init_reset_block(num_filter, num_resnet)

        self.decoder = self._init_decoder_layer(out_channels, num_filter)

    @staticmethod
    def _init_encoder_layer(in_channels, num_filter):
        pad = torch.nn.ReflectionPad2d(3)

        conv1 = ConvLayer(in_channels, num_filter,
                          kernel_size=7, stride=1, padding=0,
                          activation_fn=Activation.ReLU, batch_norm=True)

        conv2 = ConvLayer(num_filter, num_filter * 2,
                          kernel_size=3, stride=2, padding=1,
                          activation_fn=Activation.ReLU, batch_norm=True)

        conv3 = ConvLayer(num_filter * 2, num_filter * 4,
                          kernel_size=3, stride=2, padding=1,
                          activation_fn=Activation.ReLU, batch_norm=True)

        return torch.nn.Sequential(pad, conv1, conv2, conv3)

    @staticmethod
    def _init_reset_block(num_filter, num_resnet):
        resnet_blocks = []

        for i in range(num_resnet):
            resnet = ResnetBlock(num_filter * 4, 3, 1, 1)
            resnet_blocks.append(resnet)

        return torch.nn.Sequential(*resnet_blocks)

    @staticmethod
    def _init_decoder_layer(out_channels, num_filter):
        deconv1 = DeconvLayer(num_filter * 4, num_filter * 2,
                              kernel_size=3, stride=2, padding=1, output_padding=1,
                              activation_fn=Activation.ReLU, batch_norm=True)
        
        deconv2 = DeconvLayer(num_filter * 2, num_filter,
                              kernel_size=3, stride=2, padding=1, output_padding=1,
                              activation_fn=Activation.ReLU, batch_norm=True)

        pad = torch.nn.ReflectionPad2d(3)

        conv1 = ConvLayer(num_filter, out_channels,
                          kernel_size=7, stride=1, padding=0,
                          activation_fn=Activation.Tanh, batch_norm=False)

        return torch.nn.Sequential(deconv1, deconv2, pad, conv1)

    def init_weight(self, mean, std):
        for m in self._modules:
            normal_weights_initializer(self._modules[m], mean, std)

    def forward(self, x):
        output = self.encoder(x)
        output = self.transformer(output)
        output = self.decoder(output)

        return output


class Discriminator(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_filter=64):
        super(Discriminator, self).__init__()

        self.discriminator = self._init_layer(in_channels, out_channels, num_filter)

    @staticmethod
    def _init_layer(in_channels, out_channels, num_filter):
        conv1 = ConvLayer(in_channels, num_filter,
                          kernel_size=4, stride=2, padding=1,
                          activation_fn=Activation.LeakyReLU, batch_norm=False)

        conv2 = ConvLayer(num_filter, num_filter * 2,
                          kernel_size=4, stride=2, padding=1,
                          activation_fn=Activation.LeakyReLU, batch_norm=True)

        conv3 = ConvLayer(num_filter * 2, num_filter * 4,
                          kernel_size=4, stride=2, padding=1,
                          activation_fn=Activation.LeakyReLU, batch_norm=True)

        conv4 = ConvLayer(num_filter * 4, num_filter * 8,
                          kernel_size=4, stride=1, padding=1,
                          activation_fn=Activation.LeakyReLU, batch_norm=True)

        conv5 = ConvLayer(num_filter * 8, out_channels,
                          kernel_size=4, stride=1, padding=1,
                          activation_fn=None, batch_norm=False)

        return torch.nn.Sequential(conv1, conv2, conv3, conv4, conv5)

    def init_weight(self, mean, std):
        for m in self._modules:
            normal_weights_initializer(self._modules[m], mean, std)

    def forward(self, x):
        output = self.discriminator(x)

        return output

