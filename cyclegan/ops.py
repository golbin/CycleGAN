"""
Utilities for tensor operations
"""
import torch


def optimize(loss, optimizer):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def variable(data):
    if torch.cuda.is_available():
        return torch.autograd.Variable(data.cuda())
    else:
        return torch.autograd.Variable(data)


def normal_weights_initializer(m, mean, std):
    if isinstance(m, torch.nn.ConvTranspose2d) or isinstance(m, torch.nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class Activation:
    ReLU = 1
    LeakyReLU = 2
    Tanh = 3

    def __init__(self, activation_fn=None, batch_norm=True, output_size=0):
        self.activation_fn = self._init_activation_fn(activation_fn)
        self.batch_norm = self._init_batch_norm(batch_norm, output_size)

    def _init_activation_fn(self, fn):
        if fn == self.ReLU:
            return torch.nn.ReLU(True)
        elif fn == self.LeakyReLU:
            return torch.nn.LeakyReLU(0.2, True)
        elif fn == self.Tanh:
            return torch.nn.Tanh()
        else:
            return None

    @staticmethod
    def _init_batch_norm(batch_norm, output_size):
        if batch_norm:
            if torch.cuda.is_available():
                return torch.nn.InstanceNorm2d(output_size).cuda()
            else:
                return torch.nn.InstanceNorm2d(output_size)
        else:
            return None

    def apply(self, output):
        if self.batch_norm:
            output = self.batch_norm(output)

        if self.activation_fn:
            output = self.activation_fn(output)

        return output
