"""
A script for transferring
"""
import os

import scipy.misc as misc

import cyclegan.config as config
from cyclegan.model import CycleGAN
from cyclegan.utils.data import load_file


class Transfer:
    def __init__(self, args):
        self.args = args

        self.cycleGAN = CycleGAN(args.ngf, args.ndf, args.num_resnet,
                                 args.lrG, args.lrD, args.beta1, args.beta2,
                                 args.lambdaA, args.lambdaB, args.num_pool)

        self.cycleGAN.load(args.model, args.type)

    def generate(self, data_src):
        if self.args.type is 'BtoA':
            generator = self.cycleGAN.generate_B_to_A
        else:
            generator = self.cycleGAN.generate_A_to_B

        generated, _ = generator(data_src, recon=False)

        generated = generated[0].cpu().data.numpy().transpose(1, 2, 0)

        return generated

    def run_dir(self, src_dir, out_dir):
        for filename in os.listdir(src_dir):
            src_path = os.path.join(src_dir, filename)
            out_path = os.path.join(out_dir, filename)

            self.run_file(src_path, out_path)

    def run_file(self, src_path, out_path):
        print('{0} -> {1}'.format(src_path, out_path))

        data_src = load_file(src_path, self.args.input_size)

        data_out = self.generate(data_src)

        misc.imsave(out_path, data_out)


def prepare_output_dir(args):
    args.log_dir = os.path.join(args.output_dir, 'log')
    args.model_dir = os.path.join(args.output_dir, 'model')
    args.test_output_dir = os.path.join(args.output_dir, 'test')

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.test_output_dir, exist_ok=True)


if __name__ == '__main__':
    args = config.parse_args()

    transfer = Transfer(args)

    print('Running {0} Transfer'.format(args.type))

    if args.src_dir and args.out_dir:
        os.makedirs(args.out_dir, exist_ok=False)
        transfer.run_dir(args.src_dir, args.out_dir)
    elif args.src and args.out:
        transfer.run_file(args.src, args.out)
    else:
        config.print_help()
