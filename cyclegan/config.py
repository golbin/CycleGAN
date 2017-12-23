"""
Training Options
"""
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--input_size', type=int, default=256, help='input size')
parser.add_argument('--num_pool', type=int, default=50, help='number of image pool')
parser.add_argument('--ngf', type=int, default=32, help='number of generator\'s filter')
parser.add_argument('--ndf', type=int, default=64, help='number of discriminator\'s filter')
parser.add_argument('--num_resnet', type=int, default=6, help='number of resnet blocks')
parser.add_argument('--num_epochs', type=int, default=200, help='number of train epochs')
parser.add_argument('--decay_epoch', type=int, default=100, help='learning rate decay')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate for generator')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate for discriminator')
parser.add_argument('--lambdaA', type=float, default=10, help='lambdaA for cycle loss')
parser.add_argument('--lambdaB', type=float, default=10, help='lambdaB for cycle loss')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')


def parse_args(is_training=False):
    if is_training:
        parser.add_argument('--data_A_dir', type=str, required=True, help='A data dir for training')
        parser.add_argument('--data_B_dir', type=str, required=True, help='B data dir for training')
        parser.add_argument('--test_data_A_dir', type=str, required=False, help='A data dir for testing')
        parser.add_argument('--test_data_B_dir', type=str, required=False, help='B data dir for testing')
        parser.add_argument('--output_dir', type=str, required=True, help='output dir')
    else:
        parser.add_argument('--model', type=str, required=True, help='Pre-trained model dir')
        parser.add_argument('--type', type=str, default='AtoB', help='Generator type: AtoB, BtoA')
        parser.add_argument('--src', type=str, required=False, help='File path of an source image')
        parser.add_argument('--out', type=str, required=False, help='File path of an output image file path which is transferred')
        parser.add_argument('--src_dir', type=str, required=False, help='A data dir you want to transfer')
        parser.add_argument('--out_dir', type=str, required=False, help='output dir ')

    return parser.parse_args()


def print_help():
    parser.print_help()
