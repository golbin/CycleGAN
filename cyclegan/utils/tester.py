"""
Utility functions especially for testing for now
"""
import os

import scipy.misc as misc

from cyclegan.utils.data import DataLoader


def get_test_output_file_path(prefix, subdir):
    path_real = os.path.join(subdir, '{0}_real.png'.format(prefix))
    path_fake = os.path.join(subdir, '{0}_fake.png'.format(prefix))
    path_recon = os.path.join(subdir, '{0}_recon.png'.format(prefix))

    return path_real, path_fake, path_recon


def prepare_test_output_dir_by_epoch(epoch, base_dir):
    subdir_A_to_B = os.path.join(base_dir, str(epoch), 'A_to_B')
    subdir_B_to_A = os.path.join(base_dir, str(epoch), 'B_to_A')

    os.makedirs(subdir_A_to_B, exist_ok=True)
    os.makedirs(subdir_B_to_A, exist_ok=True)

    return subdir_A_to_B, subdir_B_to_A


def generate_one(generator, real_data, output_dir, prefix):
    path_real, path_fake, path_recon = get_test_output_file_path(prefix, output_dir)

    fake_data, recon_data = generator(real_data, recon=True)

    misc.imsave(path_real, real_data[0].numpy().transpose(1, 2, 0))
    misc.imsave(path_fake, fake_data[0].cpu().data.numpy().transpose(1, 2, 0))
    misc.imsave(path_recon, recon_data[0].cpu().data.numpy().transpose(1, 2, 0))


def generate_testset(epoch, model, args, max=10, shuffle=True):
    subdir_A_to_B, subdir_B_to_A = prepare_test_output_dir_by_epoch(epoch, args.test_output_dir)

    test_data_loader_A = DataLoader(args.test_data_A_dir, args.input_size, args.batch_size, shuffle=shuffle)
    test_data_loader_B = DataLoader(args.test_data_B_dir, args.input_size, args.batch_size, shuffle=shuffle)

    for i, (real_A, real_B) in enumerate(zip(test_data_loader_A, test_data_loader_B)):
        generate_one(model.generate_A_to_B, real_A, subdir_A_to_B, i)
        generate_one(model.generate_B_to_A, real_B, subdir_B_to_A, i)

        if max and i >= max - 1:
            break
