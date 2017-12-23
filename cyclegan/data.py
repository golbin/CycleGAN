"""
Utilities for loading training data
"""
import os
import random

from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms

import cyclegan.ops as ops


class Dataset(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(Dataset, self).__init__()

        self.root_path = data_dir
        self.filenames = [x for x in sorted(os.listdir(data_dir))]

        self.transform = transform

    def __getitem__(self, index):
        filepath = os.path.join(self.root_path, self.filenames[index])
        img = Image.open(filepath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.filenames)


class DataLoader(data.DataLoader):
    def __init__(self, data_dir, input_size, batch_size, shuffle=False):
        transform = transforms.Compose([
                        transforms.Resize(input_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                    ])

        dataset = Dataset(data_dir, transform)

        super(DataLoader, self).__init__(dataset, batch_size, shuffle)


class ImagePool:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.images = []

    def random_swap(self, image):
        swap = image

        if random.uniform(0, 1) > 0.5:
            random_id = random.randint(0, self.pool_size - 1)
            old = self.images[random_id].clone()
            self.images[random_id] = image
            swap = old

        return swap

    def query(self, images):
        return_images = []

        for image in images.data:
            image = torch.unsqueeze(image, 0)

            if len(self.images) < self.pool_size:
                self.images.append(image)
                return_images.append(image)
            else:
                return_image = self.random_swap(image)
                return_images.append(return_image)

        return ops.variable(torch.cat(return_images, 0))


def load_file(filepath, input_size):
    transform = transforms.Compose([
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                ])

    img = Image.open(filepath).convert('RGB')

    img = transform(img)
    img = torch.unsqueeze(img, 0)

    return img
