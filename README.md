# Cycle GAN

Yet another Cycle GAN implementation in PyTorch.

The purpose of this implementation is Well-structured, reusable and easily understandable.

- [CycleGAN Paper](https://arxiv.org/pdf/1703.10593.pdf)
- [Download datasets](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/)

## Prerequisites

- System
    - Linux or macOS
    - CPU or (NVIDIA GPU + CUDA CuDNN)
        - It can run on Single CPU/GPU or Multi GPUs.
    - Python 3

- Libraries
    - PyTorch >= 0.3.0
    - Torchvision >= 0.2.0
    - scipy >= 1.0.0
    - Pillow >= 0.2.0

## Training

```bash
python train.py \
    --data_A_dir=./datasets/apple2orange/trainA \
    --data_B_dir=./datasets/apple2orange/trainB \
    --output_dir=./outputs
```

If you set `test_data_A_dir` and `test_data_B_dir` then generate A->B and B->A when end of every epoch.

```bash
python train.py \
    --data_A_dir=./datasets/apple2orange/trainA \
    --data_B_dir=./datasets/apple2orange/trainB \
    --test_data_A_dir=./datasets/apple2orange/testA \
    --test_data_B_dir=./datasets/apple2orange/testB \
    --output_dir=./outputs
```

Use `python train.py --help` to see more options.

## Transferring

For single file

```bash
python transfer.py \
    --model=./outputs/model \
    --src=./datasets/apple2orange/testA/n07740461_41.jpg \
    --out=./outputs/apple2orange.png
```

For directory

```bash
python transfer.py \
    --src_dir=./datasets/apple2orange/testA \
    --out_dir=./outputs/testA
```

Use `python transfer.py --help` to see more options.

## File structures

`network.py` and `model.py` is main implementations.

- cyclegan
    - `config.py` : Training options
    - `network.py` : The neural network architecture of Cycle GAN
    - `model.py` : Calculate loss and optimizing
    - utils
        - `data.py` : Utilities for loading data
        - `logger.py` : Utilities for logging
        - `ops.py` : Utilities for tensor operations
        - `tester.py` : Utility functions especially for testing
- `train.py` : A script for CycleGAN training
- `transfer.py` : A script for transferring with pre-trained model

# TODO

- [ ] Visualizing training progress with Visdom
- [ ] Add some nice generated images and videos :-)

## References

- https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
- https://hardikbansal.github.io/CycleGANBlog
- https://github.com/togheppi/CycleGAN
- https://github.com/znxlwm/pytorch-CycleGAN
