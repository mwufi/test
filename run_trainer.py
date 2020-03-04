from __future__ import print_function

import random

# For servers without X windows
import matplotlib

matplotlib.use('Agg')

import torch
import torchvision.utils as vutils
import wandb

from hparams import init
from models.dcgan import DCGAN
from data import make_dataset
from utils import gpu_check

wandb.init(project='dfdf')
init(wandb.config)
op = wandb.config

# Set random seed for reproducibility
manualSeed = op.seed
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


def infinite_data(dataloader, device):
    epochs = 0
    while True:
        epochs += 1
        for iter, (images, _) in enumerate(dataloader):
            images = images.to(device)
            yield epochs, iter, images


def create_train_data(op, device):
    # Create the dataset
    dataset = make_dataset(op)
    print(dataset)

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=op.batch_size,
                                             shuffle=True, num_workers=op.workers)

    # Plot some training images
    real_batch = next(iter(dataloader))
    wandb.log({
        'Training images': wandb.Image(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu())
    })

    return infinite_data(dataloader, device)


device = gpu_check(op)

print('Creating models...')
awesome = DCGAN(op, device)

print('Creating data...')
real_data_loader = create_train_data(op, device)

print('Starting training loop..')
awesome.train(real_data_loader)
