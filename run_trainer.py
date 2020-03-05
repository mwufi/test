from __future__ import print_function

import random

import matplotlib
import torch
import wandb

# For servers without X-windows
matplotlib.use('Agg')

from hparams import init
from models.dcgan import DCGAN
from data import create_train_data
from utils import gpu_check

# Init wandb
wandb.init(project='dfdf')
init(wandb.config)
op = wandb.config

# Set random seed for reproducibility
manualSeed = op.seed
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Load data/models on GPU?
device = gpu_check(op)

print('Creating data...')
real_data_loader = create_train_data(op, device)

print('Creating models...')
awesome = DCGAN(op, device)

print('Starting training loop..')
awesome.train(real_data_loader)
