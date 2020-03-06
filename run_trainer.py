from __future__ import print_function

import argparse
import random

import matplotlib
import torch
import wandb

# For servers without X-windows
matplotlib.use('Agg')

from models.dcgan import DCGAN
from data import create_train_data
from utils import gpu_check
from configs.parser import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/dcgan.yml',
                        help="Config file to use")

    # TODO: Handle argument overrides :)
    # For now, we'll just read the config file and return that
    args = parser.parse_args()
    config = load_config(args.config)
    return config


def main():
    op = parse_args()

    # Set random seed for reproducibility
    manualSeed = op.seed
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # Load data/models on GPU?
    device = gpu_check(op)

    print('Creating data...')
    real_data_loader = create_train_data(op, device)

    # Init wandb
    wandb.init(project='dfdf', config=op)
    print('==== Config ====', wandb.config)

    # Now we can enter the training loop!
    print('Creating models...')
    awesome = DCGAN(op, device)

    print('Starting training loop..')
    awesome.train(real_data_loader)


if __name__ == "__main__":
    main()
