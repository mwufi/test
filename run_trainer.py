from __future__ import print_function

import argparse
import random

import matplotlib
import torch
import wandb

# For servers without X-windows
matplotlib.use('Agg')

from models import DCGAN, CycleGAN
from data import create_train_data
from utils import gpu_check
from configs.parser import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/dcgan.yml',
                        help="Config file to use")
    parser.add_argument('--test', action='store_true',
                        help="Whether to just load the data")
    args = parser.parse_args()
    return args


def create_model(op, device):
    model_name = op.model.name

    if model_name == 'dcgan':
        return DCGAN(op, device)
    elif model_name == 'cyclegan':
        return CycleGAN(op, device)


def main():
    args = parse_args()
    op = load_config(args.config)
    op.test = args.test
    print('Testing data....', op.test)

    # Set random seed for reproducibility
    manualSeed = op.seed
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # Load data/models on GPU?
    device = gpu_check(op)

    print('Creating data...')
    real_data_loader = create_train_data(op, device)

    if op.test:
        print('Done testing data!')
        return

    # Init wandb
    wandb.init(project='dfdf', config=op)
    print('==== Config ====', wandb.config)

    # Now we can enter the training loop!
    print('Creating models...')
    awesome = create_model(op, device)

    print('Starting training loop..')
    awesome.train(real_data_loader)


if __name__ == "__main__":
    main()
