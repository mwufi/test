import torch
import torchvision.utils as vutils
import wandb
from torch import nn

from models.layers import deconv, conv
from utils import create_optimizer, create_model

class CycleGAN:
    def __init__(self, op, device):
        self.device = device
        self.op = op


    def train(self, data_loader):
        print('Here is your cyclegan!')

