import os

import torch
import wandb
from torch import nn
from torch import optim
from torch.autograd import Variable


def to_var(x):
    """Converts numpy to variable"""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def to_data(x):
    """Converts variable to numpy"""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()


def create_dir(directory):
    """Creates a directory if it does not already exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def gpu_check(op):
    device = torch.device("cuda:0" if (torch.cuda.is_available() and op.ngpu > 0) else "cpu")
    print('Using device: %s' % device)

    return device


def parallelize(model, op):
    number_of_gpus = op.ngpu

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and number_of_gpus > 0) else "cpu")
    model = model.to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (number_of_gpus > 1):
        return nn.DataParallel(model, list(range(number_of_gpus)))

    return model


def create_optimizer(model, op):
    return optim.Adam(model.parameters(), lr=op.lr, betas=(op.beta1, 0.999))


def create_model(model_class, op):
    model = model_class(op)
    model = parallelize(model, op)

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    model.apply(weights_init)

    # Log the model
    if op.remote == 'wandb':
        wandb.watch(model, log_freq=op.log_freq)

    print(model)

    return model
