import os
import shutil

import git
import torch
from torch import nn
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


def ensure_empty(directory):
    """Removes the directory if it exists and creates a new one"""
    if os.path.isdir(directory):
        shutil.rmtree(directory)

    os.makedirs(directory)


def git_clone(remote_url, output_directory):
    ensure_empty(output_directory)

    print(f'Cloning {remote_url} into {output_directory}...')
    repo = git.Repo.init(output_directory)
    origin = repo.create_remote('origin', remote_url)
    origin.fetch()
    origin.pull(origin.refs[0].remote_head)

    print('Done!')


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def parallelize(model, op):
    number_of_gpus = op.ngpu

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and number_of_gpus > 0) else "cpu")
    model = model.to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (number_of_gpus > 1):
        return nn.DataParallel(model, list(range(number_of_gpus)))

    return model
