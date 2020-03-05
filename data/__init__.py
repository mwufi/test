import torch
import torchvision.datasets as dset
import wandb
from torchvision import utils as vutils

from . import infinite_data
from .pokemon import PokeSprites
from .utils import make_transforms, infinite_data

def make_dataset(op):
    print('Downloading data...')

    if op.dataset == 'celeba':
        op.update({'nc': 3}, allow_val_change=True)
        return dset.CelebA(root='celeba', download=True,
                           transform=make_transforms(op))

    elif op.dataset == 'fashion_mnist':
        op.update({'nc': 1}, allow_val_change=True)
        return dset.FashionMNIST(root='fashion_mnist', download=True,
                                 transform=make_transforms(op))

    elif op.dataset == 'pokemon':
        op.update({'nc': 3}, allow_val_change=True)
        return PokeSprites(op)

    else:
        raise ValueError(f'{op.dataset} not supported!')


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
        'Training images': wandb.Image(vutils.make_grid(real_batch[0].to(device)[:64],
                                                        padding=2,
                                                        normalize=True).cpu())
    })

    return infinite_data(dataloader, device)