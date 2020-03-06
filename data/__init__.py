import torch
import torchvision.datasets as dset
import wandb
from torchvision import utils as vutils

from .pokemon import PokeSprites
from .utils import make_transforms, infinite_data


def make_dataset(op):
    print('Downloading data...')
    dataset_name = op.data.name

    if dataset_name == 'celeba':
        return dset.CelebA(root='celeba', download=True,
                           transform=make_transforms(op))

    elif dataset_name == 'fashion_mnist':
        return dset.FashionMNIST(root='fashion_mnist', download=True,
                                 transform=make_transforms(op))

    elif dataset_name == 'pokemon':
        return PokeSprites(op)

    else:
        raise ValueError(f'{dataset_name} not supported!')


def create_train_data(op, device):
    # Create the dataset
    dataset = make_dataset(op)
    print(dataset)

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=op.data.loader.batch_size,
                                             shuffle=True, num_workers=op.data.loader.workers)

    # Plot some training images
    real_batch = next(iter(dataloader))

    wandb.log({
        'Training images': wandb.Image(vutils.make_grid(real_batch[0].to(device)[:64],
                                                        padding=2,
                                                        normalize=True).cpu())
    })

    return infinite_data(dataloader, device)
