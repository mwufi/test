import torch
import torchvision.datasets as dset

from .pix2pix import Pix2Pix, Pix2Pix_Datasets
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

    elif dataset_name in Pix2Pix_Datasets.keys():
        return Pix2Pix(root=dataset_name, dataset_name=dataset_name, download=True,
                       transform=make_transforms(op))

    else:
        raise ValueError(f'{dataset_name} not supported!')


def create_train_data(op, device):
    # Create the dataset
    dataset = make_dataset(op)
    print(dataset)

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=op.data.loader.batch_size,
                                             shuffle=True, num_workers=op.data.loader.workers)

    # Make it never stop
    return infinite_data(dataloader, device)
