import torchvision.datasets as dset

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
