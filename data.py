import torchvision.datasets as dset
import torchvision.transforms as transforms
from PIL import Image

from utils import git_clone, move


def rgba_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f).convert('RGBA')
        background = Image.new('RGBA', img.size, (255, 255, 255))
        alpha_composite = Image.alpha_composite(background, img)
        return alpha_composite.convert('RGB')


def make_transforms(op):
    if op.nc == 1:
        mean, std = (0.5,), (0.5,)
    else:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    return transforms.Compose([
        transforms.Resize(op.image_size),
        transforms.CenterCrop(op.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class PokeSprites(dset.ImageFolder):
    """Retrieves a small dataset of Pokesprites (1k images) from Github

    The dataset lives here: https://github.com/PokeAPI/sprites.git

    Inside the repo, we find:
    /sprites
        /items
            /dream_world
                more-png-files**.png
            /berries
                more-png-files**.png
            /gen3
                more-png-files**.png
            /gen5
                more-png-files**.png
            /underground
                more-png-files**.png
            more-png-files**.png
        /pokemon
            101-item.png
            1.png
            *.png

    Just move pokemon into /items, so that the folders become
    /sprites/items/dream_world/sdf.png
    /sprites/items/dream_world/sdf.png
    /sprites/items/pokemon/sdf.png
    ...etc

    """

    def __init__(self, op):
        # clone the pokemon repo if we don't have it already
        git_clone('https://github.com/PokeAPI/sprites.git', 'pokemon', op.clone_again)
        move('pokemon/sprites/pokemon', 'pokemon/sprites/items')

        # Call the ImageFolder constructor
        super(PokeSprites, self).__init__(root='pokemon/sprites/items',
                                          loader=rgba_loader,
                                          transform=make_transforms(op),
                                          target_transform=None)


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


def infinite_data(dataloader, device):
    epochs = 0
    while True:
        epochs += 1
        for iter, (images, _) in enumerate(dataloader):
            images = images.to(device)
            yield epochs, iter, images
