import torchvision.datasets as dset

from .utils import rgba_loader, make_transforms, move, git_clone


class PokeSprites(dset.ImageFolder):
    """Retrieves a small dataset of Pokesprites (1k images) from Github
    """

    def __init__(self, op):
        # clone the pokemon repo if we don't have it already
        git_clone('https://github.com/PokeAPI/sprites.git', 'pokemon', op.data.clone_again)
        move('pokemon/sprites/pokemon', 'pokemon/sprites/items')

        # Call the ImageFolder constructor
        super(PokeSprites, self).__init__(root='pokemon/sprites/items',
                                          loader=rgba_loader,
                                          transform=make_transforms(op),
                                          target_transform=None)
