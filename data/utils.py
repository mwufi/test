import os
import shutil

import git
import torchvision.transforms as transforms
from PIL import Image


def rgba_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f).convert('RGBA')
        background = Image.new('RGBA', img.size, (255, 255, 255))
        alpha_composite = Image.alpha_composite(background, img)
        return alpha_composite.convert('RGB')


def make_transforms(op):
    if op.data.channels == 1:
        mean, std = (0.5,), (0.5,)
    else:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    return transforms.Compose([
        transforms.Resize(op.data.image_size),
        transforms.CenterCrop(op.data.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def infinite_data(dataloader, device):
    epochs = 0
    while True:
        epochs += 1
        for iter, (images, _) in enumerate(dataloader):
            images = images.to(device)
            yield epochs, iter, images


def move(folder1, folder2):
    """Moves one folder to another folder, skipping if it already exists"""

    destination_name = os.path.join(folder2, folder1.split('/')[-1])
    if os.path.exists(destination_name):
        return

    shutil.move(folder1, folder2)


def git_clone(remote_url, output_directory, clone_again=True):
    if clone_again or not os.path.exists(output_directory):
        ensure_empty(output_directory)
        print(f'Cloning {remote_url} into {output_directory}...')
        repo = git.Repo.init(output_directory)
        origin = repo.create_remote('origin', remote_url)
        origin.fetch()
        origin.pull(origin.refs[0].remote_head)
    else:
        print(f'{output_directory} exists! Skipping clone')

    print('Done!')


def ensure_empty(directory):
    """Removes the directory if it exists and creates a new one"""
    if os.path.isdir(directory):
        shutil.rmtree(directory)

    os.makedirs(directory)