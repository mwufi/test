import os

from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg

Pix2Pix_Datasets = {
    'cityscapes': '99M',
    'edges2handbags': '8.0G',
    'edges2shoes': '2.0G',
    'facades': '29M',
    'maps': '239M'
}


class Pix2Pix(ImageFolder):
    """
    This does the same thing as

    ```
    sh data/download_pix2pix_data.sh facades
    ```

    If you specify "facades" as your dataset, for example:

    You'll get a Pix2Pix dataset with 606 images:

    ```
    /facades
        /train      400 images
        /test       106 images
        /val        100 images
    ```

    """

    BASE_URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets'

    def __init__(self, root, dataset_name='facades', split='train', download=True, **kwargs):
        root = os.path.expanduser(root)
        self.root = os.path.join('.', root)

        self.dataset_name = verify_str_arg(dataset_name, Pix2Pix_Datasets.keys())
        self.split = verify_str_arg(split, ['train', 'val'])

        if download:
            self.download()

        super(Pix2Pix, self).__init__(self.root, **kwargs)

    def download(self):
        if not os.path.isdir(self.root):
            url = f'{self.BASE_URL}/{self.dataset_name}.tar.gz'
            download_and_extract_archive(url, download_root='.')
            print("All done!")
        else:
            msg = ("You set download=True, but a folder '{}' already exists in "
                   "the root directory. If you want to re-download or re-extract the "
                   "archive, delete the folder.")

            print(msg.format(self.dataset_name))
