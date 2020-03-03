import wandb
from torch import nn

from models.layers import deconv, conv


class BaseModel(nn.Module):
    def __init__(self, opts):
        super(BaseModel, self).__init__()
        self.op = opts

    def make_dummy_input(self):
        pass


# Generator Code

class Generator(BaseModel):
    def __init__(self, opts):
        super(Generator, self).__init__(opts)
        nz, ngf, nc = opts.nz, opts.ngf, opts.nc
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            deconv(nz, ngf * 8, kernel_size=4, stride=1, padding=0, activation='relu'),
            deconv(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, activation='relu'),
            deconv(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, activation='relu'),
            deconv(ngf * 2, ngf * 1, kernel_size=4, stride=2, padding=1, activation='relu'),

            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(BaseModel):
    def __init__(self, opts):
        super(Discriminator, self).__init__(opts)
        ndf, nc = opts.ndf, opts.nc
        self.main = nn.Sequential(
            conv(nc, ndf, 4, 2, 1, activation='leaky_relu'),
            conv(ndf, ndf * 2, 4, 2, 1, activation='leaky_relu'),
            conv(ndf * 2, ndf * 4, 4, 2, 1, activation='leaky_relu'),
            conv(ndf * 4, ndf * 8, 4, 2, 1, activation='leaky_relu'),

            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
