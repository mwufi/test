import torch
import torchvision.utils as vutils
import wandb
from torch import nn

from models.layers import deconv, conv
from utils import create_model, create_optimizer


class BaseModel(nn.Module):
    def __init__(self, opts):
        super(BaseModel, self).__init__()
        self.op = opts

    def make_dummy_input(self):
        pass


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


class DCGAN:
    def __init__(self, op, device):
        self.G = create_model(Generator, op)
        self.D = create_model(Discriminator, op)

        self.D_optimizer = create_optimizer(self.D, op)
        self.G_optimizer = create_optimizer(self.G, op)

        self.loss_fn = nn.BCELoss()

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        self.fixed_noise = torch.randn(64, op.nz, 1, 1, device=self.device)

        self.op = op
        self.device = device

    def generate_noise(self, batch_size):
        return torch.randn(batch_size, self.op.nz, 1, 1, device=self.device)

    def forward_G(self, z, gradients=True):
        outputs = self.G(z)
        if not gradients:
            return outputs.detach()
        return outputs

    def generate(self, batch_size, gradients=True):
        z = self.generate_noise(batch_size=batch_size)
        return self.forward_G(z, gradients)

    def update_D(self, real_images, generated_images):
        batch_size = real_images.size(0)
        ones = torch.full((batch_size,), 1, device=self.device)
        zeros = torch.full((batch_size,), 1, device=self.device)

        self.D.zero_grad()
        real_scores = self.D(real_images).view(-1)
        fake_scores = self.D(generated_images).view(-1)

        update = self.loss_fn(real_scores, ones) + self.loss_fn(fake_scores, zeros)
        update.backward()
        self.D_optimizer.step()

        return {
            'real': real_scores.mean().item(),
            'fake': fake_scores.mean().item(),
            'loss': update
        }

    def update_G(self):
        batch_size = self.op.batch_size

        generated_images = self.generate(batch_size)
        ones = torch.full((batch_size,), 1, device=self.device)

        self.G.zero_grad()
        fake_scores = self.D(generated_images).view(-1)

        update = self.loss_fn(fake_scores, ones)
        update.backward()
        self.G_optimizer.step()

        return {
            'fake': fake_scores.mean().item(),
            'loss': update
        }

    def train(self, train_loader):
        for i in range(self.op.num_iterations):
            for _ in range(self.op.discriminator_updates):
                epoch, i, real_images = next(train_loader)
                batch_size = real_images.size(0)

                generated_images = self.forward_G(batch_size=batch_size, gradients=False)
                d = self.update_D(real_images, generated_images)

            g = self.update_G()

            wandb.log({
                'Generator loss': g['loss'],
                'Disc loss': d['loss'],
                'D(x)': d['real'],
                'D(G(x))': d['fake'],
                'i': i,
                'epoch': epoch
            })

            # Check how the generator is doing by saving G's output on fixed_noise
            if (i % self.op.eval_every == 0) or (i == self.op.num_iterations - 1):
                with torch.no_grad():
                    fake = self.G(self.fixed_noise).detach().cpu()
                wandb.log({
                    'Generated images': wandb.Image(vutils.make_grid(fake, padding=2, normalize=True))
                })
