from __future__ import print_function

import random

# For servers without X windows
import matplotlib

matplotlib.use('Agg')

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import wandb

import hparams as op
from models.dcgan import Generator, Discriminator
from utils import weights_init, parallelize

wandb.init(project='dfdf')
wandb.config.update(op)

# Set random seed for reproducibility
manualSeed = 998
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# We can use an image folder dataset the way we have it setup.
# Create the dataset
print('Downloading data...')
dataset = dset.CelebA(root=op.dataroot,
                      download=True,
                      transform=transforms.Compose([
                          transforms.Resize(op.image_size),
                          transforms.CenterCrop(op.image_size),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                      ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=op.batch_size,
                                         shuffle=True, num_workers=op.workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and op.ngpu > 0) else "cpu")

print('Using device: %s' % device)

# Plot some training images
real_batch = next(iter(dataloader))
wandb.log({
    'Training images': wandb.Image(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu())
})


def create_model(model_class, op):
    model = model_class(op)
    model = parallelize(model, op)

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    model.apply(weights_init)

    # Log the model
    if op.remote == 'wandb':
        wandb.watch(model, log_freq=op.log_freq)

    print(model)

    return model


# Create the generator
print('Creating models...')
netG = create_model(Generator, op)
netD = create_model(Discriminator, op)

# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, op.nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=op.lr, betas=(op.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=op.lr, betas=(op.beta1, 0.999))

# Lists to keep track of progress
iters = 0

print("Starting Training Loop...")


def update_D(real_cpu, label):
    """Updates D by sampling a real batch and a fake batch

    """
    netD.zero_grad()

    # Compute the size of the batch here
    b_size = real_cpu.size(0)

    output = netD(real_cpu).view(-1)

    # How much the discriminator thinks it's real (should be 1)
    D_x = output.mean().item()

    errD_real = criterion(output, label)
    errD_real.backward()

    noise = torch.randn(b_size, op.nz, 1, 1, device=device)
    fake = netG(noise)
    label.fill_(fake_label)

    # How much the discriminator thinks it's fake (should be 0)
    output = netD(fake.detach()).view(-1)
    D_G_z1 = output.mean().item()

    errD_fake = criterion(output, label)
    errD_fake.backward()

    errD = errD_real + errD_fake
    optimizerD.step()

    return errD, D_x, D_G_z1


def update_G(label):
    """Updates G by generating data and getting scores from D

    """
    netG.zero_grad()
    label.fill_(real_label)  # fake labels are real for generator cost

    # How much the discriminator thinks the generated stuff is real (should be 0)
    output = netD(fake).view(-1)
    D_G_z = output.mean().item()

    errG = criterion(output, label)
    errG.backward()

    optimizerG.step()
    return errG, D_G_z


# For each epoch
for epoch in range(op.num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)

        for _ in range(op.discriminator_updates):
            errD, D_x, D_G_z1 = update_D(real_cpu, label)

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        errG, D_G_z2 = update_G(real_label)

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, op.num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        wandb.log({
            'Generator Loss': errG.item(),
            'Disc loss': errD.item(),
            'epoch': epoch,
            'i': i,
            'D(x)': D_x,
            'D(G(z))': (D_G_z1 + D_G_z2)/2,

        })

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % op.eval_ever == 0) or ((epoch == op.num_epochs - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            wandb.log({
                'Generated images': wandb.Image(vutils.make_grid(fake, padding=2, normalize=True))
            })

        iters += 1
