import wandb

w = wandb.config

# whether to watch the training with an external tool
w.remote = 'wandb'
w.log_freq = 100      # How often to send model gradients and stuff

# Root directory for dataset
w.dataroot = "celeba"

# Number of workers for dataloader
w.workers = 20

# Batch size during training
w.batch_size = 128

# How often to log evaluation images
w.eval_every = 100

# Discriminator updates per generator update
w.discriminator_updates = 2

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
w.image_size = 64

# Number of channels in the training images. For color images this is 3
w.nc = 3

# Size of z latent vector (i.e. size of generator input)
w.nz = 100

# Size of feature maps in generator
w.ngf = 64

# Size of feature maps in discriminator
w.ndf = 64

# Number of training epochs
w.num_epochs = 5

# Learning rate for optimizers
w.lr = 0.0002

# Beta1 hyperparam for Adam optimizers
w.beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
w.ngpu = 1