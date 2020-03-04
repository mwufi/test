
def init(w):
    """Sets up the Wandb config
    """

    # whether to watch the training with an external tool
    w.remote = 'wandb'
    w.log_freq = 100  # How often to send model gradients and stuff

    # Dataset to use. One of: ['celeba', 'pokemon', 'fashion_mnist']
    w.dataset = 'pokemon'
    w.clone_again = False

    # Number of workers for dataloader
    w.workers = 20

    # Batch size during training
    w.batch_size = 128

    # How often to log evaluation images
    w.eval_every = 500

    # Discriminator updates per generator update
    w.discriminator_updates = 4

    # Random seed
    w.seed = 995

    # Known effects
    # 997 - Collapse @ 10k steps on the Pokemon dataset!

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
    # For pokemon, we only have 54 batches/epoch! So this will give us 25000 steps, or 4x DCGAN
    w.num_epochs = 500
    w.num_iterations = 10000

    # Learning rate for optimizers
    w.lr = 0.0002

    # Beta1 hyperparam for Adam optimizers
    w.beta1 = 0.5

    # Number of GPUs available. Use 0 for CPU mode.
    w.ngpu = 1