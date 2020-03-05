# Toy Box Gan

Making GANS fun!

GANS implemented:
* DCGAN
* Weight clipping (for Lipschitz continuity - WGAN)

Datasets:
* FashionMNIST
* PokeSprites!
* CelebA

Writeups & results are hosted [here](demos/writeups/README.md). I'm including both what worked and what didn't work, since that's often helpful too. Feel free to open a pull request to share your experiences!

# Using Ganbox

Deep learning projects are often made of a few key components:
* Config system
* Data - preprocessing, loading. This is important especially if you want multiple models to work on the same data!
* Models - How easy it is to make different models
* Trainer - the main code that runs everything
* Debug - Tools that help you visualize and debug what's going on! For early projects like this one, I just run the trainer and watch the plots, since the iteration speed is fast enough

If you've seen better ways of organizing, please let me know! Mainly drawing on personal experience and other repos :)

## Training
```sh
python run_trainer --config configs/dcgan.yml
```

Todo:
- parameter sweeps (integrated with wandb)

## Configs
https://github.com/cdgriffith/Box

Configs are written in yml! And can be accessed in dot-notation!

Reading values:
```python
config = load('configs/dcgan.yml')
print(config.data.loader.batch_size)
```

Writing values:
```python
config = load('configs/dcgan.yml')
config.id = 'my-awesome-gan'
```