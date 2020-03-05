
# Made with GANBox

Most of it is using Weights and Biases!

Confirmed that the following train well:
* FashionMNIST on anything
* Pokesprites... ish

These were as good as I got with PokeSprites. For another example, try [Lilian Weng's experiment on Lil'Log](https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html#example-create-new-pokemons), where a DCGAN was trained for 49 epochs with a batch size of 64 and 900 images, with 5 discriminator batches for every generator update. That would roughly be equivalent to 300 images per update, or 3 iterations per batch, or 150 iterations total! Which is to say, pretty fast!

|Interesting things| Docs
|---|---
|![image](pics/pokemonOneBatch.png)  |[Pokemon One Batch](pokemon-one-batch.md)