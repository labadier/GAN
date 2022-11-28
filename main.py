#%%
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

from utils import load_dataset, plot_training
from models import Discriminator, Generator, generator_step, discriminator_step
from models import sample_from_generator, create_state

from params import parameters
import jax

Path('results').mkdir(parents=True, exist_ok=True)

trainloader, devloader = load_dataset()

key = jax.random.PRNGKey(seed=parameters.seed)
key_generator, key_discriminator, key = jax.random.split(key, 3)

discriminator_state = create_state(key_discriminator, Discriminator, next(iter(trainloader))['data'].shape)
generator_state = create_state(key_generator, Generator, (parameters.batch_size, parameters.noise_dims))

generator_input = jax.random.normal(key, (parameters.batch_size, parameters.noise_dims))  # random noise


loss_epoch = {'generator': [], 'discriminator': []}


for epoch in range(parameters.epoches):

  # itr = tqdm(enumerate(trainloader))
  itr = enumerate(trainloader)
  # itr.set_description(f'Epoch: {epoch} ')
  loss_gen = []
  los_disc = []
  print('started epoch')
  for batch, batch_data in itr:

    if len(batch_data['data']) < parameters.batch_size:
      # print('batch skiped')
      continue

    # Generate RNG keys for generator and discriminator.
    key, key_generator, key_discriminator = jax.random.split(key, 3)

    # Take a step with the generator.
    generator_state, generator_loss = generator_step(generator_state, 
        discriminator_state, key_generator)

    # Take a step with the discriminator.
    discriminator_state, discriminator_loss = discriminator_step(
        generator_state, discriminator_state, batch_data['data'], key_discriminator)
    
    loss_gen += [generator_loss.item()]
    los_disc += [discriminator_loss.item()]
  
  loss_gen = np.mean(loss_gen)
  los_disc = np.mean(los_disc)
  print(f"Epoch {epoch}: Geneator loss: {loss_gen:.3f} Discriminator Loss: {los_disc:.3f}")
  loss_epoch['generator'] += [loss_gen]
  loss_epoch['discriminator'] += [los_disc]

  if not (epoch % 10):
    sample = sample_from_generator(generator_state, generator_input)
    sample = sample.reshape((-1, 28, 28))
    fig, axes = plt.subplots(nrows=7, ncols=7, figsize=(7, 7))
    for ax, image in zip(sum(axes.tolist(), []), sample):
      ax.imshow(image, cmap='gray')
      ax.set_axis_off()
    
    fig.savefig(f"results/GAN_epoch_{epoch}.png")
  
plot_training(loss_epoch)

# %%
