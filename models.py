
#%%
from typing import Any
from functools import partial
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

from params import parameters
from utils import load_dataset

from jax.nn.initializers import normal as normal_init
from flax import linen as nn
from jax import numpy as jnp
import jax
import optax

from flax.training import train_state as flax_train_state

class TrainState(flax_train_state.TrainState):
  batch_stats: Any

class Generator(nn.Module):
  features: int = 64
  dtype: type = jnp.float32

  @nn.compact
  def __call__(self, z: jnp.ndarray, batch_size = parameters.batch_size, train: bool = True):

    conv_transpose = partial(nn.ConvTranspose, padding='VALID',
                             kernel_init=normal_init(0.02), dtype=self.dtype)

    batch_norm = partial(nn.BatchNorm, use_running_average = not train, axis=-1, 
                         scale_init=normal_init(0.02), dtype=self.dtype)

    z = z.reshape((batch_size, 1, 1, parameters.noise_dims))

    x = conv_transpose(self.features*4, kernel_size=[3, 3], strides=[2, 2])(z)

    x = batch_norm()(x)
    x = nn.relu(x)
    x = conv_transpose(self.features*4, kernel_size=[4, 4], strides=[1, 1])(x)
    x = batch_norm()(x)
    x = nn.relu(x)
    x = conv_transpose(self.features*2, kernel_size=[3, 3], strides=[2, 2])(x)
    x = batch_norm()(x)
    x = nn.relu(x)
    x = conv_transpose(1, [4, 4], [2, 2])(x)
    x = jnp.tanh(x)
    # print('Generator Compiled')
    return x


class Discriminator(nn.Module):
  features: int = 64
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray, batch_size = parameters.batch_size, train: bool = True):

    conv = partial(nn.Conv, kernel_size=[4, 4], strides=[2, 2], padding='VALID',
                   kernel_init=normal_init(0.02), dtype=self.dtype)

    batch_norm = partial(nn.BatchNorm, use_running_average=not train, axis=-1,
                         scale_init=normal_init(0.02), dtype=self.dtype)

    x = conv(self.features)(x)
    x = batch_norm()(x)
    x = nn.leaky_relu(x, 0.2)
    x = conv(self.features*2)(x)
    x = batch_norm()(x)
    x = nn.leaky_relu(x, 0.2)
    x = conv(1)(x)
    x = x.reshape((batch_size, -1))
    # print('Discriminator Compiled')
    return x

def create_state(rng, model_cls, input_shape): 

  """Create the training state given a model class. """ 

  model = model_cls()   

  tx = optax.adam(parameters.learning_rate, b1=0.5, b2=0.999) 
  variables = model.init(rng, jnp.ones(input_shape))   

  state = TrainState.create(apply_fn=model.apply, tx=tx, 
      params=variables['params'], batch_stats=variables['batch_stats'])
  
  return state

@jax.jit
def sample_from_generator(generator_state, input_noise):
  
  """Sample from the generator in evaluation mode."""
  
  generated_data = generator_state.apply_fn(
      {'params': generator_state.params,
       'batch_stats': generator_state.batch_stats},
      input_noise, train=False, mutable=False)

  return generated_data

@jax.jit
def generator_step(generator_state, discriminator_state, key):

  r"""The generator is updated by generating data and letting the discriminator
  critique it. It's loss goes down if the discriminator wrongly predicts it to
  to be real data."""

  input_noise = jax.random.normal(key, (parameters.batch_size, parameters.noise_dims))

  def loss_fn(params):

    generated_data, mutables = generator_state.apply_fn(
        {'params': params, 'batch_stats': generator_state.batch_stats},
        input_noise, mutable=['batch_stats']) 
    
    logits,_  = discriminator_state.apply_fn(
        {'params': discriminator_state.params, 
         'batch_stats': discriminator_state.batch_stats},
         generated_data, mutable=['batch_stats'])
    
    loss = -jnp.mean(jnp.log(nn.sigmoid(logits)))
    return loss, mutables
  
  # Generate data with the Generator, critique it with the Discriminator.
  (loss, mutables), grads = jax.value_and_grad(loss_fn, has_aux=True)(generator_state.params)

  # Update the Generator through gradient descent.
  new_generator_state = generator_state.apply_gradients(
      grads=grads, batch_stats=mutables['batch_stats'])
  
  return new_generator_state, loss

@jax.jit
def discriminator_step(generator_state, discriminator_state, real_data, key):
  
  r"""The discriminator is updated by critiquing both real and generated data,
  It's loss goes down as it predicts correctly if images are real or generated.
  """
  input_noise = jax.random.normal(key, (len(real_data), parameters.noise_dims))

  generated_data, _ = generator_state.apply_fn(
        {'params': generator_state.params, 
         'batch_stats': generator_state.batch_stats},
         input_noise, len(real_data), mutable=['batch_stats'])
  
  def loss_fn(params):

    logits_real, mutables = discriminator_state.apply_fn(
        {'params': params, 'batch_stats': discriminator_state.batch_stats},
        real_data, len(real_data), mutable=['batch_stats'])
        
    logits_generated, mutables = discriminator_state.apply_fn(
        {'params': params, 'batch_stats': mutables['batch_stats']},
        generated_data, len(real_data), mutable=['batch_stats'])
    

    real_loss = optax.sigmoid_binary_cross_entropy(logits_real, parameters.true_labels_template[:len(real_data)]).mean()
    generated_loss = optax.sigmoid_binary_cross_entropy(logits_generated, parameters.false_labels_template[:len(real_data)]).mean()
    
    loss = (real_loss + generated_loss) / 2
    # print('Discriminator Steper Compiled!!')

    return loss, mutables

  # Critique real and generated data with the Discriminator.
  (loss, mutables), grads = jax.value_and_grad(loss_fn, has_aux=True)(discriminator_state.params)

  new_discriminator_state = discriminator_state.apply_gradients(
      grads=grads, batch_stats=mutables['batch_stats'])
  
  return new_discriminator_state, loss


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
    itr = enumerate(devloader)
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
    print(f"Geneator loss: {loss_gen:.3f} Discriminator Loss: {los_disc:.3f}")
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

# %%
