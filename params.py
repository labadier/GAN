from jax import numpy as jnp

class parameters:

  seed = 24
  
  batch_size = 32
  learning_rate = 2e-4
  noise_dims = 64
  epoches = 100

  true_labels_template = jnp.ones((batch_size, 1), dtype=jnp.int32)
  false_labels_template= jnp.zeros((batch_size, 1), dtype=jnp.int32)