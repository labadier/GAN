#%%
import numpy as np
from torch.utils.data import  DataLoader
import torchvision.datasets as datasets
from matplotlib import pyplot as plt
from torchvision import transforms, datasets
from params import parameters 
from jax import numpy as jnp
import torch

def imshow(img):

    img = img * 78.567 + 33.31
    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.show()

def load_dataset():

  transform = transforms.Compose(
      [transforms.ToTensor(),
      transforms.Lambda(lambda x: x.permute(1, 2, 0)),
      transforms.Normalize(33.31, 78.567)])

  train_loader = DataLoader(datasets.MNIST(root='./data', train=True, download=True, transform=transform),
                                            batch_size=parameters.batch_size,
                                            collate_fn=custom_collate_fn,
                                            shuffle=True)

  dev_loader = DataLoader(datasets.MNIST(root='./data', train=False, download=True, transform=transform),
                                          batch_size=parameters.batch_size>>1,
                                          collate_fn=custom_collate_fn,
                                          shuffle=False)
  
  return train_loader, dev_loader

def custom_collate_fn(batch):

  return {'data': jnp.stack(([jnp.array(example[0]) for example in batch]))}
  

