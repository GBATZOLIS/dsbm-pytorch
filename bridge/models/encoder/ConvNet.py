import torch
import torch.nn as nn
import functools
import pytorch_lightning as pl
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import Resize
import numpy as np

class ConvNetEncoder(pl.LightningModule):
    def __init__(self, config):
        """
        Inputs:
            - num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        num_input_channels = config.data.channels
        base_channel_size = config.encoder.base_channel_size
        self.latent_dim = latent_dim = config.latent_dim
        act_fn = nn.GELU
        self.variational = config.variational
        self.time_conditional = config.encoder.time_conditional
        self.split_output = config.encoder.split_output

        if self.variational:
          out_dim = 2*latent_dim
        else:
          out_dim = latent_dim

        c_hid = base_channel_size
        
        if config.data.image_size == 28:
          self.net = nn.Sequential(
              nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1),  # 28x28
              act_fn(),
              nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
              act_fn(),
              nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 28x28 => 14x14
              act_fn(),
              nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
              act_fn(),
              nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 14x14 => 7x7
              act_fn(),
              nn.Flatten()  # Image grid to single feature vector
          )
          if self.time_conditional:
            self.linear = nn.Linear(2*7*7*c_hid+1, out_dim)
          else:
            self.linear = nn.Linear(2*7*7*c_hid, out_dim)

        elif config.data.image_size == 32:
          self.net = nn.Sequential(
              nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2), # 32x32 => 16x16
              act_fn(),
              nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
              act_fn(),
              nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 16x16 => 8x8
              act_fn(),
              nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
              act_fn(),
              nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 8x8 => 4x4
              act_fn(),
              nn.Flatten() # Image grid to single feature vector
          )
          if self.time_conditional:
            self.linear = nn.Linear(2*16*c_hid+1, out_dim)
          else:
            self.linear = nn.Linear(2*16*c_hid, out_dim)
      
        elif config.data.image_size == 64:
          self.net = nn.Sequential(
              nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2), # 64x64 => 32x32
              act_fn(),
              nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
              act_fn(),
              nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 32x32 => 16x16
              act_fn(),
              nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
              act_fn(),
              nn.Conv2d(2*c_hid, 4*c_hid, kernel_size=3, padding=1, stride=2), # 16x16 => 8x8
              act_fn(),
              nn.Conv2d(4*c_hid, 4*c_hid, kernel_size=3, padding=1),
              act_fn(),
              nn.Conv2d(4*c_hid, 4*c_hid, kernel_size=3, padding=1, stride=2), # 8x8 => 4x4
              act_fn(),
              nn.Flatten(), # Image grid to single feature vector
            )
          if self.time_conditional:
            self.linear = nn.Linear(4*16*c_hid+1, out_dim)
          else:
            self.linear = nn.Linear(4*16*c_hid, out_dim)
      
    def forward(self, x, t=None):
        flattened_img = self.net(x)
        if self.time_conditional:
          concat = torch.cat((flattened_img, t[:, None]), dim=1)
          out = self.linear(concat)
        else:
          out = self.linear(flattened_img)
        if self.variational and self.split_output:
          return out[:,:self.latent_dim], out[:,self.latent_dim:]
        else:
          return out