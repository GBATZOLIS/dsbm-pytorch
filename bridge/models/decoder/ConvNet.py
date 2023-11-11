import torch
import torch.nn as nn
import functools
import pytorch_lightning as pl
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import Resize

class ConvNetDecoder(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        num_input_channels = config.data.channels
        base_channel_size = config.decoder.base_channel_size
        latent_dim = config.latent_dim
        act_fn = nn.GELU
        c_hid = base_channel_size

        self.reshape_fn = self.get_reshape_fn(config.data.image_size)
        
        if config.data.image_size == 28:
            self.linear = nn.Sequential(
                nn.Linear(latent_dim, 2 * 7 * 7 * c_hid),
                act_fn()
            )
            self.net = nn.Sequential(
                        nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 7x7 => 14x14
                        act_fn(),
                        nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
                        act_fn(),
                        nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 14x14 => 28x28
                        act_fn(),
                        nn.Conv2d(c_hid, num_input_channels, kernel_size=3, padding=1)
                    )
        
        elif config.data.image_size == 32:
            self.linear = nn.Sequential(
                nn.Linear(latent_dim, 2 * 16 * c_hid),
                act_fn()
            )
            self.net = nn.Sequential(
                nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 4x4 => 8x8
                act_fn(),
                nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
                act_fn(),
                nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 8x8 => 16x16
                act_fn(),
                nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
                act_fn(),
                nn.ConvTranspose2d(c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 16x16 => 32x32
                act_fn(),
                nn.Conv2d(c_hid, num_input_channels, kernel_size=3, padding=1)
            )

        elif config.data.image_size == 64:
            self.linear = nn.Sequential(
                nn.Linear(latent_dim, 4 * 16 * c_hid),
                act_fn()
            )
            self.net = nn.Sequential(
                nn.ConvTranspose2d(4*c_hid, 4*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 4x4 => 8x8 
                act_fn(),
                nn.Conv2d(4*c_hid, 4*c_hid, kernel_size=3, padding=1),
                act_fn(),
                nn.ConvTranspose2d(4*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 8x8 => 16x16
                act_fn(),
                nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
                act_fn(),
                nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 16x16 => 32x32
                act_fn(),
                nn.Conv2d(c_hid, num_input_channels, kernel_size=3, padding=1)
            )

        # Keeping the 128x128 case in case you might need it later
        elif config.data.image_size == 128:
            self.linear = nn.Sequential(
              nn.Linear(latent_dim, 8*16*c_hid),
              act_fn()
          )
            self.net = nn.Sequential(
                nn.ConvTranspose2d(8*c_hid, 8*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 4x4 => 8x8
                act_fn(),
                nn.Conv2d(8*c_hid, 8*c_hid, kernel_size=3, padding=1),
                act_fn(),

                nn.ConvTranspose2d(8*c_hid, 4*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 8x8 => 16x16
                act_fn(),
                nn.Conv2d(4*c_hid, 4*c_hid, kernel_size=3, padding=1),
                act_fn(),

                nn.ConvTranspose2d(4*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 16x16 => 32x32
                act_fn(),
                nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
                act_fn(),
                nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 32x32 => 64x64
                act_fn(),
                nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
                act_fn(),

                nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2), # 64x64 => 128x128
            )
    
    def get_reshape_fn(self, image_size):
        def reshape_fn(x):
            """Reshape tensor based on image size."""
            if image_size == 28:
                return x.reshape(x.shape[0], -1, 7, 7)
            elif image_size in [32, 64, 128]:
                return x.reshape(x.shape[0], -1, 4, 4)
            else:
                raise NotImplementedError(f"Reshaping for image size {image_size} is not implemented.")
        return reshape_fn

    def forward(self, x):
        x = self.linear(x)
        x = self.reshape_fn(x)
        x = self.net(x)
        return x