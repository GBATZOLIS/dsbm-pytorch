import torch
import torch.nn as nn
import torch.nn.functional as F

class CompressorEncoder(nn.Module):
    def __init__(self, config):
        super(CompressorEncoder, self).__init__()
        self.resolution = config.encoder.resolution
        self.num_resolutions = len(config.decoder.ch_mult) - 1
        self.z_channels = config.encoder.z_channels

        assert self.z_channels == 3, 'z_channels should be set to three for compressor encoder'

        # Compute the output dimension
        self.output_dim = self.resolution // (2 ** self.num_resolutions)

    def forward(self, x):
        x_resized = F.interpolate(x, size=(self.output_dim, self.output_dim), mode='bilinear', align_corners=False)
        return x_resized
