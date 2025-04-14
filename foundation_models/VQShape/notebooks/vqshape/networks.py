import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


class UpSampleBlock(nn.Module):
    def __init__(self, channels, kernal_size=3, scale_factor=2.0):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv1d(channels, channels, kernal_size, 1, padding='same')

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor)
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            nn.GroupNorm(16, in_channels),
            nn.GELU(),
            nn.Conv1d(in_channels, out_channels, kernel_size, 1, padding='same'),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding='same')
        )

        if in_channels != out_channels:
            self.channel_up = nn.Conv1d(in_channels, out_channels, 1, 1)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            return self.channel_up(x) + self.block(x)
        else:
            return x + self.block(x)


class ShapeDecoder(nn.Module):
    def __init__(self, dim_embedding, dim_output, out_kernel_size=3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim_embedding, dim_output*8),
            Rearrange('B L (C T) -> (B L) C T', C=128),
            ResidualBlock(128, 128),
            nn.GELU(),
            UpSampleBlock(128, scale_factor=4),
            ResidualBlock(128, 64),
            nn.GELU(),
            UpSampleBlock(64, scale_factor=4),
            nn.GELU(),
            nn.Conv1d(64, 1, out_kernel_size, 1, padding='same')
        )
    def forward(self, x):
        x = rearrange(self.layers(x), '(B L) C T -> B L C T', B=x.shape[0]).squeeze(2)
        x_mean = x.mean(dim=-1, keepdim=True)
        x_std = x.std(dim=-1, keepdim=True)
        return (x - x_mean) / (x_std + 1e-8), x_mean, x_std
