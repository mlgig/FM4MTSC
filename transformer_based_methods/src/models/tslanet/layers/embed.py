import lightning as L
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """
    Patch Embedding layer for time series data

    This layer:
    1. Splits input sequence into overlapping patches using 1D convolution
    2. Maps each patch to a fixed-dimension embedding vector
    """

    def __init__(self, seq_len, patch_size=8, in_chans=3, embed_dim=128):
        super().__init__()

        # Use half-overlapping patches by default
        stride = patch_size // 2

        # Calculate number of patches after convolution
        num_patches = (seq_len - patch_size) // stride + 1
        self.num_patches = num_patches

        # 1D convolution for patch embedding
        self.proj = nn.Conv1d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride
        )

    def forward(self, x):
        """
        Forward pass for patch embedding

        Args:
            x: Input tensor [B, C, N] where:
               B = batch size
               C = number of channels
               N = sequence length

        Returns:
            Embedded patches [B, num_patches, embed_dim]
        """
        # x: [B, C, N]
        x = self.proj(x)  # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]
        return x
