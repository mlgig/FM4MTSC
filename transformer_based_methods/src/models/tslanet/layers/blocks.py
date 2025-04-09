import lightning as L
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_


class ICB(nn.Module):
    """
    Inverted ConvFFN Block for spatial feature extraction

    This block consists of:
    1. A pointwise (1x1) convolution to expand channel dimension
    2. A depthwise (3x3) convolution for local feature extraction
    3. A pointwise (1x1) convolution to project back to original dimension
    """

    def __init__(self, in_features, hidden_features, drop=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, padding=1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        # x: [B, N, C]
        x = x.transpose(1, 2)  # [B, C, N]

        x1 = self.conv1(x)
        x1 = self.act(x1)
        x1 = self.drop(x1)

        x2 = self.conv2(x)
        x2 = self.act(x2)
        x2 = self.drop(x2)

        # Element-wise multiplication for attention-like gating
        out = x1 * x2
        x = self.conv3(out)

        return x.transpose(1, 2)  # Return to [B, N, C]


class Adaptive_Spectral_Block(nn.Module):
    """
    Adaptive Spectral Block for frequency domain processing

    This block applies:
    1. Fast Fourier Transform (FFT) to time series
    2. Learnable complex weighting in frequency domain
    3. Adaptive high-frequency filtering mechanism
    4. Inverse FFT to transform back to time domain
    """

    def __init__(self, dim, adaptive_filter=True):
        super().__init__()
        self.adaptive_filter = adaptive_filter
        self.dim = dim

        # Initialize learnable complex weights
        self.complex_weight = nn.Parameter(
            torch.randn(1, dim, 2, dtype=torch.float32) * 0.02
        )
        self.complex_weight_high = nn.Parameter(
            torch.randn(1, dim, 2, dtype=torch.float32) * 0.02
        )

        # Adaptive threshold for frequency masking
        self.threshold_param = nn.Parameter(torch.rand(1))

        # Initialize weights
        trunc_normal_(self.complex_weight, std=0.02)
        trunc_normal_(self.complex_weight_high, std=0.02)

    def create_adaptive_high_freq_mask(self, x_fft):
        """
        Create an adaptive mask for high-frequency components

        Args:
            x_fft: Frequency domain representation of input

        Returns:
            adaptive_mask: Mask to apply to high-frequency components
        """
        B, N, _ = x_fft.shape

        # Calculate energy in the frequency domain
        energy = torch.abs(x_fft).pow(2).mean(dim=-1)  # [B, N]

        # Calculate median energy for normalization
        median_energy = energy.median(dim=1, keepdim=True)[0]

        # Normalize energy
        normalized_energy = energy / (median_energy + 1e-6)

        # Create mask based on threshold parameter
        adaptive_mask = (
            (normalized_energy > self.threshold_param).float() - self.threshold_param
        ).detach() + self.threshold_param

        adaptive_mask = adaptive_mask.unsqueeze(-1)  # [B, N, 1]
        return adaptive_mask

    def forward(self, x):
        """
        Forward pass through Adaptive Spectral Block

        Args:
            x: Input tensor [B, N, C]

        Returns:
            Processed tensor with enhanced frequency representation
        """
        B, N, C = x.shape
        dtype = x.dtype

        # Convert to float32 for FFT operations
        x = x.to(torch.float32)

        # Transpose for 1D FFT over sequence dimension
        x = x.transpose(1, 2)  # [B, C, N]

        # Apply FFT
        x_fft = torch.fft.rfft(x, dim=2, norm="ortho")

        # Create complex weight from real and imaginary parts
        weight = torch.view_as_complex(self.complex_weight)

        # Apply complex weighting
        x_weighted = x_fft * weight.unsqueeze(2)

        # Apply adaptive filtering if enabled
        if self.adaptive_filter:
            # Create mask for high frequency components
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)

            # Apply mask
            x_masked = x_fft * freq_mask

            # Apply high-frequency specific weighting
            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted_high = x_masked * weight_high.unsqueeze(2)

            # Combine weighted representations
            x_weighted = x_weighted + x_weighted_high

        # Inverse FFT to return to time domain
        x = torch.fft.irfft(x_weighted, n=N, dim=2, norm="ortho")

        # Return to original dtype and shape
        x = x.to(dtype)
        x = x.transpose(1, 2)  # [B, N, C]

        return x


class TSLANet_layer(nn.Module):
    """
    TSLANet layer combining Adaptive Spectral Block and ICB

    This layer applies:
    1. Layer normalization and ASB for frequency domain processing
    2. Layer normalization and ICB for spatial feature extraction
    3. Residual connections and dropout
    """

    def __init__(
        self,
        dim,
        mlp_ratio=3.0,
        drop=0.0,
        drop_path=0.0,
        use_asb=True,
        use_icb=True,
        adaptive_filter=True,
    ):
        super().__init__()
        self.use_asb = use_asb
        self.use_icb = use_icb

        # Normalization layers
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # ASB for frequency domain processing
        if use_asb:
            self.asb = Adaptive_Spectral_Block(dim, adaptive_filter=adaptive_filter)

        # ICB for spatial feature extraction
        if use_icb:
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.icb = ICB(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        # Dropout path for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        """
        Forward pass through TSLANet layer

        Args:
            x: Input tensor [B, N, C]

        Returns:
            Processed tensor with enriched representations
        """
        # Apply ASB if enabled
        if self.use_asb:
            x = x + self.drop_path(self.asb(self.norm1(x)))

        # Apply ICB if enabled
        if self.use_icb:
            x = x + self.drop_path(self.icb(self.norm2(x)))

        return x
