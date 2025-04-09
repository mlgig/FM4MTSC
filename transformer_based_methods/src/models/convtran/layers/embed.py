import math

import pandas as pd
import torch
import torch.nn as nn


class tAPE(nn.Module):
    """
    Transformer Absolute Positional Encoding with timing scale adjustment

    This variant of positional encoding is scaled by (d_model/max_len) factor
    to provide better handling of long sequences.

    Args:
        d_model: dimension of the model
        dropout: dropout probability
        max_len: maximum sequence length
        scale_factor: scaling factor for position encodings
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(tAPE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply scaling factor to the position-dimension product
        pe[:, 0::2] = torch.sin((position * div_term) * (d_model / max_len))
        pe[:, 1::2] = torch.cos((position * div_term) * (d_model / max_len))
        pe = scale_factor * pe.unsqueeze(0)
        self.register_buffer(
            "pe", pe
        )  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        """
        Add positional encoding to input

        Args:
            x: input tensor of shape [batch_size, seq_len, embed_dim]

        Returns:
            Tensor with positional encoding added [batch_size, seq_len, embed_dim]
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class AbsolutePositionalEncoding(nn.Module):
    """
    Standard sinusoidal position encoding as introduced in
    "Attention Is All You Need" (Vaswani et al., 2017)

    Args:
        d_model: dimension of the model
        dropout: dropout probability
        max_len: maximum sequence length
        scale_factor: scaling factor for position encodings
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(AbsolutePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0)
        self.register_buffer(
            "pe", pe
        )  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        """
        Add positional encoding to input

        Args:
            x: input tensor of shape [batch_size, seq_len, embed_dim]

        Returns:
            Tensor with positional encoding added [batch_size, seq_len, embed_dim]
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding

    Instead of using fixed sinusoidal patterns, this module
    learns the optimal positional embedding for each position.

    Args:
        d_model: dimension of the model
        dropout: dropout probability
        max_len: maximum sequence length
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(
            torch.empty(max_len, d_model)
        )  # requires_grad automatically set to True
        # Initialize with small random values
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        """
        Add learnable positional encoding to input

        Args:
            x: input tensor of shape [batch_size, seq_len, embed_dim]

        Returns:
            Tensor with positional encoding added [batch_size, seq_len, embed_dim]
        """
        x = x + self.pe[: x.size(1), :]
        return self.dropout(x)
