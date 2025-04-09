import pandas as pd
import torch
import torch.nn as nn
from einops import rearrange


class Attention(nn.Module):
    """
    Standard multi-head self-attention mechanism

    This block implements the attention mechanism described in
    "Attention Is All You Need" (Vaswani et al., 2017)
    """

    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.scale = emb_size**-0.5

        # Linear projections for query, key, value
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)

    def forward(self, x):
        """
        Forward pass through attention block

        Args:
            x: Input tensor of shape [batch_size, seq_len, emb_size]

        Returns:
            Attention output of shape [batch_size, seq_len, emb_size]
        """
        batch_size, seq_len, _ = x.shape
        head_dim = x.shape[-1] // self.num_heads

        # Project queries, keys, values and reshape for multi-head attention
        k = (
            self.key(x)
            .reshape(batch_size, seq_len, self.num_heads, -1)
            .permute(0, 2, 3, 1)
        )
        v = (
            self.value(x)
            .reshape(batch_size, seq_len, self.num_heads, -1)
            .transpose(1, 2)
        )
        q = (
            self.query(x)
            .reshape(batch_size, seq_len, self.num_heads, -1)
            .transpose(1, 2)
        )
        # k shape: [batch_size, num_heads, head_dim, seq_len]
        # v,q shape: [batch_size, num_heads, seq_len, head_dim]

        # Compute scaled dot-product attention
        attn = (
            torch.matmul(q, k) * self.scale
        )  # [batch_size, num_heads, seq_len, seq_len]
        attn = nn.functional.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # [batch_size, num_heads, seq_len, head_dim]
        out = out.transpose(1, 2)  # [batch_size, seq_len, num_heads, head_dim]
        out = out.reshape(batch_size, seq_len, -1)  # [batch_size, seq_len, emb_size]
        out = self.to_out(out)

        return out


class Attention_Rel_Scl(nn.Module):
    """
    Multi-head self-attention with scalar relative positional encoding

    This block extends standard attention with a learned relative
    positional bias based on the distance between tokens.
    """

    def __init__(self, emb_size, num_heads, seq_len, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.scale = emb_size**-0.5

        # Linear projections for query, key, value
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)

        # Relative positional bias table
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.seq_len - 1), num_heads)
        )

        # Create relative position indices
        coords = torch.meshgrid((torch.arange(1), torch.arange(self.seq_len)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords[1] += self.seq_len - 1
        relative_coords = rearrange(relative_coords, "c h w -> h w c")
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)

    def forward(self, x):
        """
        Forward pass through attention block with scalar relative positional encoding

        Args:
            x: Input tensor of shape [batch_size, seq_len, emb_size]

        Returns:
            Attention output of shape [batch_size, seq_len, emb_size]
        """
        batch_size, seq_len, _ = x.shape
        head_dim = x.shape[-1] // self.num_heads

        # Project queries, keys, values and reshape for multi-head attention
        k = (
            self.key(x)
            .reshape(batch_size, seq_len, self.num_heads, -1)
            .permute(0, 2, 3, 1)
        )
        v = (
            self.value(x)
            .reshape(batch_size, seq_len, self.num_heads, -1)
            .transpose(1, 2)
        )
        q = (
            self.query(x)
            .reshape(batch_size, seq_len, self.num_heads, -1)
            .transpose(1, 2)
        )

        # Compute scaled dot-product attention
        attn = (
            torch.matmul(q, k) * self.scale
        )  # [batch_size, num_heads, seq_len, seq_len]

        # Apply relative positional bias
        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, self.num_heads)
        )
        relative_bias = rearrange(
            relative_bias, "(h w) c -> 1 c h w", h=1 * self.seq_len, w=1 * self.seq_len
        )
        attn = attn + relative_bias

        # Normalize attention weights
        attn = nn.functional.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # [batch_size, num_heads, seq_len, head_dim]
        out = out.transpose(1, 2)  # [batch_size, seq_len, num_heads, head_dim]
        out = out.reshape(batch_size, seq_len, -1)  # [batch_size, seq_len, emb_size]
        out = self.to_out(out)

        return out


class Attention_Rel_Vec(nn.Module):
    """
    Multi-head self-attention with vector-based relative positional encoding

    Implements the relative positional encoding from
    "Self-Attention with Relative Position Representations" (Shaw et al., 2018)
    """

    def __init__(self, emb_size, num_heads, seq_len, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.scale = emb_size**-0.5

        # Linear projections for query, key, value
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)

        # Relative positional embedding
        self.Er = nn.Parameter(torch.randn(self.seq_len, int(emb_size / num_heads)))

        # Causal attention mask (for autoregressive modeling)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(self.seq_len, self.seq_len))
            .unsqueeze(0)
            .unsqueeze(0),
        )

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)

    def forward(self, x):
        """
        Forward pass through attention block with vector-based relative positional encoding

        Args:
            x: Input tensor of shape [batch_size, seq_len, emb_size]

        Returns:
            Attention output of shape [batch_size, seq_len, emb_size]
        """
        batch_size, seq_len, _ = x.shape
        head_dim = x.shape[-1] // self.num_heads

        # Project queries, keys, values and reshape for multi-head attention
        k = (
            self.key(x)
            .reshape(batch_size, seq_len, self.num_heads, -1)
            .permute(0, 2, 3, 1)
        )
        v = (
            self.value(x)
            .reshape(batch_size, seq_len, self.num_heads, -1)
            .transpose(1, 2)
        )
        q = (
            self.query(x)
            .reshape(batch_size, seq_len, self.num_heads, -1)
            .transpose(1, 2)
        )

        # Compute relative positional encoding attention
        QEr = torch.matmul(
            q, self.Er.transpose(0, 1)
        )  # [batch_size, num_heads, seq_len, seq_len]
        Srel = self.skew(QEr)  # [batch_size, num_heads, seq_len, seq_len]

        # Compute content-based attention and add positional component
        attn = torch.matmul(q, k)  # [batch_size, num_heads, seq_len, seq_len]
        attn = (attn + Srel) * self.scale

        # Apply attention mask if doing causal attention
        # attn = attn.masked_fill(self.mask[:,:,:seq_len,:seq_len] == 0, float('-inf'))

        # Normalize attention weights
        attn = nn.functional.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # [batch_size, num_heads, seq_len, head_dim]
        out = out.transpose(1, 2)  # [batch_size, seq_len, num_heads, head_dim]
        out = out.reshape(batch_size, seq_len, -1)  # [batch_size, seq_len, emb_size]
        out = self.to_out(out)

        return out

    def skew(self, QEr):
        """
        Skew the relative position encoding for proper alignment

        This implements the relative shift operation described in
        "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"

        Args:
            QEr: Tensor of shape [batch_size, num_heads, seq_len, seq_len]

        Returns:
            Skewed tensor of shape [batch_size, num_heads, seq_len, seq_len]
        """
        # QEr shape: [batch_size, num_heads, seq_len, seq_len]
        padded = nn.functional.pad(
            QEr, (1, 0)
        )  # [batch_size, num_heads, seq_len, 1 + seq_len]
        batch_size, num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(
            batch_size, num_heads, num_cols, num_rows
        )  # [batch_size, num_heads, 1 + seq_len, seq_len]
        Srel = reshaped[:, :, 1:, :]  # [batch_size, num_heads, seq_len, seq_len]
        return Srel
