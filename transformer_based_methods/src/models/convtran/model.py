import torch
from torch import nn

from .layers.blocks import Attention, Attention_Rel_Scl, Attention_Rel_Vec
from .layers.embed import AbsolutePositionalEncoding, LearnablePositionalEncoding, tAPE


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ConvTran(nn.Module):
    """
    ConvTran model: Convolutional Transformer for time series classification

    This model combines:
    - Convolutional embedding
    - Optional positional encoding (fixed and/or relative)
    - Multi-headed self-attention
    - Feed-forward networks
    """

    def __init__(
        self,
        seq_len,
        num_classes,
        num_channels=3,
        embed_dim=128,
        num_heads=8,
        dim_ff=256,
        dropout_rate=0.1,
        fix_pos_encode="tAPE",
        rel_pos_encode="None",
    ):
        super().__init__()
        self.fix_pos_encode = fix_pos_encode
        self.rel_pos_encode = rel_pos_encode

        # Embedding Layer with 2D convolutions
        self.embed_layer = nn.Sequential(
            nn.Conv2d(1, embed_dim * 4, kernel_size=[1, 8], padding="same"),
            nn.BatchNorm2d(embed_dim * 4),
            nn.GELU(),
        )

        self.embed_layer2 = nn.Sequential(
            nn.Conv2d(
                embed_dim * 4, embed_dim, kernel_size=[num_channels, 1], padding="valid"
            ),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        # Positional Encoding
        if self.fix_pos_encode == "tAPE":
            self.fix_position = tAPE(embed_dim, dropout=dropout_rate, max_len=seq_len)
        elif self.fix_pos_encode == "Sin":
            self.fix_position = AbsolutePositionalEncoding(
                embed_dim, dropout=dropout_rate, max_len=seq_len
            )
        elif fix_pos_encode == "Learn":
            self.fix_position = LearnablePositionalEncoding(
                embed_dim, dropout=dropout_rate, max_len=seq_len
            )

        # Attention mechanism
        if self.rel_pos_encode == "eRPE":
            self.attention_layer = Attention_Rel_Scl(
                embed_dim, num_heads, seq_len, dropout_rate
            )
        elif self.rel_pos_encode == "Vector":
            self.attention_layer = Attention_Rel_Vec(
                embed_dim, num_heads, seq_len, dropout_rate
            )
        else:
            self.attention_layer = Attention(embed_dim, num_heads, dropout_rate)

        # Layer normalization and feed-forward
        self.layer_norm1 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.layer_norm2 = nn.LayerNorm(embed_dim, eps=1e-5)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_ff, embed_dim),
            nn.Dropout(dropout_rate),
        )

        # Classification head
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Input shape: [batch_size, channels, seq_len]
        x = x.unsqueeze(1)  # [batch_size, 1, channels, seq_len]

        # Apply embedding layers
        x_src = self.embed_layer(x)
        x_src = self.embed_layer2(x_src).squeeze(2)  # [batch_size, embed_dim, seq_len]
        x_src = x_src.permute(0, 2, 1)  # [batch_size, seq_len, embed_dim]

        # Apply positional encoding if specified
        if self.fix_pos_encode != "None":
            x_src_pos = self.fix_position(x_src)
            att = x_src + self.attention_layer(x_src_pos)
        else:
            att = x_src + self.attention_layer(x_src)

        # Apply normalization and feed-forward
        att = self.layer_norm1(att)
        out = att + self.feed_forward(att)
        out = self.layer_norm2(out)

        # Apply classification head
        out = out.permute(0, 2, 1)  # [batch_size, embed_dim, seq_len]
        out = self.gap(out)  # [batch_size, embed_dim, 1]
        out = self.flatten(out)  # [batch_size, embed_dim]
        out = self.out(out)  # [batch_size, num_classes]

        return out


class CasualConvTran(nn.Module):
    """
    CasualConvTran: A variant of ConvTran with causal convolutions
    This ensures that the model respects the temporal ordering of the input sequence
    """

    def __init__(
        self,
        seq_len,
        num_classes,
        num_channels=3,
        embed_dim=128,
        num_heads=8,
        dim_ff=256,
        dropout_rate=0.1,
        fix_pos_encode="tAPE",
        rel_pos_encode="None",
    ):
        super().__init__()
        self.fix_pos_encode = fix_pos_encode
        self.rel_pos_encode = rel_pos_encode

        # Replace standard convolutional embedding with causal convolutions
        self.causal_conv1 = nn.Sequential(
            CausalConv1d(num_channels, embed_dim, kernel_size=8, stride=2, dilation=1),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )

        self.causal_conv2 = nn.Sequential(
            CausalConv1d(embed_dim, embed_dim, kernel_size=5, stride=2, dilation=2),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )

        self.causal_conv3 = nn.Sequential(
            CausalConv1d(embed_dim, embed_dim, kernel_size=3, stride=2, dilation=2),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )

        # Positional Encoding
        if self.fix_pos_encode == "tAPE":
            self.fix_position = tAPE(embed_dim, dropout=dropout_rate, max_len=seq_len)
        elif self.fix_pos_encode == "Sin":
            self.fix_position = AbsolutePositionalEncoding(
                embed_dim, dropout=dropout_rate, max_len=seq_len
            )
        elif fix_pos_encode == "Learn":
            self.fix_position = LearnablePositionalEncoding(
                embed_dim, dropout=dropout_rate, max_len=seq_len
            )

        # Attention mechanism
        if self.rel_pos_encode == "eRPE":
            self.attention_layer = Attention_Rel_Scl(
                embed_dim, num_heads, seq_len, dropout_rate
            )
        elif self.rel_pos_encode == "Vector":
            self.attention_layer = Attention_Rel_Vec(
                embed_dim, num_heads, seq_len, dropout_rate
            )
        else:
            self.attention_layer = Attention(embed_dim, num_heads, dropout_rate)

        # Layer normalization and feed-forward
        self.layer_norm1 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.layer_norm2 = nn.LayerNorm(embed_dim, eps=1e-5)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_ff, embed_dim),
            nn.Dropout(dropout_rate),
        )

        # Classification head
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Apply causal convolutions
        x_src = self.causal_conv1(x)
        x_src = self.causal_conv2(x_src)
        x_src = self.causal_conv3(x_src)
        x_src = x_src.permute(0, 2, 1)  # [batch_size, seq_len, embed_dim]

        # Apply positional encoding if specified
        if self.fix_pos_encode != "None":
            x_src_pos = self.fix_position(x_src)
            att = x_src + self.attention_layer(x_src_pos)
        else:
            att = x_src + self.attention_layer(x_src)

        # Apply normalization and feed-forward
        att = self.layer_norm1(att)
        out = att + self.feed_forward(att)
        out = self.layer_norm2(out)

        # Apply classification head
        out = out.permute(0, 2, 1)  # [batch_size, embed_dim, seq_len]
        out = self.gap(out)  # [batch_size, embed_dim, 1]
        out = self.flatten(out)  # [batch_size, embed_dim]
        out = self.out(out)  # [batch_size, num_classes]

        return out


class CausalConv1d(nn.Conv1d):
    """
    1D Causal convolution layer - ensures output at time t depends only on inputs up to time t
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.__padding = (kernel_size - 1) * dilation

    def forward(self, x):
        return super(CausalConv1d, self).forward(
            nn.functional.pad(x, (self.__padding, 0))
        )


def model_factory(config):
    """Factory function to create the appropriate model based on configuration"""
    if config["Net_Type"] == "CC-T":
        model = CasualConvTran(
            seq_len=config["Data_shape"][2],
            num_classes=config["num_labels"],
            num_channels=config["Data_shape"][1],
            embed_dim=config["emb_size"],
            num_heads=config["num_heads"],
            dim_ff=config["dim_ff"],
            dropout_rate=config["dropout"],
            fix_pos_encode=config["Fix_pos_encode"],
            rel_pos_encode=config["Rel_pos_encode"],
        )
        print("###########CASUALCONVTRAN IS RUNNING#######")
    else:
        model = ConvTran(
            seq_len=config["Data_shape"][2],
            num_classes=config["num_labels"],
            num_channels=config["Data_shape"][1],
            embed_dim=config["emb_size"],
            num_heads=config["num_heads"],
            dim_ff=config["dim_ff"],
            dropout_rate=config["dropout"],
            fix_pos_encode=config["Fix_pos_encode"],
            rel_pos_encode=config["Rel_pos_encode"],
        )
        print("###########CONVTRAN IS RUNNING#######")
    return model
