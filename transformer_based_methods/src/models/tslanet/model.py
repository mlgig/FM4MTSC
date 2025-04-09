import lightning as L
import torch
import torch.nn as nn
from timm.loss import LabelSmoothingCrossEntropy
from timm.models.layers import DropPath, trunc_normal_
from torchmetrics.classification import MulticlassF1Score

from .layers.blocks import ICB, Adaptive_Spectral_Block, TSLANet_layer
from .layers.embed import PatchEmbed


class TSLANet(L.LightningModule):
    """
    Time Series with Lightweight Adaptive attention Network (TSLANet)

    A transformer-based model for time series analysis that combines:
    - Adaptive Spectral Block (ASB) for frequency domain processing
    - Inverted ConvFFN Block (ICB) for spatial feature extraction
    """

    def __init__(
        self,
        seq_len,
        num_classes,
        num_channels=3,
        embed_dim=128,
        depth=2,
        patch_size=8,
        dropout_rate=0.1,
        learning_rate=1e-3,
        use_asb=True,
        use_icb=True,
        adaptive_filter=True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.use_asb = use_asb
        self.use_icb = use_icb
        self.adaptive_filter = adaptive_filter

        # Patch embedding
        self.patch_embed = PatchEmbed(
            seq_len=seq_len,
            patch_size=patch_size,
            in_chans=num_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout_rate)

        # Transformer layers
        dpr = [x.item() for x in torch.linspace(0, dropout_rate, depth)]
        self.layers = nn.ModuleList(
            [
                TSLANet_layer(
                    dim=embed_dim,
                    drop=dropout_rate,
                    drop_path=dpr[i],
                    use_asb=use_asb,
                    use_icb=use_icb,
                    adaptive_filter=adaptive_filter,
                )
                for i in range(depth)
            ]
        )

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes)

        # Loss and metrics
        self.criterion = LabelSmoothingCrossEntropy()
        self.f1 = MulticlassF1Score(num_classes=num_classes)

        # Initialize weights
        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

        # For pretraining
        self.mask = None

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def pretrain(self, x_in, mask_ratio=0.4):
        """
        Forward pass for pretraining with masked patch prediction

        Args:
            x_in: Input data [B, C, N]
            mask_ratio: Ratio of patches to mask

        Returns:
            x_masked: Output from masked sequence
            x_patched: Original patched sequence for reconstruction loss
        """
        from ...utils.utils import random_masking_3D

        x = self.patch_embed(x_in)  # [B, num_patches, embed_dim]
        x = x + self.pos_embed
        x_patched = self.pos_drop(x)

        x_masked, _, self.mask, _ = random_masking_3D(x, mask_ratio=mask_ratio)
        self.mask = self.mask.bool()  # mask: [bs x num_patch x n_vars]

        for layer in self.layers:
            x_masked = layer(x_masked)

        return x_masked, x_patched

    def forward(self, x):
        """
        Forward pass for classification

        Args:
            x: Input data [B, C, N]

        Returns:
            Class logits [B, num_classes]
        """
        # x: [B, C, N]
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = x.mean(1)  # Global average pooling
        return self.head(x)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        f1 = self.f1(logits, y)

        self.log("train_loss", loss)
        self.log("train_acc", acc)
        self.log("train_f1", f1)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        f1 = self.f1(logits, y)

        self.log("val_loss", loss)
        self.log("val_acc", acc)
        self.log("val_f1", f1)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        f1 = self.f1(logits, y)

        self.log("test_loss", loss)
        self.log("test_acc", acc)
        self.log("test_f1", f1)


class TSLANetPretraining(L.LightningModule):
    """
    Pretraining wrapper for TSLANet using masked patch prediction
    """

    def __init__(
        self,
        seq_len,
        num_classes,
        num_channels=3,
        embed_dim=128,
        depth=2,
        patch_size=8,
        dropout_rate=0.1,
        learning_rate=1e-3,
        masking_ratio=0.4,
        use_asb=True,
        use_icb=True,
        adaptive_filter=True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.masking_ratio = masking_ratio
        self.learning_rate = learning_rate

        self.model = TSLANet(
            seq_len=seq_len,
            num_classes=num_classes,
            num_channels=num_channels,
            embed_dim=embed_dim,
            depth=depth,
            patch_size=patch_size,
            dropout_rate=dropout_rate,
            use_asb=use_asb,
            use_icb=use_icb,
            adaptive_filter=adaptive_filter,
        )

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        return optimizer

    def _calculate_loss(self, batch, mode="train"):
        data = batch[0]

        preds, target = self.model.pretrain(data, mask_ratio=self.masking_ratio)

        loss = (preds - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * self.model.mask).sum() / self.model.mask.sum()

        # Logging for both step and epoch
        self.log(
            f"{mode}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")
