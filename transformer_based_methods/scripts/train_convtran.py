#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for ConvTran model on time series classification tasks.

This script supports:
1. Training from scratch
2. Various dataset formats (UCR/UEA, custom .pt files)
3. Different positional encoding variants
"""

import argparse
import datetime
import json
import logging
import os
import sys
import time

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

# Add parent directory to path to allow importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import (
    TimeSeriesDataset,
    create_dataloaders,
    load_dataset,
    load_pt_dataset,
    load_ucr_dataset,
)
from src.models.convtran.model import model_factory
from src.utils.utils import plot_confusion_matrix, save_source_files, str2bool

# Configure logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train ConvTran for time series classification"
    )

    # Dataset parameters
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to dataset directory"
    )
    parser.add_argument(
        "--dataset_format",
        type=str,
        choices=["ucr", "pt", "npy", "auto"],
        default="auto",
        help="Dataset format (ucr, pt, npy, or auto)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Dataset name (for UCR/UEA datasets)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Proportion of training data to use for validation",
    )
    parser.add_argument(
        "--normalize", type=str2bool, default=True, help="Whether to normalize the data"
    )

    # Model parameters
    parser.add_argument(
        "--net_type",
        type=str,
        choices=["C-T", "CC-T"],
        default="C-T",
        help="Network type (C-T: ConvTran, CC-T: CausalConvTran)",
    )
    parser.add_argument("--emb_size", type=int, default=128, help="Embedding dimension")
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--dim_ff", type=int, default=256, help="Dimension of feed-forward network"
    )
    parser.add_argument(
        "--fix_pos_encode",
        type=str,
        choices=["tAPE", "Sin", "Learn", "None"],
        default="tAPE",
        help="Type of fixed positional encoding",
    )
    parser.add_argument(
        "--rel_pos_encode",
        type=str,
        choices=["eRPE", "Vector", "None"],
        default="None",
        help="Type of relative positional encoding",
    )
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--max_epochs", type=int, default=100, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Patience for early stopping"
    )
    parser.add_argument(
        "--monitor_metric",
        type=str,
        choices=["loss", "accuracy"],
        default="accuracy",
        help="Metric to monitor for early stopping",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--pin_memory",
        type=str2bool,
        default=True,
        help="Whether to pin memory for data loading",
    )

    # Output parameters
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Output directory"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    return parser.parse_args()


def setup_environment(args):
    """Set up environment without creating directories"""
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Return config without file operations
    config = vars(args)
    config["output_dir"] = "results"  # Dummy value, not used
    return config


def load_dataset_wrapper(config):
    """Load dataset using the unified loader function"""
    logger.info(
        f"Loading dataset from {config['data_path']} with format {config['dataset_format']}"
    )

    # Support single .npy dict file directly
    if config["data_path"].endswith('.npy'):
        if not os.path.exists(config["data_path"]):
            raise FileNotFoundError(f"Data file not found: {config['data_path']}")
        data = np.load(config["data_path"], allow_pickle=True).item()
        X_train = data["train"]["X"]
        y_train = np.array([int(x) for x in data["train"]["y"]])
        X_test = data["test"]["X"]
        y_test = np.array([int(x) for x in data["test"]["y"]])
        train_dataset = TimeSeriesDataset(X_train, y_train, normalize=config["normalize"])
        test_dataset = TimeSeriesDataset(X_test, y_test, normalize=config["normalize"])
        train_loader, val_loader, test_loader = create_dataloaders(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            batch_size=config["batch_size"] if isinstance(config, dict) else config.batch_size,
            val_split=config["val_ratio"] if isinstance(config, dict) else getattr(config, 'val_ratio', 0.2),
            num_workers=config["num_workers"] if isinstance(config, dict) else getattr(config, 'num_workers', 4),
            pin_memory=config["pin_memory"] if isinstance(config, dict) else getattr(config, 'pin_memory', True),
        )
        # Set dataset properties for config
        sample_x, sample_y = train_dataset[0][:2]
        num_classes = len(np.unique(y_train))
        seq_len = sample_x.shape[-1]
        num_channels = sample_x.shape[0]
        class_names = [str(i) for i in range(num_classes)]
        if isinstance(config, dict):
            config["num_classes"] = num_classes
            config["seq_len"] = seq_len
            config["num_channels"] = num_channels
            config["class_names"] = class_names
        else:
            config.num_classes = num_classes
            config.seq_len = seq_len
            config.num_channels = num_channels
            config.class_names = class_names
        return train_loader, val_loader, test_loader

    # Otherwise, use the original loader logic
    data_path = config["data_path"] if isinstance(config, dict) else config.data_path
    normalize = config["normalize"] if isinstance(config, dict) else getattr(config, 'normalize', False)
    train_dataset, test_dataset = load_dataset(
        data_path=data_path,
        format_type=config["dataset_format"] if isinstance(config, dict) else getattr(config, 'dataset_format', 'auto'),
        normalize=normalize,
        dataset_name=config["dataset_name"] if isinstance(config, dict) else getattr(config, 'dataset_name', None),
    )
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=config["batch_size"] if isinstance(config, dict) else config.batch_size,
        val_split=config["val_ratio"] if isinstance(config, dict) else getattr(config, 'val_ratio', 0.2),
        num_workers=config["num_workers"] if isinstance(config, dict) else getattr(config, 'num_workers', 4),
        pin_memory=config["pin_memory"] if isinstance(config, dict) else getattr(config, 'pin_memory', True),
    )
    # Set dataset properties for config
    sample_x, sample_y = train_dataset[0][:2]
    num_classes = len(np.unique([item[1] for item in train_dataset if item[1] is not None]))
    seq_len = sample_x.shape[-1]
    num_channels = sample_x.shape[0]
    class_names = [str(i) for i in range(num_classes)]
    if isinstance(config, dict):
        config["num_classes"] = num_classes
        config["seq_len"] = seq_len
        config["num_channels"] = num_channels
        config["class_names"] = class_names
    else:
        config.num_classes = num_classes
        config.seq_len = seq_len
        config.num_channels = num_channels
        config.class_names = class_names
    return train_loader, val_loader, test_loader


class ConvTranModule(L.LightningModule):
    """PyTorch Lightning module for ConvTran model"""

    def __init__(self, model, learning_rate=1e-3, weight_decay=1e-4):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch[
            :2
        ]  # First two elements are input and target, third might be index
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[:2]
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        x, y = batch[:2]
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        # Calculate F1 score for multi-class (requires sklearn)
        # Commented out to avoid dependency, uncomment if sklearn is available
        """
        from sklearn.metrics import f1_score
        f1 = f1_score(y.cpu().numpy(), preds.cpu().numpy(), average='macro')
        self.log('test_f1', f1)
        """

        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return {"test_loss": loss, "test_acc": acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


def train_convtran(model, train_loader, val_loader, test_loader, config):
    """
    Train the ConvTran model
    """
    # Create Lightning module
    model_module = ConvTranModule(
        model=model,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    # Setup callbacks (no ModelCheckpoint, no file saving)
    early_stopping = EarlyStopping(
        monitor=f"val_{'loss' if config['monitor_metric'] == 'loss' else 'acc'}",
        patience=config["patience"],
        mode="min" if config["monitor_metric"] == "loss" else "max",
        verbose=True,
    )

    # No logger at all
    logger = None

    # Setup trainer
    trainer = L.Trainer(
        max_epochs=config["max_epochs"],
        callbacks=[early_stopping],
        logger=False,
        log_every_n_steps=10,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        enable_model_summary=False,
    )

    # Train the model
    trainer.fit(model_module, train_loader, val_loader)

    # Test the model (no checkpoint reload)
    test_results = trainer.test(model_module, test_loader)

    # Save results to simple file with config and data info
    with open("results.txt", "w") as f:
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data file: {config['data_path']}\n")
        f.write("Config:\n")
        for k, v in config.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nResults:\n")
        f.write(f"Test Accuracy: {test_results[0]['test_acc']:.4f}\n")
        f.write(f"Test Loss: {test_results[0]['test_loss']:.4f}\n")

    return model, test_results[0]


def main():
    """Main function"""
    # Parse arguments
    args = parse_args()

    # Setup environment
    config = setup_environment(args)

    # Source files saving removed - only metrics are saved

    # Load dataset using the unified loader
    train_loader, val_loader, test_loader = load_dataset_wrapper(config)

    # Log dataset info
    logger.info(f"Dataset loaded with {config['num_classes']} classes")
    logger.info(
        f"Sequence length: {config['seq_len']}, Channels: {config['num_channels']}"
    )

    # Create model
    logger.info("Creating ConvTran model...")
    model_config = {
        "Net_Type": config["net_type"],
        "Data_shape": [None, config["num_channels"], config["seq_len"]],
        "num_labels": config["num_classes"],
        "emb_size": config["emb_size"],
        "num_heads": config["num_heads"],
        "dim_ff": config["dim_ff"],
        "Fix_pos_encode": config["fix_pos_encode"],
        "Rel_pos_encode": config["rel_pos_encode"],
        "dropout": config["dropout_rate"],
    }

    model = model_factory(model_config)
    logger.info(
        f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters"
    )

    # Train the model
    logger.info("Starting model training...")
    start_time = time.time()
    model, results = train_convtran(
        model, train_loader, val_loader, test_loader, config
    )
    end_time = time.time()

    # Log training time and results
    logger.info(f"Training completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Test accuracy: {results['test_acc']:.4f}")
    if 'test_f1' in results and results['test_f1'] is not None:
        logger.info(f"Test F1 score: {results['test_f1']:.4f}")
    logger.info("Results saved to results.txt")

    return results


if __name__ == "__main__":
    main()
