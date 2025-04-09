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
    """Set up experiment directories and save configuration"""
    # Create output directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        args.output_dir, f"{args.dataset_name or 'dataset'}_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Create subdirectories
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    predictions_dir = os.path.join(output_dir, "predictions")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    # Save configuration
    config = vars(args)
    config["output_dir"] = output_dir
    config["checkpoints_dir"] = checkpoints_dir
    config["predictions_dir"] = predictions_dir

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    return config


def load_dataset_wrapper(config):
    """Load dataset using the unified loader function"""
    logger.info(
        f"Loading dataset from {config['data_path']} with format {config['dataset_format']}"
    )

    # Use the unified load_dataset function
    train_dataset, test_dataset = load_dataset(
        data_path=config["data_path"],
        format_type=config["dataset_format"],
        normalize=config["normalize"],
        dataset_name=config["dataset_name"],
    )

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=config["batch_size"],
        val_split=config["val_ratio"],
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )

    # Update config with dataset properties
    # Get a sample to determine dimensions
    sample_x, sample_y = train_dataset[0][:2]  # Ignore index if present

    config["num_classes"] = len(
        torch.unique(
            torch.tensor([item[1] for item in train_dataset if item[1] is not None])
        )
    )
    config["seq_len"] = sample_x.shape[-1]
    config["num_channels"] = sample_x.shape[0]
    config["class_names"] = [str(i) for i in range(config["num_classes"])]

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

    Args:
        model: ConvTran model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        config: Configuration dictionary

    Returns:
        Trained model and results
    """
    # Create Lightning module
    model_module = ConvTranModule(
        model=model,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config["checkpoints_dir"],
        filename="{epoch:02d}-{val_acc:.4f}",
        monitor=f"val_{'loss' if config['monitor_metric'] == 'loss' else 'acc'}",
        mode="min" if config["monitor_metric"] == "loss" else "max",
        save_top_k=1,
        save_last=True,
    )

    early_stopping = EarlyStopping(
        monitor=f"val_{'loss' if config['monitor_metric'] == 'loss' else 'acc'}",
        patience=config["patience"],
        mode="min" if config["monitor_metric"] == "loss" else "max",
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Setup logger
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(config["output_dir"], "logs"), name="convtran"
    )

    # Setup trainer
    trainer = L.Trainer(
        max_epochs=config["max_epochs"],
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        logger=tb_logger,
        log_every_n_steps=10,
        accelerator="auto",
        devices=1,
    )

    # Train the model
    trainer.fit(model_module, train_loader, val_loader)

    # Test the model
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        logger.info(f"Loading best model from {best_model_path}")
        model_module = ConvTranModule.load_from_checkpoint(best_model_path, model=model)

    test_results = trainer.test(model_module, test_loader)

    # Calculate and save additional metrics
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch[:2]  # Ignore sample ID if present
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Save confusion matrix
    try:
        plot_confusion_matrix(
            y_true,
            y_pred,
            class_names=config["class_names"],
            output_path=os.path.join(config["output_dir"], "confusion_matrix.png"),
        )
    except Exception as e:
        logger.error(f"Failed to plot confusion matrix: {e}")

    # Save results
    results = {
        "test_accuracy": test_results[0]["test_acc"],
        "test_loss": test_results[0]["test_loss"],
        "test_f1": test_results[0].get("test_f1", None),
        "best_model_path": best_model_path,
        "num_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }

    with open(os.path.join(config["output_dir"], "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    return model, results


def main():
    """Main function"""
    # Parse arguments
    args = parse_args()

    # Setup environment
    config = setup_environment(args)

    # Save source files for reproducibility
    try:
        save_source_files(config["output_dir"])
    except Exception as e:
        logger.warning(f"Failed to save source files: {e}")

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
    logger.info(f"Test accuracy: {results['test_accuracy']:.4f}")
    if results["test_f1"]:
        logger.info(f"Test F1 score: {results['test_f1']:.4f}")
    logger.info(f"Results saved to {config['output_dir']}")

    return results


if __name__ == "__main__":
    main()
