import datetime
import json
import os

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.metrics import accuracy_score, classification_report, f1_score


def get_run_name(args):
    """
    Create a standardized run name based on model configuration

    Args:
        args: Argument namespace with model config

    Returns:
        Formatted run name string
    """
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
    dataset_name = os.path.basename(args.data_path)

    # Compose components that define this experiment
    components = [
        f"{dataset_name}",
        f"dim{args.embed_dim}",
        f"depth{args.depth}",
    ]

    # Add model-specific components
    if hasattr(args, "use_asb"):
        components.append(f"ASB_{args.use_asb}")
    if hasattr(args, "use_icb"):
        components.append(f"ICB_{args.use_icb}")
    if hasattr(args, "adaptive_filter"):
        components.append(f"AF_{args.adaptive_filter}")
    if hasattr(args, "pretrained"):
        components.append(f"pretrained_{args.pretrained}")

    run_name = "_".join(components) + f"_{timestamp}"
    return run_name


def setup_training_environment(args, run_name=None):
    """
    Set up the Lightning training environment without file operations

    Args:
        args: Argument namespace with training configs
        run_name: Optional custom run name

    Returns:
        Dummy directory path (not used)
    """
    # For reproducibility only
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    L.seed_everything(args.seed if hasattr(args, "seed") else 42)

    return "dummy_checkpoint_dir"  # Not used


def create_callbacks(checkpoint_dir, monitor="val_loss", mode="min", patience=10):
    """
    Create standard callbacks for Lightning training

    Args:
        checkpoint_dir: Directory to save checkpoints
        monitor: Metric to monitor for early stopping and checkpointing
        mode: 'min' or 'max' for the monitored metric
        patience: Number of epochs to wait for improvement

    Returns:
        List of callbacks
    """
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch}-{val_loss:.4f}-{val_acc:.4f}",
        monitor=monitor,
        mode=mode,
        save_top_k=1,
        save_last=True,
    )

    early_stopping = EarlyStopping(
        monitor=monitor, patience=patience, mode=mode, verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    progress_bar = TQDMProgressBar(refresh_rate=10)

    return [checkpoint_callback, early_stopping, lr_monitor, progress_bar]


def train_tslanet(model, train_loader, val_loader, test_loader, args, checkpoint_dir):
    """
    Train a TSLANet model
    """
    # Set up logger (optional, can be removed if you want no logs at all)
    logger = None

    # Create callbacks (remove ModelCheckpoint and file saving)
    callbacks = []
    if args.patience > 0:
        early_stopping = EarlyStopping(
            monitor="val_acc" if args.monitor_metric == "accuracy" else "val_loss",
            patience=args.patience,
            mode="max" if args.monitor_metric == "accuracy" else "min",
            verbose=True,
        )
        progress_bar = TQDMProgressBar(refresh_rate=10)
        callbacks = [early_stopping, progress_bar]

    # Setup trainer
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=False,
        log_every_n_steps=10,
        accelerator="auto",
        devices=1,
        deterministic=True,
        enable_progress_bar=True,
        enable_model_summary=False,
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)

    # Test model (no checkpoint reload)
    test_results = trainer.test(model, test_loader)[0]

    # Save results to simple file with config and data info
    with open("results.txt", "w") as f:
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data file: {args.data_path}\n")
        f.write("Config:\n")
        for k, v in vars(args).items():
            f.write(f"  {k}: {v}\n")
        f.write("\nResults:\n")
        f.write(f"Test Accuracy: {test_results['test_acc']:.4f}\n")
        f.write(f"Test F1: {test_results['test_f1']:.4f}\n")
        f.write(f"Test Loss: {test_results['test_loss']:.4f}\n")
    
    return model, test_results


def pretrain_tslanet(model, train_loader, val_loader, args, checkpoint_dir):
    """
    Pretrain a TSLANet model using masked sequence modeling

    Args:
        model: TSLANet pretraining model
        train_loader: Training data loader
        val_loader: Validation data loader
        args: Training arguments
        checkpoint_dir: Directory to save checkpoints

    Returns:
        Path to best pretrained model checkpoint
    """
    # Set up logger
    logger = TensorBoardLogger(save_dir=os.path.join(checkpoint_dir, "pretrain_logs"))

    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="pretrain-{epoch}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=args.pretrain_patience if hasattr(args, "pretrain_patience") else 10,
        mode="min",
        verbose=True,
    )

    callbacks = [
        checkpoint_callback,
        early_stopping,
        LearningRateMonitor("epoch"),
        TQDMProgressBar(refresh_rate=10),
    ]

    # Setup trainer
    trainer = L.Trainer(
        max_epochs=args.pretrain_epochs,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        accelerator="auto",
        devices=1,
        deterministic=True,
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)

    # Return path to best model
    best_model_path = checkpoint_callback.best_model_path
    return best_model_path


def generate_classification_report(
    model, test_loader, class_names=None, output_dir=None
):
    """
    Generate detailed classification report for model evaluation

    Args:
        model: Trained model
        test_loader: Test data loader
        class_names: List of class names (optional)
        output_dir: Directory to save report (optional)

    Returns:
        DataFrame with classification metrics
    """
    device = next(model.parameters()).device
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch[:2]
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    # Generate report
    report_dict = classification_report(
        all_targets, all_preds, target_names=class_names, digits=4, output_dict=True
    )

    # Convert to DataFrame
    df = pd.DataFrame(report_dict).transpose()

    # Add accuracy
    accuracy = accuracy_score(all_targets, all_preds)
    df.loc["accuracy", :] = [accuracy, None, None, None]

    # Save report if output_dir provided
    if output_dir is not None:
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f"classification_report_{timestamp}.csv")
        df.to_csv(report_path)

        # Also save as Excel if pandas has Excel support
        try:
            excel_path = os.path.join(
                output_dir, f"classification_report_{timestamp}.xlsx"
            )
            df.to_excel(excel_path)
        except:
            pass

    return df
