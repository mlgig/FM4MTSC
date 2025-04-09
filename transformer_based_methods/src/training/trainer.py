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
    ModelCheckpoint,
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
    Set up the Lightning training environment

    Args:
        args: Argument namespace with training configs
        run_name: Optional custom run name

    Returns:
        Directory path for checkpoints, callbacks
    """
    if run_name is None:
        run_name = get_run_name(args)

    # Create checkpoint directory
    checkpoint_dir = os.path.join("checkpoints", run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save config
    with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
        config_dict = vars(args)
        json.dump(config_dict, f, indent=4)

    # For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    L.seed_everything(args.seed if hasattr(args, "seed") else 42)

    return checkpoint_dir


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

    Args:
        model: TSLANet model (LightningModule)
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        args: Training arguments
        checkpoint_dir: Directory to save checkpoints

    Returns:
        Trained model, results dictionary
    """
    # Set up logger
    logger = TensorBoardLogger(save_dir=os.path.join(checkpoint_dir, "logs"))

    # Create callbacks
    callbacks = create_callbacks(
        checkpoint_dir=checkpoint_dir,
        monitor="val_acc" if args.monitor_metric == "accuracy" else "val_loss",
        mode="max" if args.monitor_metric == "accuracy" else "min",
        patience=args.patience,
    )

    # Setup trainer
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        accelerator="auto",
        devices=1,
        deterministic=True,
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)

    # Load best model for evaluation
    best_model_path = callbacks[0].best_model_path
    if os.path.exists(best_model_path):
        model = type(model).load_from_checkpoint(best_model_path)

    # Test model
    test_results = trainer.test(model, test_loader)[0]

    # Save test results
    results = {
        "test_accuracy": test_results["test_acc"],
        "test_f1": test_results["test_f1"],
        "test_loss": test_results["test_loss"],
        "best_model_path": best_model_path,
    }

    with open(os.path.join(checkpoint_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    return model, results


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
            x, y = batch
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
