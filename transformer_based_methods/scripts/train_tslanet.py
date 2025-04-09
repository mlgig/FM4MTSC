"""
Training script for TSLANet model on time series classification tasks.

This script supports:
1. Training from scratch
2. Training with pretraining (masked patch prediction)
3. Various dataset formats (UCR/UEA, custom .pt files)
"""

import argparse
import datetime
import os
import sys
import time

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

# Add parent directory to path to allow importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import create_dataloaders, load_pt_dataset, load_ucr_dataset
from src.models.tslanet.model import TSLANet, TSLANetPretraining
from src.training.trainer import (
    generate_classification_report,
    pretrain_tslanet,
    setup_training_environment,
    train_tslanet,
)
from src.utils.utils import plot_confusion_matrix, save_source_files, str2bool


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train TSLANet for time series classification"
    )

    # Dataset parameters
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to dataset directory"
    )
    parser.add_argument(
        "--dataset_format",
        type=str,
        choices=["ucr", "pt", "custom"],
        default="ucr",
        help="Dataset format (ucr, pt, or custom)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Dataset name (for UCR/UEA datasets)",
    )

    # Model parameters
    parser.add_argument(
        "--embed_dim", type=int, default=128, help="Embedding dimension"
    )
    parser.add_argument("--depth", type=int, default=2, help="Number of TSLANet layers")
    parser.add_argument(
        "--patch_size", type=int, default=8, help="Patch size for embedding"
    )
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--use_asb", type=str2bool, default=True, help="Use Adaptive Spectral Block"
    )
    parser.add_argument(
        "--use_icb", type=str2bool, default=True, help="Use Inverted ConvFFN Block"
    )
    parser.add_argument(
        "--adaptive_filter",
        type=str2bool,
        default=True,
        help="Use adaptive filtering in ASB",
    )

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
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    # Pretraining parameters
    parser.add_argument(
        "--pretrain", type=str2bool, default=False, help="Whether to use pretraining"
    )
    parser.add_argument(
        "--pretrain_epochs",
        type=int,
        default=50,
        help="Maximum number of pretraining epochs",
    )
    parser.add_argument(
        "--pretrain_lr", type=float, default=1e-3, help="Pretraining learning rate"
    )
    parser.add_argument(
        "--masking_ratio", type=float, default=0.4, help="Masking ratio for pretraining"
    )

    # Output parameters
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Output directory"
    )
    parser.add_argument(
        "--run_name", type=str, default=None, help="Run name (auto-generated if None)"
    )

    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()

    # Create checkpoint directory and save config
    checkpoint_dir = setup_training_environment(args)

    # Save source files for reproducibility
    save_source_files(checkpoint_dir)

    # Log start time
    start_time = time.time()
    print(
        f"Starting TSLANet training: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print(f"Checkpoint directory: {checkpoint_dir}")

    # Load dataset
    if args.dataset_format == "ucr":
        if args.dataset_name is None:
            args.dataset_name = os.path.basename(args.data_path)

        train_dataset, test_dataset = load_ucr_dataset(
            os.path.dirname(args.data_path), args.dataset_name
        )

        train_loader, val_loader, test_loader = create_dataloaders(
            train_dataset, test_dataset, batch_size=args.batch_size
        )

        # Set dataset properties
        args.num_classes = len(torch.unique(train_dataset.y))
        args.seq_len = train_dataset.X.shape[2]
        args.num_channels = train_dataset.X.shape[1]

    elif args.dataset_format == "pt":
        train_loader, val_loader, test_loader = load_pt_dataset(
            args.data_path, args.patch_size
        )

        # Set dataset properties
        sample_batch = next(iter(train_loader))
        x_sample, y_sample = sample_batch
        args.num_classes = len(torch.unique(y_sample))
        args.seq_len = x_sample.shape[2]
        args.num_channels = x_sample.shape[1]

    else:  # custom
        # Implement your custom dataset loading logic here
        raise NotImplementedError("Custom dataset loading not implemented")

    print(
        f"Dataset loaded: {args.num_classes} classes, {args.seq_len} sequence length, {args.num_channels} channels"
    )
    print(f"Batch size: {args.batch_size}, Number of batches: {len(train_loader)}")

    # Pretraining if requested
    best_pretrained_path = None
    if args.pretrain:
        print("Starting pretraining phase...")
        pretraining_model = TSLANetPretraining(
            seq_len=args.seq_len,
            num_classes=args.num_classes,
            num_channels=args.num_channels,
            embed_dim=args.embed_dim,
            depth=args.depth,
            patch_size=args.patch_size,
            dropout_rate=args.dropout_rate,
            learning_rate=args.pretrain_lr,
            masking_ratio=args.masking_ratio,
            use_asb=args.use_asb,
            use_icb=args.use_icb,
            adaptive_filter=args.adaptive_filter,
        )

        best_pretrained_path = pretrain_tslanet(
            pretraining_model, train_loader, val_loader, args, checkpoint_dir
        )
        print(f"Pretraining complete. Best model saved at: {best_pretrained_path}")

    # Create model for finetuning
    if best_pretrained_path and os.path.exists(best_pretrained_path):
        print(f"Loading pretrained model from: {best_pretrained_path}")

        # Load the model from the pretraining checkpoint
        # We need to extract just the TSLANet part since we don't need the pretraining head
        pretrained_ckpt = torch.load(best_pretrained_path)

        # Create a new model for fine-tuning
        model = TSLANet(
            seq_len=args.seq_len,
            num_classes=args.num_classes,
            num_channels=args.num_channels,
            embed_dim=args.embed_dim,
            depth=args.depth,
            patch_size=args.patch_size,
            dropout_rate=args.dropout_rate,
            learning_rate=args.learning_rate,
            use_asb=args.use_asb,
            use_icb=args.use_icb,
            adaptive_filter=args.adaptive_filter,
        )

        # Extract model weights from the pretraining checkpoint
        # This ignores the pretraining-specific parts
        model_dict = model.state_dict()
        pretrained_dict = {
            k.replace("model.", ""): v
            for k, v in pretrained_ckpt["state_dict"].items()
            if k.replace("model.", "") in model_dict
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        print(
            f"Loaded {len(pretrained_dict)}/{len(model_dict)} parameters from pretrained model"
        )
    else:
        print("Training model from scratch")
        model = TSLANet(
            seq_len=args.seq_len,
            num_classes=args.num_classes,
            num_channels=args.num_channels,
            embed_dim=args.embed_dim,
            depth=args.depth,
            patch_size=args.patch_size,
            dropout_rate=args.dropout_rate,
            learning_rate=args.learning_rate,
            use_asb=args.use_asb,
            use_icb=args.use_icb,
            adaptive_filter=args.adaptive_filter,
        )

    # Train model
    model, results = train_tslanet(
        model, train_loader, val_loader, test_loader, args, checkpoint_dir
    )

    # Generate classification report
    class_names = [str(i) for i in range(args.num_classes)]
    df_report = generate_classification_report(
        model, test_loader, class_names, checkpoint_dir
    )

    # Extract predictions for confusion matrix
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

    # Plot confusion matrix
    fig = plot_confusion_matrix(
        all_targets,
        all_preds,
        class_names=class_names,
        output_path=os.path.join(checkpoint_dir, "confusion_matrix.png"),
    )

    # Log end time and results
    end_time = time.time()
    training_duration = end_time - start_time

    print(f"\n====== Training completed in {training_duration:.2f} seconds ======")
    print(f"Test accuracy: {results['test_accuracy']:.4f}")
    print(f"Test F1 score: {results['test_f1']:.4f}")
    print(f"Best model saved at: {results['best_model_path']}")
    print(f"Full results and logs saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()
