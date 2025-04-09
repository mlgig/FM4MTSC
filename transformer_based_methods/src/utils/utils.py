import argparse
import datetime
import inspect
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix


def str2bool(v):
    """
    Convert string representation of boolean to actual boolean

    Args:
        v: String or boolean input

    Returns:
        Boolean value
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def save_source_files(checkpoint_dir):
    """
    Save copies of source files to checkpoint directory for reproducibility

    Args:
        checkpoint_dir: Directory to save files
    """
    # Get the frame of the caller
    caller_frame = inspect.currentframe().f_back

    # Get the filename of the caller
    caller_filename = caller_frame.f_globals["__file__"]

    # Get the absolute path of the caller script
    caller_script_path = os.path.abspath(caller_filename)

    # Ensure the destination directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Copy the caller script to the destination directory
    shutil.copy(caller_script_path, checkpoint_dir)

    # Also copy important python files from the project
    src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".py"):
                rel_dir = os.path.relpath(root, src_dir)
                # Create the corresponding directory structure in checkpoint_dir
                os.makedirs(os.path.join(checkpoint_dir, "src", rel_dir), exist_ok=True)
                # Copy the file
                src_file = os.path.join(root, file)
                dst_file = os.path.join(checkpoint_dir, "src", rel_dir, file)
                shutil.copy(src_file, dst_file)


def random_masking_3D(xb, mask_ratio):
    """
    Random masking for pretraining with masked patch prediction

    Args:
        xb: Input tensor [B, N, D] where:
            B = batch size
            N = number of patches
            D = embedding dimension
        mask_ratio: Fraction of patches to mask

    Returns:
        x_masked: Masked input with zeros in masked positions
        x_kept: Only the kept (unmasked) patches
        mask: Binary mask (1 = masked, 0 = kept)
        ids_restore: Indices to restore original ordering
    """
    # xb: [bs x num_patch x dim]
    bs, L, D = xb.shape
    x = xb.clone()

    len_keep = int(L * (1 - mask_ratio))

    # Generate random noise for masking
    noise = torch.rand(bs, L, device=xb.device)  # noise in [0, 1], bs x L

    # Sort noise for each sample (ascending: small is keep, large is remove)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)  # Indices to restore original order

    # Keep the first subset (lower noise values)
    ids_keep = ids_shuffle[:, :len_keep]  # ids_keep: [bs x len_keep]
    x_kept = torch.gather(
        x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
    )  # x_kept: [bs x len_keep x dim]

    # Create tensor of zeros for masked patches
    x_removed = torch.zeros(
        bs, L - len_keep, D, device=xb.device
    )  # x_removed: [bs x (L-len_keep) x dim]

    # Concatenate kept and zero patches
    x_ = torch.cat([x_kept, x_removed], dim=1)  # x_: [bs x L x dim]

    # Restore original patch ordering with masking applied
    x_masked = torch.gather(
        x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D)
    )  # x_masked: [bs x num_patch x dim]

    # Generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([bs, L], device=x.device)  # mask: [bs x num_patch]
    mask[:, :len_keep] = 0
    # Unshuffle to get the binary mask in the original order
    mask = torch.gather(mask, dim=1, index=ids_restore)  # [bs x num_patch]

    return x_masked, x_kept, mask, ids_restore


def plot_confusion_matrix(
    y_true, y_pred, class_names=None, figsize=(10, 8), output_path=None
):
    """
    Plot confusion matrix for classification results

    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        class_names: List of class names (optional)
        figsize: Figure size (width, height) in inches
        output_path: Path to save the plot (optional)

    Returns:
        Matplotlib figure
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize confusion matrix
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
    )

    # Set labels
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Normalized Confusion Matrix")

    # Save figure if output path provided
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)

    return fig


def plot_training_curves(
    train_metrics, val_metrics, metric_name="loss", figsize=(10, 6), output_path=None
):
    """
    Plot training and validation curves

    Args:
        train_metrics: List of training metric values
        val_metrics: List of validation metric values
        metric_name: Name of the metric (for labeling)
        figsize: Figure size (width, height) in inches
        output_path: Path to save the plot (optional)

    Returns:
        Matplotlib figure
    """
    epochs = range(1, len(train_metrics) + 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(epochs, train_metrics, "b-", label=f"Training {metric_name}")
    ax.plot(epochs, val_metrics, "r-", label=f"Validation {metric_name}")

    ax.set_title(f"Training and Validation {metric_name.capitalize()}")
    ax.set_xlabel("Epochs")
    ax.set_ylabel(metric_name.capitalize())
    ax.legend()
    ax.grid(True)

    # Save figure if output path provided
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)

    return fig


def visualize_spectral_attention(
    model, input_data, layer_idx=0, sample_idx=0, figsize=(15, 10), output_path=None
):
    """
    Visualize the adaptive spectral attention mechanism

    Args:
        model: Trained TSLANet model
        input_data: Input time series [B, C, L]
        layer_idx: Index of the TSLANet layer to visualize
        sample_idx: Index of the sample in the batch to visualize
        figsize: Figure size (width, height) in inches
        output_path: Path to save the plot (optional)

    Returns:
        Matplotlib figure showing original signal, frequency spectrum, and attention weights
    """
    device = next(model.parameters()).device
    model.eval()

    # Forward pass to get patch embeddings
    with torch.no_grad():
        # Get input sample
        x = input_data[sample_idx : sample_idx + 1].to(device)

        # Get patch embeddings
        patches = model.patch_embed(x)
        patches = patches + model.pos_embed
        patches = model.pos_drop(patches)

        # Get the adaptive spectral block from the specified layer
        asb = model.layers[layer_idx].asb

        # Apply layer norm as in the forward pass
        norm_patches = model.layers[layer_idx].norm1(patches)

        # Get the frequency representation
        norm_patches_t = norm_patches.transpose(1, 2)  # [1, C, N]
        x_fft = torch.fft.rfft(norm_patches_t, dim=2, norm="ortho")

        # Get the adaptive mask
        if hasattr(asb, "create_adaptive_high_freq_mask"):
            freq_mask = asb.create_adaptive_high_freq_mask(x_fft)
        else:
            freq_mask = torch.ones_like(x_fft.abs())

        # Get the complex weights
        weight = torch.view_as_complex(asb.complex_weight)
        weight_high = torch.view_as_complex(asb.complex_weight_high)

        # Convert to numpy for plotting
        orig_signal = x[0, 0].cpu().numpy()
        x_fft_abs = x_fft[0, 0].abs().cpu().numpy()
        freq_mask_np = freq_mask[0].cpu().numpy()
        weight_abs = weight.abs().cpu().numpy()
        weight_high_abs = weight_high.abs().cpu().numpy()

    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=figsize)

    # Plot original signal
    axes[0].plot(orig_signal)
    axes[0].set_title("Original Time Series")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Amplitude")

    # Plot frequency spectrum
    freq_indices = np.arange(len(x_fft_abs))
    axes[1].bar(freq_indices, x_fft_abs, alpha=0.7, label="Signal Spectrum")
    axes[1].set_title("Frequency Spectrum")
    axes[1].set_xlabel("Frequency Bin")
    axes[1].set_ylabel("Magnitude")

    # Plot attention weights and mask
    axes[2].bar(freq_indices, weight_abs, alpha=0.5, label="Base Weights")
    axes[2].bar(
        freq_indices,
        weight_high_abs * freq_mask_np.squeeze(),
        alpha=0.5,
        label="High-Freq Weights * Mask",
    )
    axes[2].plot(freq_indices, freq_mask_np.squeeze(), "r--", label="Adaptive Mask")
    axes[2].set_title("Spectral Attention Weights")
    axes[2].set_xlabel("Frequency Bin")
    axes[2].set_ylabel("Weight Magnitude")
    axes[2].legend()

    plt.tight_layout()

    # Save figure if output path provided
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)

    return fig
