import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange


def plot_code_heatmap(code_indices, num_codes, title=''):
    """
    Plots a heatmap for visualizing the use of codes in a vector-quantization model.

    Parameters:
    - code_indices: torch.Tensor, a 2D tensor where each element is a code index.
    - num_codes: int, the total number of different codes.

    The function creates a heatmap where each row represents a different code and
    each column represents a position in the input tensor, showing the frequency of each code.
    """
    code_indices = code_indices.cpu()
    # Initialize a frequency matrix
    codes, counts = torch.unique(code_indices, return_counts=True)
    heatmap = torch.zeros(num_codes).scatter_(-1, codes, counts.float())
    if num_codes <= 64:
        heatmap = heatmap.view(8, -1)
    elif num_codes <= 256:
        heatmap = heatmap.view(16, -1)
    elif num_codes <= 1024:
        heatmap = heatmap.view(32, -1)
    else:
        heatmap = heatmap.view(64, -1)
    # heatmap = heatmap.view(int(np.sqrt(num_codes)), -1)

    heatmap = heatmap.numpy()

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(heatmap, aspect='auto')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Frequency")
    ax.set_title(f'Code Usage Heatmap - step {title}')
    plt.tight_layout()

    return fig


def visualize_shapes(attribute_dict, num_sample=10, num_s_sample=25, title=''):
    '''
    Visualize the decoded shapes and time series.
    attribute_dict: a dict of attributes to visualize. With the following keys:
        x_true: real time series.
        x_pred: reconstructed time series.
        s_true: real subsequences.
        s_pred: decoded shapes.
        t_pred: start times of the shapes.
        l_pred: lengths of the shapes.
        mu_pred: offset of the shapes.
        sigma_pred: standard deviation of the shapes.
    '''
    for k, v in attribute_dict.items():
        attribute_dict[k] = v.float().cpu().numpy()
    
    sample_idx = np.random.randint(0, attribute_dict['x_true'].shape[0], num_sample)

    # Visualize time series and all 64 shapes
    fig = plt.figure(figsize=(30, 4))
    for i, idx in enumerate(sample_idx):
        ax = fig.add_subplot(num_sample//5, 5, i+1)
        ax.plot(np.linspace(0, 1, attribute_dict['x_true'].shape[-1]), attribute_dict['x_true'][idx], color='tab:grey', linewidth=5, alpha=0.3)
        ax.plot(np.linspace(0, 1, attribute_dict['x_true'].shape[-1]), attribute_dict['x_pred'][idx], color='tab:blue', linewidth=5, alpha=0.3)
        for j in range(attribute_dict['t_pred'].shape[1]):
            ts = np.linspace(attribute_dict['t_pred'][idx, j], min(attribute_dict['t_pred'][idx, j]+attribute_dict['l_pred'][idx, j], 1), attribute_dict['s_pred'][idx, j].shape[-1])
            ax.plot(ts, attribute_dict['s_pred'][idx, j])
    plt.tight_layout()

    # Visualize each decoded shape and its corresponding real subsequence
    s_true = rearrange(attribute_dict['s_true'], 'B N L -> (B N) L')
    s_pred = rearrange(attribute_dict['s_pred'], 'B N L -> (B N) L')
    t_pred = rearrange(attribute_dict['t_pred'], 'B N L -> (B N) L')
    l_pred = rearrange(attribute_dict['l_pred'], 'B N L -> (B N) L')
    s_samples_idx = np.random.randint(0, s_true.shape[0], num_s_sample)
    s_fig = plt.figure(figsize=(15, 8))
    for i, idx in enumerate(s_samples_idx):
        ax = s_fig.add_subplot(5, num_s_sample//5, i+1)
        ax.plot(np.linspace(t_pred[idx], t_pred[idx] + l_pred[idx], s_true.shape[-1]), s_true[idx], alpha=0.5)
        ax.plot(np.linspace(t_pred[idx], t_pred[idx] + l_pred[idx], s_pred.shape[-1]), s_pred[idx], alpha=0.5)
    plt.tight_layout()

    return fig, s_fig


class Timer:
    def __init__(self):
        self.t = time.time_ns()

    def __call__(self):
        ret = f"Interval: {(time.time_ns() - self.t)/1e6:.1f} ms"
        self.t = time.time_ns()
        return ret


def compute_accuracy(logits, labels):
    """
    Compute the accuracy for multi-class classification.

    Args:
    logits (torch.Tensor): The logits output by the model. Shape: [n_samples, n_classes].
    labels (torch.Tensor): The true labels for the data. Shape: [n_samples].

    Returns:
    float: The accuracy of the predictions as a percentage.
    """
    # Get the indices of the maximum logit values along the second dimension (class dimension)
    # These indices correspond to the predicted classes.
    _, predicted_classes = torch.max(logits, dim=1)

    # Compare the predicted classes to the true labels
    correct_predictions = (predicted_classes == labels).float()  # Convert boolean to float

    # Calculate accuracy
    accuracy = correct_predictions.sum() / len(labels)

    return accuracy.item()  # Convert to Python scalar


def compute_binary_accuracy(logits, labels):
    """
    Compute the accuracy of binary classification predictions.

    Args:
    logits (torch.Tensor): The logits output by the model. Logits are raw, unnormalized scores.
    labels (torch.Tensor): The true labels for the data.

    Returns:
    float: The accuracy of the predictions as a percentage.
    """
    # Convert logits to predictions
    predictions = nn.functional.sigmoid(logits) >= 0.5  # Apply sigmoid and threshold
    labels = labels >= 0.5

    # Compare predictions with true labels
    correct_predictions = (predictions == labels).float()  # Convert boolean to float for summing

    # Calculate accuracy
    accuracy = correct_predictions.sum() / len(labels)

    return accuracy.item()  # Convert to Python scalar


def smooth_labels(labels: torch.Tensor, smoothing: float = 0.05):
    """
    Apply label smoothing to a tensor of binary labels.

    Args:
    labels (torch.Tensor): Tensor of binary labels (0 or 1).
    smoothing (float): Smoothing factor to apply to the labels.

    Returns:
    torch.Tensor: Tensor with smoothed labels.
    """
    # Ensure labels are in float format for the smoothing operation
    labels = labels.float()
    
    # Apply label smoothing
    smoothed_labels = labels * (1 - smoothing) + (1 - labels) * smoothing

    return smoothed_labels


def get_gpu_usage():
    gpu_mem = {}
    for i in range(torch.cuda.device_count()):
        gpu_mem[f'GPU {i}'] = torch.cuda.max_memory_allocated(i)/1e9
        # torch.cuda.reset_peak_memory_stats(i)
    return gpu_mem