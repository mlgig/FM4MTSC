"""
TimesNet model implementation for time series classification.
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.models.timesnet.layers.Conv_Blocks import Inception_Block_V1
from src.models.timesnet.layers.Embed import DataEmbedding


class TimeSeriesDataset(Dataset):
    """
    Dataset for time series data in PyTorch.
    """

    def __init__(self, X, y):
        """
        Initialize the dataset.

        Args:
            X: Features
            y: Labels
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_mark = torch.ones(self.X[idx].shape[0])
        return self.X[idx], x_mark, self.y[idx]


class Config:
    """
    Configuration class for TimesNet model.
    """

    def __init__(
        self,
        task_name="classification",
        seq_len=None,
        num_class=None,
        enc_in=None,
        d_model=64,
        d_ff=64,
        dropout=0.1,
        e_layers=2,
        top_k=3,
        num_kernels=6,
        embed="fixed",
        freq="h",
        label_len=0,
        pred_len=0,
    ):
        """
        Initialize configuration.

        Args:
            task_name: Task name (classification, forecasting, etc.)
            seq_len: Sequence length
            num_class: Number of classes
            enc_in: Number of input features
            d_model: Model dimension
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            e_layers: Number of encoder layers
            top_k: Top-k frequencies for FFT
            num_kernels: Number of kernels in inception blocks
            embed: Embedding type
            freq: Data frequency
            label_len: Label length (for forecasting)
            pred_len: Prediction length (for forecasting)
        """
        self.task_name = task_name
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.num_class = num_class
        self.enc_in = enc_in
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.e_layers = e_layers
        self.top_k = top_k
        self.num_kernels = num_kernels
        self.embed = embed
        self.freq = freq
        self.c_out = enc_in


class TimesBlock(nn.Module):
    """
    TimesBlock module for TimesNet.
    """

    def __init__(self, configs):
        """
        Initialize TimesBlock.

        Args:
            configs: Model configuration
        """
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # Parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(
                configs.d_model, configs.d_ff, num_kernels=configs.num_kernels
            ),
            nn.GELU(),
            Inception_Block_V1(
                configs.d_ff, configs.d_model, num_kernels=configs.num_kernels
            ),
        )

    def FFT_for_Period(self, x, k=2):
        """
        Perform FFT to find periods in time series.

        Args:
            x: Input tensor
            k: Number of top frequencies to return

        Returns:
            Tuple of (periods, period weights)
        """
        # [B, T, C]
        xf = torch.fft.rfft(x, dim=1)
        # Find period by amplitudes
        frequency_list = abs(xf).mean(0).mean(-1)
        frequency_list[0] = 0
        _, top_list = torch.topk(frequency_list, k)
        top_list = top_list.detach().cpu().numpy()
        period = x.shape[1] // top_list
        return period, abs(xf).mean(-1)[:, top_list]

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        B, T, N = x.size()
        period_list, period_weight = self.FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # Padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros(
                    [x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]
                ).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x
            # Reshape
            out = (
                out.reshape(B, length // period, period, N)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # Reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, : (self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # Adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # Residual connection
        res = res + x
        return res


class TimesNetModel(nn.Module):
    """
    TimesNet model implementation.
    """

    def __init__(self, configs):
        """
        Initialize TimesNet model.

        Args:
            configs: Model configuration
        """
        super(TimesNetModel, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList(
            [TimesBlock(configs) for _ in range(configs.e_layers)]
        )
        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)

        # Task-specific layers
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == "imputation" or self.task_name == "anomaly_detection":
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class
            )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Forecasting task.

        Args:
            x_enc: Encoder input
            x_mark_enc: Encoder marks
            x_dec: Decoder input
            x_mark_dec: Decoder marks

        Returns:
            Forecast output
        """
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1
        )  # Align temporal dimension
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # Project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (
            stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        )
        dec_out = dec_out + (
            means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        )
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        """
        Imputation task.

        Args:
            x_enc: Encoder input
            x_mark_enc: Encoder marks
            x_dec: Decoder input
            x_mark_dec: Decoder marks
            mask: Mask tensor

        Returns:
            Imputation output
        """
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(
            torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5
        )
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # Project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (
            stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        )
        dec_out = dec_out + (
            means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        )
        return dec_out

    def anomaly_detection(self, x_enc):
        """
        Anomaly detection task.

        Args:
            x_enc: Encoder input

        Returns:
            Anomaly detection output
        """
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # Project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (
            stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        )
        dec_out = dec_out + (
            means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        )
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        """
        Classification task.

        Args:
            x_enc: Encoder input
            x_mark_enc: Encoder marks

        Returns:
            Classification output
        """
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # The output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # Zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None):
        """
        Forward pass based on task type.

        Args:
            x_enc: Encoder input
            x_mark_enc: Encoder marks
            x_dec: Decoder input (optional)
            x_mark_dec: Decoder marks (optional)
            mask: Mask tensor (optional)

        Returns:
            Task-specific output
        """
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]
        if self.task_name == "imputation":
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == "anomaly_detection":
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == "classification":
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None


class TimesNetWrapper:
    """
    Wrapper for TimesNet model with training and evaluation functionality.
    """

    def __init__(
        self,
        seq_len=None,
        enc_in=None,
        num_class=None,
        d_model=64,
        d_ff=64,
        dropout=0.1,
        e_layers=2,
        top_k=3,
        num_kernels=6,
        batch_size=32,
        epochs=100,
        patience=10,
        learning_rate=0.001,
        device=None,
        model_path="results/timesnet_model.pth",
    ):
        """
        Initialize TimesNet wrapper.

        Args:
            seq_len: Sequence length
            enc_in: Number of input features
            num_class: Number of classes
            d_model: Model dimension
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            e_layers: Number of encoder layers
            top_k: Top-k frequencies for FFT
            num_kernels: Number of kernels in inception blocks
            batch_size: Batch size for training
            epochs: Maximum number of training epochs
            patience: Patience for early stopping
            learning_rate: Learning rate for optimization
            device: Device to use ('cpu', 'cuda', or 'mps')
            model_path: Path to save the model
        """
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.num_class = num_class
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.e_layers = e_layers
        self.top_k = top_k
        self.num_kernels = num_kernels
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.model_path = model_path

        # Set device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Initialize model if dimensions are provided
        if seq_len is not None and enc_in is not None and num_class is not None:
            self._build_model()
        else:
            self.config = None
            self.model = None
            self.criterion = None
            self.optimizer = None

    def _build_model(self):
        """Build the model with current parameters."""
        # Create configuration
        self.config = Config(
            task_name="classification",
            seq_len=self.seq_len,
            num_class=self.num_class,
            enc_in=self.enc_in,
            d_model=self.d_model,
            d_ff=self.d_ff,
            dropout=self.dropout,
            e_layers=self.e_layers,
            top_k=self.top_k,
            num_kernels=self.num_kernels,
        )

        # Create model
        self.model = TimesNetModel(self.config).to(self.device)

        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        verbose: int = 1,
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
            validation_data: Tuple of (X_val, y_val) for validation
            verbose: Verbosity level

        Returns:
            Dictionary with training history
        """
        # Ensure input is in correct format (batch, length, channels)
        if len(X_train.shape) != 3:
            raise ValueError(
                "Input data must be 3D with shape (batch, length, channels)"
            )

        # Set dimensions if not already set
        if self.seq_len is None:
            self.seq_len = X_train.shape[1]

        if self.enc_in is None:
            self.enc_in = X_train.shape[2]

        if self.num_class is None:
            self.num_class = len(np.unique(y_train))

        # Build model if not already built
        if self.model is None:
            self._build_model()

        # Create directory for model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        # Split data for validation if not provided
        if validation_data is None:
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
        else:
            X_train_split, y_train_split = X_train, y_train
            X_val, y_val = validation_data

        # Create datasets and data loaders
        train_dataset = TimeSeriesDataset(X_train_split, y_train_split)
        val_dataset = TimeSeriesDataset(X_val, y_val)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Training history
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "epoch_times": [],
        }

        # Training loop
        best_val_acc, epochs_no_improve = 0, 0
        start_train_time = time.time()

        for epoch in range(self.epochs):
            epoch_start_time = time.time()

            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch_x, batch_x_mark, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_x_mark = batch_x_mark.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_x, batch_x_mark)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

            train_loss = train_loss / len(train_loader)
            train_acc = 100.0 * train_correct / train_total

            # Validation phase
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_x, batch_x_mark, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_x_mark = batch_x_mark.to(self.device)
                    outputs = self.model(batch_x, batch_x_mark)
                    loss = self.criterion(outputs, batch_y.to(self.device))

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted.cpu() == batch_y).sum().item()

            val_loss = val_loss / len(val_loader)
            val_acc = 100.0 * val_correct / val_total
            epoch_time = time.time() - epoch_start_time

            # Store history
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["epoch_times"].append(epoch_time)

            # Print progress
            if verbose > 0:
                print(f"Epoch [{epoch + 1}/{self.epochs}] - Time: {epoch_time:.2f}s")
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Early stopping logic
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), self.model_path)
                if verbose > 0:
                    print(
                        f"New best model saved with validation accuracy: {val_acc:.2f}%"
                    )
            else:
                epochs_no_improve += 1
                if verbose > 0:
                    print(
                        f"Epochs without improvement: {epochs_no_improve}/{self.patience}"
                    )

                if epochs_no_improve >= self.patience:
                    if verbose > 0:
                        print(f"Early stopping triggered after {epoch + 1} epochs")
                    break

        total_train_time = time.time() - start_train_time

        # Add metadata to history
        history["total_time"] = total_train_time
        history["best_val_acc"] = best_val_acc
        history["epochs_completed"] = epoch + 1

        # Load best model
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device)
        )

        if verbose > 0:
            print(f"\nTraining completed in {total_train_time:.2f} seconds")
            print(f"Best validation accuracy: {best_val_acc:.2f}%")

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the trained model.

        Args:
            X: Input features

        Returns:
            Array of predicted class labels
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")

        # Create dataset and loader
        dataset = TimeSeriesDataset(X, np.zeros(X.shape[0]))  # Dummy labels
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch_x, batch_x_mark, _ in loader:
                batch_x = batch_x.to(self.device)
                batch_x_mark = batch_x_mark.to(self.device)
                outputs = self.model(batch_x, batch_x_mark)
                _, predicted = outputs.max(1)
                predictions.extend(predicted.cpu().numpy())

        return np.array(predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability estimates for each class.

        Args:
            X: Input features

        Returns:
            Array of class probabilities
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")

        # Create dataset and loader
        dataset = TimeSeriesDataset(X, np.zeros(X.shape[0]))  # Dummy labels
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        probabilities = []

        with torch.no_grad():
            for batch_x, batch_x_mark, _ in loader:
                batch_x = batch_x.to(self.device)
                batch_x_mark = batch_x_mark.to(self.device)
                outputs = self.model(batch_x, batch_x_mark)
                probabilities.extend(torch.softmax(outputs, dim=1).cpu().numpy())

        return np.array(probabilities)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the accuracy score on the given data.

        Args:
            X: Input features
            y: Ground truth labels

        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def get_model_size(self) -> Dict[str, Any]:
        """
        Calculate model size in different metrics.

        Returns:
            Dictionary with model size information
        """
        if self.model is None:
            raise ValueError("Model must be initialized to get its size")

        param_size = sum(
            p.nelement() * p.element_size() for p in self.model.parameters()
        )
        buffer_size = sum(b.nelement() * b.element_size() for b in self.model.buffers())
        size_all_mb = (param_size + buffer_size) / 1024**2

        return {
            "num_params": sum(p.nelement() for p in self.model.parameters()),
            "size_in_mb": size_all_mb,
        }

    def summary(self) -> str:
        """
        Get a summary of the model.

        Returns:
            String summary of the model
        """
        if self.model is None:
            return "Model not initialized"

        model_size = self.get_model_size()

        return (
            f"TimesNet Model for Classification\n"
            f"Sequence Length: {self.seq_len}\n"
            f"Input Features: {self.enc_in}\n"
            f"Number of Classes: {self.num_class}\n"
            f"Model Dimension: {self.d_model}\n"
            f"Feed-forward Dimension: {self.d_ff}\n"
            f"Encoder Layers: {self.e_layers}\n"
            f"Top-k Frequencies: {self.top_k}\n"
            f"Number of Kernels: {self.num_kernels}\n"
            f"Total Parameters: {model_size['num_params']:,}\n"
            f"Model Size: {model_size['size_in_mb']:.2f} MB\n"
            f"Device: {self.device}"
        )

    def save(self, path: str) -> None:
        """
        Save the model to disk.

        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model must be trained before saving")

        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        """
        Load the model from disk.

        Args:
            path: Path to load the model from
        """
        if self.model is None:
            raise ValueError("Model must be initialized before loading weights")

        self.model.load_state_dict(torch.load(path, map_location=self.device))
