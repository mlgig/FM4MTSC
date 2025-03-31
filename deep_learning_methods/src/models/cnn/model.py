"""
Custom CNN Model implementation for time series classification.
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


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
        return self.X[idx], self.y[idx]


class CNNModel:
    """
    Custom CNN model for time series classification.
    """

    def __init__(
        self,
        num_channels: Optional[int] = None,
        seq_length: Optional[int] = None,
        num_classes: Optional[int] = None,
        learning_rate: float = 5e-5,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = 10,
        min_delta: float = 0.001,
        device: Optional[str] = None,
        model_path: str = "results/cnn_model.pth",
    ):
        """
        Initialize the CNN model.

        Args:
            num_channels: Number of input channels
            seq_length: Length of input sequence
            num_classes: Number of output classes
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
            epochs: Maximum number of training epochs
            patience: Number of epochs to wait for improvement before early stopping
            min_delta: Minimum change in validation accuracy to qualify as improvement
            device: Device to use for training ('cpu', 'cuda', or 'mps')
            model_path: Path to save the best model
        """
        self.num_channels = num_channels
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.min_delta = min_delta
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

        # Initialize model, criterion, and optimizer
        self.model = None
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None

        # Create model if all required parameters are provided
        if (
            num_channels is not None
            and seq_length is not None
            and num_classes is not None
        ):
            self._build_model()

    def _build_model(self):
        """Build the CNN model with current parameters."""
        self.model = MultivariateTimeSeriesCNN(
            self.num_channels, self.seq_length, self.num_classes
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        verbose: int = 1,
    ) -> Dict[str, Any]:
        """
        Train the model on the provided data.

        Args:
            X_train: Training features
            y_train: Training labels
            validation_data: Optional tuple of (X_val, y_val)
            verbose: Verbosity level

        Returns:
            Dictionary containing training history
        """
        # Infer dimensions if not provided
        if self.num_channels is None or self.seq_length is None:
            # Check if data is transposed correctly (batch, channels, timesteps)
            if len(X_train.shape) != 3:
                raise ValueError(
                    "Input data must be 3D with shape (batch, channels, timesteps)"
                )

            self.num_channels = X_train.shape[1]
            self.seq_length = X_train.shape[2]

        if self.num_classes is None:
            self.num_classes = len(np.unique(y_train))

        # Build model if not already built
        if self.model is None:
            self._build_model()

        # Create training dataset and loader
        train_dataset = TimeSeriesDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        # Create validation dataset and loader if validation data is provided
        if validation_data is not None:
            X_val, y_val = validation_data
            val_dataset = TimeSeriesDataset(X_val, y_val)
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )
        else:
            # Split training data to create validation set
            from sklearn.model_selection import train_test_split

            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )

            train_dataset = TimeSeriesDataset(X_train_split, y_train_split)
            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            )

            val_dataset = TimeSeriesDataset(X_val, y_val)
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )

        # Train the model
        history = self._train_model_internal(
            train_loader=train_loader, val_loader=val_loader, verbose=verbose
        )

        # Load the best model weights
        if os.path.exists(self.model_path):
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )

        return history

    def _train_model_internal(
        self, train_loader: DataLoader, val_loader: DataLoader, verbose: int = 1
    ) -> Dict[str, Any]:
        """
        Internal method to train the model with early stopping.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            verbose: Verbosity level

        Returns:
            Dictionary containing training history
        """
        # Ensure model directory exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        # Calculate and log model size
        model_size = self._count_parameters()
        if verbose > 0:
            print(f"Model Size: {model_size:,} parameters")

        best_val_acc = 0
        epochs_without_improvement = 0
        training_history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "epoch_times": [],
        }

        total_start_time = time.time()

        for epoch in range(self.epochs):
            epoch_start_time = time.time()

            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

            # Validation phase
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()

            # Calculate metrics
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            epoch_time = time.time() - epoch_start_time

            # Store history
            training_history["train_loss"].append(train_loss / len(train_loader))
            training_history["train_acc"].append(train_acc)
            training_history["val_loss"].append(val_loss / len(val_loader))
            training_history["val_acc"].append(val_acc)
            training_history["epoch_times"].append(epoch_time)

            # Print progress
            if verbose > 0:
                print(f"Epoch [{epoch + 1}/{self.epochs}] - Time: {epoch_time:.2f}s")
                print(
                    f"Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {train_acc:.2f}%"
                )
                print(
                    f"Val Loss: {val_loss / len(val_loader):.4f}, Val Acc: {val_acc:.2f}%"
                )

            # Early stopping logic
            if val_acc > best_val_acc + self.min_delta:
                best_val_acc = val_acc
                epochs_without_improvement = 0
                torch.save(self.model.state_dict(), self.model_path)
                if verbose > 0:
                    print(
                        f"New best model saved with validation accuracy: {val_acc:.2f}%"
                    )
            else:
                epochs_without_improvement += 1
                if verbose > 0:
                    print(
                        f"Epochs without improvement: {epochs_without_improvement}/{self.patience}"
                    )

            if epochs_without_improvement >= self.patience:
                if verbose > 0:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        total_training_time = time.time() - total_start_time

        # Add metadata to history
        training_history["total_time"] = total_training_time
        training_history["model_size"] = model_size
        training_history["best_val_acc"] = best_val_acc
        training_history["epochs_completed"] = epoch + 1

        if verbose > 0:
            print(f"\nTraining completed in {total_training_time:.2f} seconds")
            print(f"Best validation accuracy: {best_val_acc:.2f}%")

        return training_history

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
            for batch_X, _ in loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
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
            for batch_X, _ in loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
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

    def _count_parameters(self) -> int:
        """
        Count the number of trainable parameters in the model.

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def summary(self) -> str:
        """
        Get a summary of the model.

        Returns:
            String summary of the model
        """
        if self.model is None:
            return "Model not initialized"

        num_params = self._count_parameters()

        return (
            f"MultivariateTimeSeriesCNN\n"
            f"Input Channels: {self.num_channels}\n"
            f"Sequence Length: {self.seq_length}\n"
            f"Number of Classes: {self.num_classes}\n"
            f"Trainable Parameters: {num_params:,}\n"
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


class MultivariateTimeSeriesCNN(nn.Module):
    """
    PyTorch CNN network for multivariate time series classification.
    """

    def __init__(self, num_channels: int, seq_length: int, num_classes: int):
        """
        Initialize the CNN network.

        Args:
            num_channels: Number of input channels (features)
            seq_length: Length of input sequence
            num_classes: Number of output classes
        """
        super(MultivariateTimeSeriesCNN, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=num_channels, out_channels=64, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)

        self.flat_size = 128 * (seq_length // 4)

        self.fc1 = nn.Linear(self.flat_size, 256)
        self.fc_relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor with shape (batch_size, channels, sequence_length)

        Returns:
            Output tensor with shape (batch_size, num_classes)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc_relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
