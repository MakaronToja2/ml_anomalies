"""
Autoencoder for Anomaly Detection (PyTorch)

A simple Multi-Layer Perceptron (MLP) autoencoder for detecting anomalies
based on reconstruction error.

Reference: Goodfellow, I., Bengio, Y., & Courville, A. (2016).
Deep Learning. MIT Press.
"""

import numpy as np
from typing import Optional, Tuple
from numpy.typing import ArrayLike
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class AutoencoderNet(nn.Module):
    """
    PyTorch neural network for autoencoder.

    Parameters
    ----------
    input_dim : int
        Number of input features.

    encoding_dim : int
        Dimension of the encoding (bottleneck) layer.

    hidden_dims : list of int, optional
        Dimensions of hidden layers between input and encoding.
        Default: [128, 64] for encoder, mirrored for decoder.
    """

    def __init__(
        self,
        input_dim: int,
        encoding_dim: int = 32,
        hidden_dims: Optional[list] = None
    ):
        super(AutoencoderNet, self).__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        # Encoder layers
        encoder_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim

        encoder_layers.append(nn.Linear(prev_dim, encoding_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder layers (mirror of encoder)
        decoder_layers = []
        prev_dim = encoding_dim

        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        encoded : torch.Tensor
            Encoded representation.

        decoded : torch.Tensor
            Reconstructed output.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class Autoencoder:
    """
    Autoencoder for anomaly detection.

    Anomalies are detected based on high reconstruction error.

    Parameters
    ----------
    encoding_dim : int, default=32
        Dimension of the encoding (bottleneck) layer.

    hidden_dims : list of int, optional
        Dimensions of hidden layers. Default: [128, 64].

    learning_rate : float, default=0.001
        Learning rate for optimizer.

    batch_size : int, default=32
        Batch size for training.

    epochs : int, default=50
        Number of training epochs.

    contamination : float, default=0.1
        Expected proportion of outliers (for setting threshold).

    device : str, optional
        Device to use ('cpu' or 'cuda'). If None, auto-detect.

    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    model_ : AutoencoderNet
        The PyTorch autoencoder model.

    threshold_ : float
        Threshold for anomaly detection.

    training_losses_ : list
        Training losses per epoch.

    Examples
    --------
    >>> from src.algorithms import Autoencoder
    >>> import numpy as np
    >>> X = np.random.randn(1000, 20)
    >>> autoenc = Autoencoder(encoding_dim=10, epochs=50)
    >>> autoenc.fit(X)
    >>> labels = autoenc.predict(X)
    >>> print(f"Detected {np.sum(labels)} outliers")
    """

    def __init__(
        self,
        encoding_dim: int = 32,
        hidden_dims: Optional[list] = None,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 50,
        contamination: float = 0.1,
        device: Optional[str] = None,
        random_state: Optional[int] = None,
        verbose: bool = True
    ):
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else [128, 64]
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.contamination = contamination
        self.random_state = random_state
        self.verbose = verbose

        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Set random seed
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        # Model attributes
        self.model_ = None
        self.threshold_ = None
        self.training_losses_ = []
        self.input_dim_ = None

    def fit(self, X: ArrayLike, validation_split: float = 0.1) -> 'Autoencoder':
        """
        Fit the autoencoder model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        validation_split : float, default=0.1
            Fraction of data to use for validation.

        Returns
        -------
        self : Autoencoder
            Fitted estimator.
        """
        X = np.asarray(X, dtype=np.float32)
        n_samples, n_features = X.shape
        self.input_dim_ = n_features

        # Split into train and validation
        n_val = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        X_train = X[train_indices]
        X_val = X[val_indices] if n_val > 0 else None

        # Create DataLoader
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(X_train)  # Target is same as input
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        # Initialize model
        self.model_ = AutoencoderNet(
            input_dim=n_features,
            encoding_dim=self.encoding_dim,
            hidden_dims=self.hidden_dims
        ).to(self.device)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)

        # Training loop
        self.training_losses_ = []

        for epoch in range(self.epochs):
            self.model_.train()
            train_loss = 0.0

            for batch_data, batch_target in train_loader:
                batch_data = batch_data.to(self.device)
                batch_target = batch_target.to(self.device)

                # Forward pass
                _, reconstructed = self.model_(batch_data)

                # Compute loss
                loss = criterion(reconstructed, batch_target)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            self.training_losses_.append(avg_train_loss)

            # Validation
            if X_val is not None and (epoch + 1) % 10 == 0:
                val_loss = self._validate(X_val, criterion)
                if self.verbose:
                    print(f"Epoch [{epoch+1}/{self.epochs}], "
                          f"Train Loss: {avg_train_loss:.6f}, "
                          f"Val Loss: {val_loss:.6f}")
            elif self.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], "
                      f"Train Loss: {avg_train_loss:.6f}")

        # Set threshold based on training data
        self._set_threshold(X)

        return self

    def _validate(self, X_val: np.ndarray, criterion: nn.Module) -> float:
        """Compute validation loss"""
        self.model_.eval()

        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            _, reconstructed = self.model_(X_val_tensor)
            val_loss = criterion(reconstructed, X_val_tensor).item()

        return val_loss

    def _set_threshold(self, X: np.ndarray) -> None:
        """Set anomaly detection threshold based on contamination"""
        errors = self.reconstruction_error(X)
        self.threshold_ = np.percentile(errors, (1 - self.contamination) * 100)

    def reconstruction_error(self, X: ArrayLike) -> np.ndarray:
        """
        Compute reconstruction error for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        errors : ndarray of shape (n_samples,)
            Reconstruction error for each sample.
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before computing reconstruction error")

        X = np.asarray(X, dtype=np.float32)

        self.model_.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            _, reconstructed = self.model_(X_tensor)

            # Compute MSE per sample
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            errors = errors.cpu().numpy()

        return errors

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict if samples are outliers.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to classify.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Binary labels: 1 for outliers, 0 for inliers.
        """
        if self.threshold_ is None:
            raise ValueError("Model must be fitted before calling predict")

        errors = self.reconstruction_error(X)
        labels = (errors > self.threshold_).astype(int)

        return labels

    def fit_predict(self, X: ArrayLike) -> np.ndarray:
        """
        Fit the model and predict outliers.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Binary labels: 1 for outliers, 0 for inliers.
        """
        self.fit(X)
        return self.predict(X)

    def score_samples(self, X: ArrayLike) -> np.ndarray:
        """
        Compute anomaly scores for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Anomaly scores (reconstruction errors).
        """
        return self.reconstruction_error(X)

    def encode(self, X: ArrayLike) -> np.ndarray:
        """
        Encode samples to latent space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to encode.

        Returns
        -------
        encoded : ndarray of shape (n_samples, encoding_dim)
            Encoded representations.
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before encoding")

        X = np.asarray(X, dtype=np.float32)

        self.model_.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            encoded, _ = self.model_(X_tensor)
            encoded = encoded.cpu().numpy()

        return encoded

    def decode(self, encoded: ArrayLike) -> np.ndarray:
        """
        Decode samples from latent space.

        Parameters
        ----------
        encoded : array-like of shape (n_samples, encoding_dim)
            Encoded representations.

        Returns
        -------
        decoded : ndarray of shape (n_samples, input_dim)
            Decoded (reconstructed) samples.
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before decoding")

        encoded = np.asarray(encoded, dtype=np.float32)

        self.model_.eval()

        with torch.no_grad():
            encoded_tensor = torch.FloatTensor(encoded).to(self.device)
            decoded = self.model_.decoder(encoded_tensor)
            decoded = decoded.cpu().numpy()

        return decoded
