"""
Unit tests for Autoencoder
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Check if torch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from src.algorithms.autoencoder import Autoencoder, AutoencoderNet


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestAutoencoder:
    """Test cases for Autoencoder"""

    def test_basic_training(self):
        """Test basic autoencoder training"""
        np.random.seed(42)
        X = np.random.randn(100, 20).astype(np.float32)

        autoenc = Autoencoder(
            encoding_dim=10,
            hidden_dims=[64, 32],
            epochs=10,
            batch_size=16,
            random_state=42,
            verbose=False
        )

        autoenc.fit(X)

        # Should have training losses
        assert len(autoenc.training_losses_) == 10
        # Losses should decrease (roughly)
        assert autoenc.training_losses_[-1] < autoenc.training_losses_[0]

    def test_reconstruction_error(self):
        """Test reconstruction error computation"""
        np.random.seed(42)
        X = np.random.randn(50, 10).astype(np.float32)

        autoenc = Autoencoder(encoding_dim=5, epochs=20, verbose=False, random_state=42)
        autoenc.fit(X)

        errors = autoenc.reconstruction_error(X)

        # Should return error for each sample
        assert errors.shape == (50,)
        # All errors should be non-negative
        assert np.all(errors >= 0)

    def test_fit_predict(self):
        """Test fit_predict method"""
        np.random.seed(42)
        X_inliers = np.random.randn(100, 15).astype(np.float32)
        X_outliers = np.random.randn(10, 15).astype(np.float32) * 5 + 10
        X = np.vstack([X_inliers, X_outliers])

        autoenc = Autoencoder(
            encoding_dim=8,
            epochs=30,
            contamination=0.15,
            verbose=False,
            random_state=42
        )

        labels = autoenc.fit_predict(X)

        # Should return labels
        assert labels.shape == (110,)
        # Should detect some outliers
        assert np.sum(labels) > 0

    def test_encode_decode(self):
        """Test encode and decode methods"""
        np.random.seed(42)
        X = np.random.randn(30, 20).astype(np.float32)

        autoenc = Autoencoder(encoding_dim=10, epochs=20, verbose=False, random_state=42)
        autoenc.fit(X)

        # Encode
        encoded = autoenc.encode(X)
        assert encoded.shape == (30, 10)

        # Decode
        decoded = autoenc.decode(encoded)
        assert decoded.shape == (30, 20)

        # Encoded then decoded should be similar to reconstruction
        _, reconstructed = autoenc.model_(torch.FloatTensor(X).to(autoenc.device))
        reconstructed = reconstructed.cpu().detach().numpy()

        assert_allclose(decoded, reconstructed, rtol=1e-5)

    def test_threshold_setting(self):
        """Test that threshold is set correctly"""
        np.random.seed(42)
        X = np.random.randn(100, 10).astype(np.float32)

        autoenc = Autoencoder(contamination=0.1, epochs=10, verbose=False, random_state=42)
        autoenc.fit(X)

        # Threshold should be set
        assert autoenc.threshold_ is not None
        assert autoenc.threshold_ > 0

    def test_predict_without_fit(self):
        """Test that predict raises error if not fitted"""
        autoenc = Autoencoder()

        X = np.random.randn(10, 5).astype(np.float32)

        with pytest.raises(ValueError, match="must be fitted"):
            autoenc.predict(X)

    def test_different_architectures(self):
        """Test different encoder architectures"""
        np.random.seed(42)
        X = np.random.randn(50, 15).astype(np.float32)

        # Small network
        autoenc_small = Autoencoder(
            encoding_dim=5,
            hidden_dims=[32],
            epochs=10,
            verbose=False,
            random_state=42
        )
        autoenc_small.fit(X)

        # Large network
        autoenc_large = Autoencoder(
            encoding_dim=5,
            hidden_dims=[128, 64, 32],
            epochs=10,
            verbose=False,
            random_state=42
        )
        autoenc_large.fit(X)

        # Both should work
        assert autoenc_small.model_ is not None
        assert autoenc_large.model_ is not None

    def test_score_samples(self):
        """Test score_samples method"""
        np.random.seed(42)
        X = np.random.randn(40, 12).astype(np.float32)

        autoenc = Autoencoder(encoding_dim=6, epochs=15, verbose=False, random_state=42)
        autoenc.fit(X)

        scores = autoenc.score_samples(X)

        # Scores should be reconstruction errors
        errors = autoenc.reconstruction_error(X)
        assert_allclose(scores, errors)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestAutoencoderNet:
    """Test cases for AutoencoderNet model"""

    def test_forward_pass(self):
        """Test forward pass through network"""
        model = AutoencoderNet(input_dim=20, encoding_dim=10, hidden_dims=[64, 32])

        X = torch.randn(10, 20)
        encoded, decoded = model(X)

        assert encoded.shape == (10, 10)
        assert decoded.shape == (10, 20)

    def test_gradient_flow(self):
        """Test that gradients flow through network"""
        model = AutoencoderNet(input_dim=15, encoding_dim=8)

        X = torch.randn(5, 15, requires_grad=True)
        encoded, decoded = model(X)

        loss = torch.mean((X - decoded) ** 2)
        loss.backward()

        # Gradients should exist
        assert X.grad is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
