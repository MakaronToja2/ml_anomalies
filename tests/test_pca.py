"""
Unit tests for PCA-based Anomaly Detection
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.algorithms.pca_anomaly import PCAAnomaly


class TestPCAAnomaly:
    """Test cases for PCA anomaly detection"""

    def test_simple_reconstruction_error(self):
        """Test PCA reconstruction error on simple 2D data"""
        # Create data along x-axis
        X_inliers = np.column_stack([
            np.linspace(0, 10, 20),
            np.random.randn(20) * 0.1
        ])
        # Outlier perpendicular to the main axis
        X_outlier = np.array([[5, 5]])
        X = np.vstack([X_inliers, X_outlier])

        pca = PCAAnomaly(n_components=1, method='reconstruction')
        pca.fit(X)

        errors = pca.reconstruction_error(X)

        # Outlier should have higher reconstruction error than most inliers
        assert errors[-1] > np.percentile(errors[:-1], 75)

    def test_mahalanobis_distance(self):
        """Test PCA with Mahalanobis distance"""
        np.random.seed(42)

        # Generate elongated cluster
        X = np.random.randn(100, 2)
        X[:, 0] *= 3  # Stretch along x-axis

        # Add outlier
        X = np.vstack([X, [[10, 10]]])

        pca = PCAAnomaly(n_components=2, method='mahalanobis')
        pca.fit(X)

        labels = pca.predict(X)

        # Should identify outlier
        assert labels[-1] == 1

    def test_explained_variance(self):
        """Test explained variance calculation"""
        np.random.seed(42)
        X = np.random.randn(100, 5)

        pca = PCAAnomaly(n_components=5)
        pca.fit(X)

        # Explained variance should sum to 1 (approximately)
        total_variance = np.sum(pca.explained_variance_ratio_)
        assert_allclose(total_variance, 1.0, rtol=1e-5)

        # Variance should be in descending order
        assert np.all(np.diff(pca.explained_variance_ratio_) <= 0)

    def test_variance_threshold(self):
        """Test selecting components by variance threshold"""
        np.random.seed(42)

        # Create data with clear principal components
        # First component has most variance
        X = np.random.randn(100, 1) * 10
        # Add small variance in other dimensions
        X = np.hstack([X, np.random.randn(100, 4) * 0.1])

        pca = PCAAnomaly(n_components=0.95)  # Keep 95% of variance
        pca.fit(X)

        # Should select components to meet variance threshold
        assert pca.n_components_actual_ <= 5

        # Should explain at least 95% variance
        assert np.sum(pca.explained_variance_ratio_) >= 0.95

    def test_transform_inverse_transform(self):
        """Test transform and inverse_transform are approximate inverses"""
        np.random.seed(42)
        X = np.random.randn(20, 5)

        pca = PCAAnomaly(n_components=5)
        pca.fit(X)

        # Transform and inverse transform
        X_transformed = pca.transform(X)
        X_reconstructed = pca.inverse_transform(X_transformed)

        # Should get back original data (approximately)
        assert_allclose(X, X_reconstructed, rtol=1e-5)

    def test_dimensionality_reduction(self):
        """Test dimensionality reduction"""
        np.random.seed(42)
        X = np.random.randn(50, 10)

        pca = PCAAnomaly(n_components=3)
        pca.fit(X)

        X_transformed = pca.transform(X)

        # Should reduce dimensions
        assert X_transformed.shape == (50, 3)

    def test_reconstruction_error_with_reduction(self):
        """Test reconstruction error when reducing dimensions"""
        np.random.seed(42)
        X = np.random.randn(50, 10)

        pca = PCAAnomaly(n_components=5)
        pca.fit(X)

        errors = pca.reconstruction_error(X)

        # All errors should be non-negative
        assert np.all(errors >= 0)

        # Errors should be positive (we're losing information)
        assert np.mean(errors) > 0

    def test_fit_predict(self):
        """Test fit_predict method"""
        X = np.array([
            [1, 1],
            [2, 2],
            [3, 3],
            [10, 1]  # Outlier
        ])

        pca = PCAAnomaly(n_components=1, contamination=0.25)
        labels = pca.fit_predict(X)

        # Should identify one outlier
        assert np.sum(labels) == 1

    def test_contamination_parameter(self):
        """Test that contamination parameter affects threshold"""
        np.random.seed(42)
        X = np.random.randn(100, 2)

        pca1 = PCAAnomaly(contamination=0.1)
        pca1.fit(X)
        labels1 = pca1.predict(X)

        pca2 = PCAAnomaly(contamination=0.3)
        pca2.fit(X)
        labels2 = pca2.predict(X)

        # Higher contamination should identify more outliers
        assert np.sum(labels2) > np.sum(labels1)

    def test_standardization(self):
        """Test that data is properly standardized"""
        X = np.array([
            [1, 100],
            [2, 200],
            [3, 300],
            [4, 400]
        ])

        pca = PCAAnomaly(n_components=2)
        pca.fit(X)

        # Mean should be stored
        assert pca.mean_ is not None
        assert_allclose(pca.mean_, np.mean(X, axis=0))

        # Std should be stored
        assert pca.std_ is not None
        assert_allclose(pca.std_, np.std(X, axis=0))

    def test_perfect_line_1d_reduction(self):
        """Test PCA on data that lies perfectly on a line"""
        # Points on line y = 2x
        X = np.array([
            [1, 2],
            [2, 4],
            [3, 6],
            [4, 8]
        ])

        pca = PCAAnomaly(n_components=1)
        pca.fit(X)

        # With 1 component, should explain almost all variance
        assert pca.explained_variance_ratio_[0] > 0.99

        # Reconstruction error should be very small
        errors = pca.reconstruction_error(X)
        assert np.all(errors < 1e-10)

    def test_different_methods(self):
        """Test that different methods give different results"""
        np.random.seed(42)
        X = np.random.randn(50, 3)

        pca_rec = PCAAnomaly(n_components=2, method='reconstruction')
        pca_rec.fit(X)
        scores_rec = pca_rec.score_samples(X)

        pca_maha = PCAAnomaly(n_components=2, method='mahalanobis')
        pca_maha.fit(X)
        scores_maha = pca_maha.score_samples(X)

        # Scores should be different
        assert not np.allclose(scores_rec, scores_maha)

    def test_sklearn_comparison(self):
        """Compare PCA decomposition with sklearn"""
        try:
            from sklearn.decomposition import PCA as SklearnPCA
        except ImportError:
            pytest.skip("sklearn not installed")

        np.random.seed(42)
        X = np.random.randn(50, 5)

        # Our implementation
        our_pca = PCAAnomaly(n_components=3)
        our_pca.fit(X)

        # sklearn implementation
        sklearn_pca = SklearnPCA(n_components=3)
        sklearn_pca.fit(X)

        # Total explained variance should be similar
        # Note: exact values may differ due to covariance calculation differences
        our_total_var = np.sum(our_pca.explained_variance_ratio_)
        sklearn_total_var = np.sum(sklearn_pca.explained_variance_ratio_)

        assert_allclose(our_total_var, sklearn_total_var, rtol=0.05)

        # Order of variances should be descending
        assert np.all(np.diff(our_pca.explained_variance_) <= 1e-10)

    def test_invalid_n_components(self):
        """Test invalid n_components parameter"""
        X = np.random.randn(20, 5)

        # Invalid type
        pca = PCAAnomaly(n_components="invalid")
        with pytest.raises(ValueError):
            pca.fit(X)

    def test_predict_without_fit(self):
        """Test that predict raises error if not fitted"""
        pca = PCAAnomaly()

        X = np.random.randn(10, 2)

        with pytest.raises(ValueError, match="must be fitted"):
            pca.predict(X)

    def test_transform_without_fit(self):
        """Test that transform raises error if not fitted"""
        pca = PCAAnomaly()

        X = np.random.randn(10, 2)

        with pytest.raises(ValueError, match="must be fitted"):
            pca.transform(X)

    def test_1d_data(self):
        """Test PCA on 1D data"""
        X = np.array([[1], [2], [3], [4], [100]])

        pca = PCAAnomaly(n_components=1, method='reconstruction')
        pca.fit(X)

        # Should work without errors
        scores = pca.score_samples(X)
        assert scores.shape == (5,)

    def test_high_dimensional_data(self):
        """Test PCA on high-dimensional data"""
        np.random.seed(42)
        X = np.random.randn(100, 50)

        pca = PCAAnomaly(n_components=10)
        pca.fit(X)

        # Should work without errors
        assert pca.components_.shape == (10, 50)

    def test_deterministic(self):
        """Test that PCA is deterministic"""
        np.random.seed(42)
        X = np.random.randn(50, 5)

        pca1 = PCAAnomaly(n_components=3)
        pca1.fit(X)
        scores1 = pca1.score_samples(X)

        pca2 = PCAAnomaly(n_components=3)
        pca2.fit(X)
        scores2 = pca2.score_samples(X)

        assert_array_almost_equal(scores1, scores2)

    def test_all_components(self):
        """Test keeping all components (n_components=None)"""
        X = np.random.randn(20, 5)

        pca = PCAAnomaly(n_components=None)
        pca.fit(X)

        # Should keep all components
        assert pca.n_components_actual_ == 5

    def test_gaussian_blob_with_outlier(self):
        """Test on Gaussian blob with clear outlier"""
        np.random.seed(42)

        # Generate Gaussian cluster
        X_inliers = np.random.randn(100, 2) * 0.5

        # Add clear outlier
        X = np.vstack([X_inliers, [[5, 5]]])

        pca = PCAAnomaly(n_components=2, method='reconstruction', contamination=0.01)
        labels = pca.fit_predict(X)

        # Should identify the outlier
        assert labels[-1] == 1

        # Most inliers should be labeled as normal
        assert np.sum(labels[:-1]) <= 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
