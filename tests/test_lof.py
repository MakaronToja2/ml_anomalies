"""
Unit tests for Local Outlier Factor (LOF) algorithm
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.algorithms.lof import LOF


class TestLOF:
    """Test cases for LOF algorithm"""

    def test_simple_outlier_2d(self):
        """Test LOF on simple 2D data with one clear outlier"""
        # Create data: 4 points in cluster + 1 outlier
        X = np.array([
            [0, 0],
            [1, 1],
            [1, 0],
            [0, 1],
            [10, 10]  # Clear outlier
        ])

        lof = LOF(n_neighbors=2)
        scores = lof.fit_predict(X)

        # Outlier should have significantly higher LOF score
        assert scores[-1] > 1.5, "Outlier should have LOF > 1.5"
        assert np.all(scores[:4] < 1.5), "Inliers should have LOF < 1.5"

    def test_no_outliers(self):
        """Test LOF on data with no outliers (uniform grid)"""
        # Create uniform grid
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5)
        X = np.array([[xi, yi] for xi in x for yi in y])

        lof = LOF(n_neighbors=4)
        scores = lof.fit_predict(X)

        # All points should have LOF close to 1
        assert_allclose(scores, 1.0, atol=0.3)

    def test_gaussian_with_outliers(self):
        """Test LOF on Gaussian data with added outliers"""
        np.random.seed(42)

        # Generate Gaussian cluster
        X_inliers = np.random.randn(100, 2)

        # Add outliers far from cluster
        X_outliers = np.random.randn(5, 2) * 5 + 10

        X = np.vstack([X_inliers, X_outliers])

        lof = LOF(n_neighbors=10)
        scores = lof.fit_predict(X)

        # Outliers should generally have higher scores
        outlier_scores = scores[-5:]
        inlier_scores = scores[:-5]

        assert np.mean(outlier_scores) > np.mean(inlier_scores)

    def test_fit_predict_consistency(self):
        """Test that fit + predict gives same results as fit_predict"""
        np.random.seed(42)
        X = np.random.randn(50, 3)

        lof1 = LOF(n_neighbors=5)
        scores1 = lof1.fit_predict(X)

        lof2 = LOF(n_neighbors=5)
        lof2.fit(X)
        scores2 = lof2.lof_scores_

        assert_array_almost_equal(scores1, scores2)

    def test_negative_outlier_factor(self):
        """Test negative_outlier_factor_ attribute"""
        X = np.array([
            [0, 0],
            [1, 1],
            [1, 0],
            [0, 1],
            [10, 10]
        ])

        lof = LOF(n_neighbors=2)
        lof.fit(X)

        # negative_outlier_factor_ should be negative of lof_scores_
        assert_array_almost_equal(
            lof.negative_outlier_factor_,
            -lof.lof_scores_
        )

    def test_different_k_values(self):
        """Test LOF with different k (n_neighbors) values"""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        X = np.vstack([X, [[10, 10]]])  # Add one outlier

        scores_k5 = LOF(n_neighbors=5).fit_predict(X)
        scores_k20 = LOF(n_neighbors=20).fit_predict(X)

        # Scores should differ with different k
        assert not np.allclose(scores_k5, scores_k20)

        # Both should identify the outlier
        assert scores_k5[-1] > 1.5
        assert scores_k20[-1] > 1.5

    def test_predict_threshold(self):
        """Test predict method with threshold"""
        X = np.array([
            [0, 0],
            [1, 1],
            [1, 0],
            [0, 1],
            [10, 10]
        ])

        lof = LOF(n_neighbors=2)
        lof.fit(X)

        labels = lof.predict(X, threshold=1.5)

        # Should label last point as outlier (1)
        assert labels[-1] == 1
        # Should label first points as inliers (0)
        assert np.all(labels[:4] == 0)

    def test_score_samples(self):
        """Test score_samples method"""
        X = np.array([
            [0, 0],
            [1, 1],
            [1, 0],
            [0, 1]
        ])

        lof = LOF(n_neighbors=2)
        lof.fit(X)

        # Test on new data
        X_new = np.array([[0.5, 0.5], [20, 20]])
        scores = lof.score_samples(X_new)

        # Score should be more negative for outlier
        assert scores[1] < scores[0]

    def test_invalid_k(self):
        """Test that invalid n_neighbors raises error"""
        X = np.array([[0, 0], [1, 1], [2, 2]])

        lof = LOF(n_neighbors=5)  # k > n_samples

        with pytest.raises(ValueError, match="n_neighbors.*must be less than"):
            lof.fit(X)

    def test_1d_data(self):
        """Test LOF on 1D data"""
        X = np.array([[0], [1], [2], [3], [4], [100]])

        lof = LOF(n_neighbors=2)
        scores = lof.fit_predict(X)

        # Last point should be outlier
        assert scores[-1] > scores[0]

    def test_high_dimensional_data(self):
        """Test LOF on high-dimensional data"""
        np.random.seed(42)
        X = np.random.randn(50, 10)

        lof = LOF(n_neighbors=5)
        scores = lof.fit_predict(X)

        # Should run without errors
        assert scores.shape == (50,)
        assert np.all(np.isfinite(scores))

    def test_sklearn_comparison(self):
        """Compare with sklearn LOF implementation"""
        try:
            from sklearn.neighbors import LocalOutlierFactor as SklearnLOF
        except ImportError:
            pytest.skip("sklearn not installed")

        np.random.seed(42)
        X = np.random.randn(30, 2)

        # Our implementation
        our_lof = LOF(n_neighbors=5)
        our_scores = our_lof.fit_predict(X)

        # sklearn implementation
        sklearn_lof = SklearnLOF(n_neighbors=5, novelty=False)
        sklearn_lof.fit(X)
        sklearn_scores = -sklearn_lof.negative_outlier_factor_

        # Scores should be similar (may not be identical due to implementation details)
        correlation = np.corrcoef(our_scores, sklearn_scores)[0, 1]
        assert correlation > 0.95, f"Correlation with sklearn: {correlation}"

    def test_deterministic(self):
        """Test that LOF is deterministic"""
        np.random.seed(42)
        X = np.random.randn(30, 2)

        lof1 = LOF(n_neighbors=5)
        scores1 = lof1.fit_predict(X)

        lof2 = LOF(n_neighbors=5)
        scores2 = lof2.fit_predict(X)

        assert_array_almost_equal(scores1, scores2)

    def test_kdtree_vs_bruteforce(self):
        """Test that KD-Tree optimization produces same results as brute-force"""
        np.random.seed(123)
        X = np.random.randn(50, 3)

        # LOF with KD-Tree (default)
        lof_kdtree = LOF(n_neighbors=10, use_kdtree=True)
        scores_kdtree = lof_kdtree.fit_predict(X)

        # LOF with brute-force
        lof_brute = LOF(n_neighbors=10, use_kdtree=False)
        scores_brute = lof_brute.fit_predict(X)

        # Results should be nearly identical
        assert_allclose(scores_kdtree, scores_brute, rtol=1e-5, atol=1e-8)

    def test_kdtree_predict(self):
        """Test KD-Tree optimization with predict on new data"""
        np.random.seed(456)
        X_train = np.random.randn(40, 2)
        X_test = np.random.randn(10, 2)

        # With KD-Tree
        lof_kdtree = LOF(n_neighbors=5, use_kdtree=True)
        lof_kdtree.fit(X_train)
        predictions_kdtree = lof_kdtree.predict(X_test, threshold=1.5)

        # Without KD-Tree
        lof_brute = LOF(n_neighbors=5, use_kdtree=False)
        lof_brute.fit(X_train)
        predictions_brute = lof_brute.predict(X_test, threshold=1.5)

        # Predictions should be identical
        assert_array_almost_equal(predictions_kdtree, predictions_brute)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
