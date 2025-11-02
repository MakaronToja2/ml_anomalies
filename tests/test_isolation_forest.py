"""
Unit tests for Isolation Forest wrapper
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.algorithms.isolation_forest import IsolationForest


class TestIsolationForest:
    """Test cases for Isolation Forest"""

    def test_simple_outlier_2d(self):
        """Test IF on simple 2D data with one clear outlier"""
        np.random.seed(42)

        # Create data: cluster + 1 outlier
        X_inliers = np.random.randn(50, 2) * 0.5
        X_outlier = np.array([[5, 5]])
        X = np.vstack([X_inliers, X_outlier])

        iforest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
        labels = iforest.fit_predict(X)

        # Should detect at least the clear outlier
        assert labels[-1] == 1, "Clear outlier should be detected"

    def test_no_outliers_low_contamination(self):
        """Test IF on uniform data with low contamination"""
        np.random.seed(42)

        # Create uniform data
        X = np.random.randn(100, 2)

        iforest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
        labels = iforest.fit_predict(X)

        # Very few should be labeled as outliers
        assert np.sum(labels) <= 5, "Should detect very few outliers in uniform data"

    def test_fit_predict_consistency(self):
        """Test that fit + predict gives same results as fit_predict"""
        np.random.seed(42)
        X = np.random.randn(50, 3)

        iforest1 = IsolationForest(n_estimators=50, random_state=42)
        labels1 = iforest1.fit_predict(X)

        iforest2 = IsolationForest(n_estimators=50, random_state=42)
        iforest2.fit(X)
        labels2 = iforest2.predict(X)

        assert_array_almost_equal(labels1, labels2)

    def test_score_samples(self):
        """Test score_samples method"""
        np.random.seed(42)

        X_inliers = np.random.randn(50, 2)
        X_outlier = np.array([[10, 10]])
        X = np.vstack([X_inliers, X_outlier])

        iforest = IsolationForest(n_estimators=100, random_state=42)
        iforest.fit(X)

        scores = iforest.score_samples(X)

        # Outlier should have higher score (more anomalous)
        assert scores[-1] > np.mean(scores[:-1])

    def test_anomaly_scores_attribute(self):
        """Test that anomaly_scores_ is set after fit"""
        np.random.seed(42)
        X = np.random.randn(30, 2)

        iforest = IsolationForest(random_state=42)
        iforest.fit(X)

        assert iforest.anomaly_scores_ is not None
        assert len(iforest.anomaly_scores_) == len(X)

    def test_contamination_parameter(self):
        """Test that contamination affects number of detected outliers"""
        np.random.seed(42)
        X = np.random.randn(100, 2)

        iforest1 = IsolationForest(contamination=0.05, random_state=42)
        labels1 = iforest1.fit_predict(X)

        iforest2 = IsolationForest(contamination=0.20, random_state=42)
        labels2 = iforest2.fit_predict(X)

        # Higher contamination should detect more outliers
        assert np.sum(labels2) > np.sum(labels1)

    def test_n_estimators_parameter(self):
        """Test different numbers of estimators"""
        np.random.seed(42)
        X = np.random.randn(50, 2)

        # Should work with different n_estimators
        for n_est in [10, 50, 100, 200]:
            iforest = IsolationForest(n_estimators=n_est, random_state=42)
            iforest.fit(X)
            assert iforest.model_.n_estimators == n_est

    def test_1d_data(self):
        """Test IF on 1D data"""
        X = np.array([[1], [2], [3], [4], [100]])

        iforest = IsolationForest(contamination=0.2, random_state=42)
        labels = iforest.fit_predict(X)

        # Should detect the clear outlier
        assert labels[-1] == 1

    def test_high_dimensional_data(self):
        """Test IF on high-dimensional data"""
        np.random.seed(42)
        X = np.random.randn(100, 50)

        iforest = IsolationForest(n_estimators=100, random_state=42)
        iforest.fit(X)

        # Should run without errors
        labels = iforest.predict(X)
        assert labels.shape == (100,)

    def test_deterministic_with_random_state(self):
        """Test that results are deterministic with random_state"""
        np.random.seed(42)
        X = np.random.randn(50, 3)

        iforest1 = IsolationForest(n_estimators=100, random_state=42)
        labels1 = iforest1.fit_predict(X)

        iforest2 = IsolationForest(n_estimators=100, random_state=42)
        labels2 = iforest2.fit_predict(X)

        assert_array_almost_equal(labels1, labels2)

    def test_decision_function(self):
        """Test decision_function method"""
        np.random.seed(42)
        X = np.random.randn(30, 2)

        iforest = IsolationForest(random_state=42)
        iforest.fit(X)

        # decision_function should return scores
        scores = iforest.decision_function(X)
        assert scores.shape == (30,)
        assert np.all(np.isfinite(scores))

    def test_get_set_params(self):
        """Test get_params and set_params methods"""
        iforest = IsolationForest(n_estimators=100, contamination=0.1)

        params = iforest.get_params()
        assert params['n_estimators'] == 100
        assert params['contamination'] == 0.1

        iforest.set_params(n_estimators=200)
        assert iforest.n_estimators == 200

    def test_n_jobs_parallel(self):
        """Test that n_jobs parameter is accepted"""
        np.random.seed(42)
        X = np.random.randn(100, 5)

        # Should work with n_jobs=-1 (all cores)
        iforest = IsolationForest(n_estimators=100, n_jobs=-1, random_state=42)
        iforest.fit(X)
        labels = iforest.predict(X)

        assert labels.shape == (100,)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
