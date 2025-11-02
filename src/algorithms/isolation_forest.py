"""
Isolation Forest Wrapper for Anomaly Detection

This is a wrapper around sklearn's IsolationForest to maintain a consistent API
with our custom LOF and PCA implementations.

Based on: Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008).
Isolation Forest. 2008 Eighth IEEE International Conference on Data Mining.
"""

import numpy as np
from typing import Optional, Literal
from numpy.typing import ArrayLike
from sklearn.ensemble import IsolationForest as SklearnIsolationForest


class IsolationForest:
    """
    Isolation Forest for anomaly detection (sklearn wrapper).

    Isolation Forest isolates anomalies by randomly selecting a feature and
    then randomly selecting a split value between the maximum and minimum
    values of the selected feature. Anomalies are easier to isolate and
    require fewer splits.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of base estimators (trees) in the ensemble.

    max_samples : int or float, default='auto'
        The number of samples to draw to train each base estimator.
        - If int: draw max_samples samples.
        - If float: draw max_samples * n_samples samples.
        - If 'auto': max_samples=min(256, n_samples).

    contamination : float, default=0.1
        The proportion of outliers in the dataset. Used to define the threshold.

    max_features : int or float, default=1.0
        The number of features to draw to train each base estimator.

    random_state : int, optional
        Random seed for reproducibility.

    n_jobs : int, default=1
        Number of parallel jobs. -1 means using all processors.

    Attributes
    ----------
    model_ : SklearnIsolationForest
        The underlying sklearn Isolation Forest model.

    anomaly_scores_ : ndarray of shape (n_samples,)
        Anomaly scores for training samples (negative of decision_function).

    Examples
    --------
    >>> from src.algorithms import IsolationForest
    >>> import numpy as np
    >>> X = np.random.randn(100, 2)
    >>> X = np.vstack([X, [[5, 5]]])  # Add outlier
    >>> iforest = IsolationForest(n_estimators=100, contamination=0.1)
    >>> iforest.fit(X)
    >>> labels = iforest.predict(X)
    >>> print(f"Detected {np.sum(labels)} outliers")
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: str | int | float = 'auto',
        contamination: float = 0.1,
        max_features: int | float = 1.0,
        random_state: Optional[int] = None,
        n_jobs: int = 1
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs

        # Initialize sklearn model
        self.model_ = SklearnIsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs
        )

        self.anomaly_scores_ = None

    def fit(self, X: ArrayLike) -> 'IsolationForest':
        """
        Fit the Isolation Forest model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : IsolationForest
            Fitted estimator.
        """
        X = np.asarray(X, dtype=np.float64)

        # Fit the sklearn model
        self.model_.fit(X)

        # Store anomaly scores (negative of decision_function for consistency)
        # Higher scores = more anomalous
        self.anomaly_scores_ = -self.model_.decision_function(X)

        return self

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
        X = np.asarray(X, dtype=np.float64)

        # sklearn returns -1 for outliers, 1 for inliers
        # We convert to 0/1 format
        sklearn_labels = self.model_.predict(X)
        labels = (sklearn_labels == -1).astype(int)

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

        Lower scores indicate more anomalous samples (sklearn convention).
        We return negative to maintain consistency: higher = more anomalous.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Anomaly scores. Higher values indicate more anomalous samples.
        """
        X = np.asarray(X, dtype=np.float64)

        # decision_function: higher = more normal
        # We negate it so higher = more anomalous
        scores = -self.model_.decision_function(X)

        return scores

    def decision_function(self, X: ArrayLike) -> np.ndarray:
        """
        Compute decision function (sklearn compatible).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Decision function values. Higher = more normal.
        """
        X = np.asarray(X, dtype=np.float64)
        return self.model_.decision_function(X)

    def get_params(self) -> dict:
        """
        Get parameters of the model.

        Returns
        -------
        params : dict
            Model parameters.
        """
        return {
            'n_estimators': self.n_estimators,
            'max_samples': self.max_samples,
            'contamination': self.contamination,
            'max_features': self.max_features,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs
        }

    def set_params(self, **params) -> 'IsolationForest':
        """
        Set parameters of the model.

        Parameters
        ----------
        **params : dict
            Model parameters to set.

        Returns
        -------
        self : IsolationForest
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Reinitialize sklearn model with new parameters
        self.model_ = SklearnIsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )

        return self
