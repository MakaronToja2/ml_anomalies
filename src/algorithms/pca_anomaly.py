"""
PCA-based Anomaly Detection Implementation

Based on: Jolliffe, I. T., & Cadima, J. (2016).
Principal component analysis: a review and recent developments.
"""

import numpy as np
from typing import Optional, Literal
from numpy.typing import ArrayLike
from scipy import stats
from typing import Union


class PCAAnomaly:
    """
    Principal Component Analysis for anomaly detection.

    PCA can detect anomalies through reconstruction error or
    Mahalanobis distance in the principal component space.

    Parameters
    ----------
    n_components : int or float, default=None
        Number of components to keep:
        - If int: exact number of components
        - If float (0 < n_components < 1): select number of components
          such that the amount of variance explained is >= n_components
        - If None: keep all components

    method : {'reconstruction', 'mahalanobis'}, default='reconstruction'
        Method for anomaly detection:
        - 'reconstruction': Use reconstruction error
        - 'mahalanobis': Use Mahalanobis distance in PC space

    contamination : float, default=0.1
        Expected proportion of outliers in the dataset.
        Used to set threshold automatically.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Principal components (eigenvectors).

    explained_variance_ : ndarray of shape (n_components,)
        Variance explained by each component (eigenvalues).

    mean_ : ndarray of shape (n_features,)
        Mean of training data.

    std_ : ndarray of shape (n_features,)
        Standard deviation of training data.

    threshold_ : float
        Threshold for anomaly detection.

    Examples
    --------
    >>> from src.algorithms import PCAAnomaly
    >>> import numpy as np
    >>> X = np.array([[0, 0], [1, 1], [1, 0], [0, 1], [10, 10]])
    >>> pca = PCAAnomaly(n_components=1, method='reconstruction')
    >>> pca.fit(X)
    >>> scores = pca.score_samples(X)
    >>> print(scores[-1] < scores[0])  # Last point should have higher error
    True
    """

    def __init__(
        self,
        n_components: Optional[Union[int, float]] = None,
        method: Literal['reconstruction', 'mahalanobis'] = 'reconstruction',
        contamination: float = 0.1
    ):
        self.n_components = n_components
        self.method = method
        self.contamination = contamination

        # Attributes set during fit
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        self.std_ = None
        self.threshold_ = None
        self.n_components_actual_ = None

    def fit(self, X: ArrayLike) -> 'PCAAnomaly':
        """
        Fit the PCA model on training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : PCAAnomaly
            Fitted estimator.
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape

        # Step 1: Standardize data
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0) + 1e-10  # Avoid division by zero
        X_std = (X - self.mean_) / self.std_

        # Step 2: Compute covariance matrix
        # Cov = (1/n) * X^T * X (for standardized data)
        if n_features == 1:
            # Special case: 1D data
            cov_matrix = np.array([[np.var(X_std)]])
            eigenvalues = np.array([cov_matrix[0, 0]])
            eigenvectors = np.array([[1.0]])
        else:
            cov_matrix = np.cov(X_std, rowvar=False)

            # Step 3: Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Step 4: Sort by eigenvalues (descending order)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Step 5: Select number of components
        if self.n_components is None:
            self.n_components_actual_ = n_features
        elif isinstance(self.n_components, int):
            self.n_components_actual_ = min(self.n_components, n_features)
        elif isinstance(self.n_components, float):
            # Select components to explain desired variance
            total_variance = np.sum(eigenvalues)
            cumsum_variance = np.cumsum(eigenvalues) / total_variance
            self.n_components_actual_ = np.searchsorted(
                cumsum_variance, self.n_components
            ) + 1
        else:
            raise ValueError(
                f"n_components must be int, float, or None, got {type(self.n_components)}"
            )

        # Step 6: Store principal components and explained variance
        self.components_ = eigenvectors[:, :self.n_components_actual_].T
        self.explained_variance_ = eigenvalues[:self.n_components_actual_]

        # Calculate explained variance ratio
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance

        # Step 7: Set threshold based on training data
        self._set_threshold(X)

        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Transform data to principal component space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        if self.components_ is None:
            raise ValueError("Model must be fitted before calling transform")

        X = np.asarray(X, dtype=np.float64)

        # Standardize
        X_std = (X - self.mean_) / self.std_

        # Project onto principal components
        X_transformed = X_std @ self.components_.T

        return X_transformed

    def inverse_transform(self, X_transformed: ArrayLike) -> np.ndarray:
        """
        Transform data back to original space.

        Parameters
        ----------
        X_transformed : array-like of shape (n_samples, n_components)
            Data in principal component space.

        Returns
        -------
        X_reconstructed : ndarray of shape (n_samples, n_features)
            Reconstructed data in original space.
        """
        if self.components_ is None:
            raise ValueError("Model must be fitted before calling inverse_transform")

        X_transformed = np.asarray(X_transformed, dtype=np.float64)

        # Project back to original space
        X_std = X_transformed @ self.components_

        # Unstandardize
        X_reconstructed = X_std * self.std_ + self.mean_

        return X_reconstructed

    def reconstruction_error(self, X: ArrayLike) -> np.ndarray:
        """
        Compute reconstruction error for samples.

        Reconstruction error = ||x - x_reconstructed||^2

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        errors : ndarray of shape (n_samples,)
            Reconstruction error for each sample.
        """
        X = np.asarray(X, dtype=np.float64)

        # Transform to PC space and back
        X_transformed = self.transform(X)
        X_reconstructed = self.inverse_transform(X_transformed)

        # Compute squared error
        errors = np.sum((X - X_reconstructed) ** 2, axis=1)

        return errors

    def mahalanobis_distance(self, X: ArrayLike) -> np.ndarray:
        """
        Compute Mahalanobis distance in principal component space.

        Mahalanobis distance considers the variance along each PC.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        distances : ndarray of shape (n_samples,)
            Mahalanobis distance for each sample.
        """
        X = np.asarray(X, dtype=np.float64)

        # Transform to PC space
        X_transformed = self.transform(X)

        # Compute Mahalanobis distance
        # Since PCs are orthogonal, covariance matrix is diagonal
        # Mahalanobis = sqrt(sum((x_i / sqrt(lambda_i))^2))
        distances = np.sqrt(
            np.sum(
                (X_transformed ** 2) / (self.explained_variance_ + 1e-10),
                axis=1
            )
        )

        return distances

    def score_samples(self, X: ArrayLike) -> np.ndarray:
        """
        Compute anomaly scores for samples.

        Lower scores indicate more anomalous samples (sklearn convention).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Anomaly scores (negative for anomalies).
        """
        if self.components_ is None:
            raise ValueError("Model must be fitted before calling score_samples")

        X = np.asarray(X, dtype=np.float64)

        if self.method == 'reconstruction':
            errors = self.reconstruction_error(X)
            # Negative error so that outliers have lower scores
            return -errors
        elif self.method == 'mahalanobis':
            distances = self.mahalanobis_distance(X)
            return -distances
        else:
            raise ValueError(f"Unknown method: {self.method}")

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

        scores = self.score_samples(X)

        # Outliers have scores below threshold (more negative)
        return (scores < self.threshold_).astype(int)

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

    def _set_threshold(self, X: ArrayLike) -> None:
        """
        Set threshold for anomaly detection based on contamination level.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        """
        X = np.asarray(X, dtype=np.float64)

        # Compute scores for training data
        scores = self.score_samples(X)

        # Set threshold as the (contamination * 100)th percentile
        self.threshold_ = np.percentile(scores, self.contamination * 100)

    def get_explained_variance_ratio(self) -> np.ndarray:
        """
        Get the ratio of variance explained by each component.

        Returns
        -------
        explained_variance_ratio : ndarray of shape (n_components,)
            Variance explained by each component (as ratio of total variance).
        """
        if self.explained_variance_ratio_ is None:
            raise ValueError("Model must be fitted first")

        return self.explained_variance_ratio_

    def plot_explained_variance(self):
        """
        Plot cumulative explained variance (requires matplotlib).

        Returns
        -------
        fig, ax : matplotlib figure and axis
        """
        import matplotlib.pyplot as plt

        if self.explained_variance_ratio_ is None:
            raise ValueError("Model must be fitted first")

        cumsum_variance = np.cumsum(self.explained_variance_ratio_)

        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        # Individual variance
        ax[0].bar(
            range(1, len(self.explained_variance_ratio_) + 1),
            self.explained_variance_ratio_
        )
        ax[0].set_xlabel('Principal Component')
        ax[0].set_ylabel('Explained Variance Ratio')
        ax[0].set_title('Variance Explained by Each Component')
        ax[0].grid(True, alpha=0.3)

        # Cumulative variance
        ax[1].plot(
            range(1, len(cumsum_variance) + 1),
            cumsum_variance,
            marker='o'
        )
        ax[1].axhline(y=0.95, color='r', linestyle='--', label='95% variance')
        ax[1].set_xlabel('Number of Components')
        ax[1].set_ylabel('Cumulative Explained Variance')
        ax[1].set_title('Cumulative Explained Variance')
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, ax

    def qq_plot(self, X: ArrayLike):
        """
        Generate Q-Q plot for anomaly detection (requires matplotlib and scipy).

        Points far from the diagonal line are potential anomalies.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to analyze.

        Returns
        -------
        fig, ax : matplotlib figure and axis
        """
        import matplotlib.pyplot as plt

        X = np.asarray(X, dtype=np.float64)

        if self.method == 'mahalanobis':
            distances = self.mahalanobis_distance(X)

            fig, ax = plt.subplots(figsize=(8, 8))

            # Q-Q plot
            stats.probplot(distances, dist="chi2", sparams=(self.n_components_actual_,), plot=ax)
            ax.set_title('Q-Q Plot: Mahalanobis Distance vs Chi-Square Distribution')
            ax.grid(True, alpha=0.3)

            return fig, ax
        else:
            raise ValueError("Q-Q plot is only available for mahalanobis method")
