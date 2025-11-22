"""
Local Outlier Factor (LOF) Algorithm Implementation

Based on: Breunig, M. M., Kriegel, H.-P., Ng, R. T., & Sander, J. (2000).
LOF: Identifying density-based local outliers.
"""

import numpy as np
from typing import Optional, Union
from numpy.typing import ArrayLike
from scipy.spatial import KDTree


class LOF:
    """
    Local Outlier Factor for anomaly detection.

    LOF identifies outliers by comparing the local density of a point
    with the local densities of its neighbors. Points with substantially
    lower density than their neighbors are considered outliers.

    Parameters
    ----------
    n_neighbors : int, default=20
        Number of neighbors to use for computing local outlier factor.

    metric : str, default='euclidean'
        Distance metric to use. Currently only 'euclidean' is supported.

    use_kdtree : bool, default=True
        Whether to use KD-Tree for k-NN search optimization.
        If True, complexity is O(n log n). If False, uses brute force O(n²).

    Attributes
    ----------
    lof_scores_ : ndarray of shape (n_samples,)
        The Local Outlier Factor for each sample.

    negative_outlier_factor_ : ndarray of shape (n_samples,)
        The opposite of LOF scores (for sklearn compatibility).

    Examples
    --------
    >>> from src.algorithms import LOF
    >>> import numpy as np
    >>> X = np.array([[0, 0], [1, 1], [1, 0], [0, 1], [10, 10]])
    >>> lof = LOF(n_neighbors=2)
    >>> lof.fit(X)
    >>> scores = lof.lof_scores_
    >>> print(scores[-1] > 1.5)  # Last point should be outlier
    True
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        metric: str = 'euclidean',
        use_kdtree: bool = True
    ):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.use_kdtree = use_kdtree

        # Attributes set during fit
        self.X_train_ = None
        self.lof_scores_ = None
        self.negative_outlier_factor_ = None
        self.kdtree_ = None  # KD-Tree for optimized search

    def fit(self, X: ArrayLike) -> 'LOF':
        """
        Fit the LOF model on training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : LOF
            Fitted estimator.
        """
        X = np.asarray(X, dtype=np.float64)

        if X.shape[0] <= self.n_neighbors:
            raise ValueError(
                f"n_neighbors ({self.n_neighbors}) must be less than "
                f"n_samples ({X.shape[0]})"
            )

        self.X_train_ = X

        # Build KD-Tree if optimization is enabled
        if self.use_kdtree:
            self.kdtree_ = KDTree(X)

        # Compute LOF scores for all training samples
        self.lof_scores_ = self._compute_lof_scores(X)
        self.negative_outlier_factor_ = -self.lof_scores_

        return self

    def fit_predict(self, X: ArrayLike) -> np.ndarray:
        """
        Fit the model and return LOF scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        lof_scores : ndarray of shape (n_samples,)
            LOF scores for each sample. Higher values indicate outliers.
        """
        self.fit(X)
        return self.lof_scores_

    def predict(self, X: ArrayLike, threshold: float = 1.5) -> np.ndarray:
        """
        Predict if samples are outliers.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to classify.

        threshold : float, default=1.5
            LOF threshold above which samples are considered outliers.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Binary labels: 1 for outliers, 0 for inliers.
        """
        if self.X_train_ is None:
            raise ValueError("Model must be fitted before calling predict")

        X = np.asarray(X, dtype=np.float64)
        lof_scores = self._compute_lof_scores_for_new_data(X)

        return (lof_scores > threshold).astype(int)

    def _get_neighbors_kdtree(self, X: np.ndarray, tree: KDTree = None) -> tuple:
        """
        Get k-nearest neighbors using KD-Tree (optimized).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Query points.

        tree : KDTree, optional
            KD-Tree to query. If None, uses self.kdtree_.

        Returns
        -------
        distances : ndarray of shape (n_samples, k)
            Distances to k nearest neighbors.

        neighbors : ndarray of shape (n_samples, k)
            Indices of k nearest neighbors.
        """
        if tree is None:
            tree = self.kdtree_

        # Query k+1 neighbors (including self for training data)
        # We'll filter out self-distances later if needed
        k = self.n_neighbors + 1
        distances, neighbors = tree.query(X, k=k)

        # If querying training data on itself, remove self (first column)
        if tree is self.kdtree_ and np.array_equal(X, self.X_train_):
            distances = distances[:, 1:]
            neighbors = neighbors[:, 1:]
        else:
            # For new data, use all k+1 neighbors and take only k
            distances = distances[:, :self.n_neighbors]
            neighbors = neighbors[:, :self.n_neighbors]

        return distances, neighbors

    def _compute_distances(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute pairwise distances between samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            First set of samples.

        Y : ndarray of shape (n_samples_Y, n_features), optional
            Second set of samples. If None, compute distances within X.

        Returns
        -------
        distances : ndarray of shape (n_samples_X, n_samples_Y)
            Pairwise distances.
        """
        if Y is None:
            Y = X

        if self.metric == 'euclidean':
            # Vectorized euclidean distance computation
            # dist(x, y) = sqrt(sum((x - y)^2))
            X_squared = np.sum(X ** 2, axis=1, keepdims=True)
            Y_squared = np.sum(Y ** 2, axis=1, keepdims=True).T
            XY = X @ Y.T

            distances = np.sqrt(
                np.maximum(X_squared - 2 * XY + Y_squared, 0)
            )
            return distances
        else:
            raise ValueError(f"Metric '{self.metric}' is not supported")

    def _k_distance(self, distances: np.ndarray, k: int) -> np.ndarray:
        """
        Compute k-distance for each point.

        The k-distance of point p is the distance to its k-th nearest neighbor.

        Parameters
        ----------
        distances : ndarray of shape (n_samples, n_samples)
            Pairwise distance matrix.

        k : int
            Number of neighbors.

        Returns
        -------
        k_distances : ndarray of shape (n_samples,)
            k-distance for each point.
        """
        # Sort distances and get k-th nearest (skip first which is 0 - self)
        sorted_distances = np.sort(distances, axis=1)
        k_distances = sorted_distances[:, k]  # k+1-th column (0-indexed, skip self)

        return k_distances

    def _get_k_neighbors(self, distances: np.ndarray, k: int) -> np.ndarray:
        """
        Get indices of k nearest neighbors for each point.

        Parameters
        ----------
        distances : ndarray of shape (n_samples, n_samples)
            Pairwise distance matrix.

        k : int
            Number of neighbors.

        Returns
        -------
        neighbors : ndarray of shape (n_samples, k)
            Indices of k nearest neighbors for each point.
        """
        # Get indices of k+1 nearest neighbors (including self)
        neighbors_with_self = np.argsort(distances, axis=1)[:, :k+1]

        # Remove self (first column)
        neighbors = neighbors_with_self[:, 1:k+1]

        return neighbors

    def _reachability_distance(
        self,
        distances: np.ndarray,
        k_distances: np.ndarray,
        neighbors: np.ndarray
    ) -> np.ndarray:
        """
        Compute reachability distance for each point to its neighbors.

        reach-dist(p, o) = max(k-distance(o), d(p, o))

        Parameters
        ----------
        distances : ndarray of shape (n_samples, n_samples)
            Pairwise distance matrix.

        k_distances : ndarray of shape (n_samples,)
            k-distance for each point.

        neighbors : ndarray of shape (n_samples, k)
            Indices of k nearest neighbors.

        Returns
        -------
        reach_distances : ndarray of shape (n_samples, k)
            Reachability distance to each neighbor.
        """
        n_samples = distances.shape[0]
        k = neighbors.shape[1]

        reach_distances = np.zeros((n_samples, k))

        for i in range(n_samples):
            for j, neighbor_idx in enumerate(neighbors[i]):
                # reach-dist(i, neighbor) = max(k-distance(neighbor), d(i, neighbor))
                reach_distances[i, j] = max(
                    k_distances[neighbor_idx],
                    distances[i, neighbor_idx]
                )

        return reach_distances

    def _local_reachability_density(
        self,
        reach_distances: np.ndarray
    ) -> np.ndarray:
        """
        Compute local reachability density for each point.

        LRD(p) = 1 / (average reachability distance of p to its k neighbors)

        Parameters
        ----------
        reach_distances : ndarray of shape (n_samples, k)
            Reachability distances to neighbors.

        Returns
        -------
        lrd : ndarray of shape (n_samples,)
            Local reachability density for each point.
        """
        # Average reachability distance
        avg_reach_dist = np.mean(reach_distances, axis=1)

        # LRD is inverse of average reachability distance
        # Add small epsilon to avoid division by zero
        lrd = 1.0 / (avg_reach_dist + 1e-10)

        return lrd

    def _compute_lof_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Compute LOF scores for samples.

        LOF(p) = (average LRD of neighbors) / LRD(p)

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        lof_scores : ndarray of shape (n_samples,)
            LOF score for each sample.
        """
        if self.use_kdtree and self.kdtree_ is not None:
            # Optimized path using KD-Tree: O(n log n)
            neighbor_dists, neighbors = self._get_neighbors_kdtree(X)

            # k-distance is the distance to the k-th neighbor (last column)
            k_distances = neighbor_dists[:, -1]

            # Create full distance matrix for reachability computation
            # We only need distances between points and their neighbors
            n_samples = X.shape[0]
            reach_distances = np.zeros_like(neighbor_dists)

            for i in range(n_samples):
                for j, neighbor_idx in enumerate(neighbors[i]):
                    # reach-dist(i, neighbor) = max(k-distance(neighbor), d(i, neighbor))
                    reach_distances[i, j] = max(
                        k_distances[neighbor_idx],
                        neighbor_dists[i, j]
                    )
        else:
            # Original brute-force path: O(n²)
            # Step 1: Compute pairwise distances
            distances = self._compute_distances(X)

            # Step 2: Compute k-distance for each point
            k_distances = self._k_distance(distances, self.n_neighbors)

            # Step 3: Get k nearest neighbors
            neighbors = self._get_k_neighbors(distances, self.n_neighbors)

            # Step 4: Compute reachability distances
            reach_distances = self._reachability_distance(distances, k_distances, neighbors)

        # Step 5: Compute local reachability density (LRD)
        lrd = self._local_reachability_density(reach_distances)

        # Step 6: Compute LOF scores
        n_samples = X.shape[0]
        lof_scores = np.zeros(n_samples)

        for i in range(n_samples):
            # Average LRD of neighbors
            neighbor_lrds = lrd[neighbors[i]]
            avg_neighbor_lrd = np.mean(neighbor_lrds)

            # LOF(i) = avg_neighbor_lrd / lrd(i)
            lof_scores[i] = avg_neighbor_lrd / (lrd[i] + 1e-10)

        return lof_scores

    def _compute_lof_scores_for_new_data(self, X: np.ndarray) -> np.ndarray:
        """
        Compute LOF scores for new data points based on fitted model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            New samples.

        Returns
        -------
        lof_scores : ndarray of shape (n_samples,)
            LOF scores for new samples.
        """
        # Get training data statistics (recompute or cache from fit)
        if self.use_kdtree and self.kdtree_ is not None:
            # Use KD-Tree for training data
            train_neighbor_dists, train_neighbors = self._get_neighbors_kdtree(self.X_train_)
            train_k_distances = train_neighbor_dists[:, -1]

            # Compute reachability distances for training data
            n_train = self.X_train_.shape[0]
            train_reach_distances = np.zeros_like(train_neighbor_dists)
            for i in range(n_train):
                for j, neighbor_idx in enumerate(train_neighbors[i]):
                    train_reach_distances[i, j] = max(
                        train_k_distances[neighbor_idx],
                        train_neighbor_dists[i, j]
                    )

            # Query new points against training KD-Tree
            new_neighbor_dists, neighbors = self._get_neighbors_kdtree(X, tree=self.kdtree_)
        else:
            # Brute force approach
            train_distances = self._compute_distances(self.X_train_)
            train_k_distances = self._k_distance(train_distances, self.n_neighbors)
            train_neighbors = self._get_k_neighbors(train_distances, self.n_neighbors)
            train_reach_distances = self._reachability_distance(
                train_distances, train_k_distances, train_neighbors
            )

            # Compute distances from new points to training points
            distances = self._compute_distances(X, self.X_train_)
            # For each new point, find its k nearest neighbors in training set
            neighbors = np.argsort(distances, axis=1)[:, :self.n_neighbors]
            new_neighbor_dists = None  # Will compute in loop

        train_lrd = self._local_reachability_density(train_reach_distances)

        # Compute LOF for new points
        n_samples = X.shape[0]
        lof_scores = np.zeros(n_samples)

        for i in range(n_samples):
            # Compute reachability distances from new point to its neighbors
            reach_dists = []
            for j, neighbor_idx in enumerate(neighbors[i]):
                if new_neighbor_dists is not None:
                    # KD-Tree path
                    dist_to_neighbor = new_neighbor_dists[i, j]
                else:
                    # Brute force path
                    dist_to_neighbor = distances[i, neighbor_idx]

                reach_dist = max(
                    train_k_distances[neighbor_idx],
                    dist_to_neighbor
                )
                reach_dists.append(reach_dist)

            # LRD of new point
            avg_reach_dist = np.mean(reach_dists)
            lrd_new = 1.0 / (avg_reach_dist + 1e-10)

            # Average LRD of neighbors
            neighbor_lrds = train_lrd[neighbors[i]]
            avg_neighbor_lrd = np.mean(neighbor_lrds)

            # LOF score
            lof_scores[i] = avg_neighbor_lrd / (lrd_new + 1e-10)

        return lof_scores

    def score_samples(self, X: ArrayLike) -> np.ndarray:
        """
        Compute anomaly scores for samples (sklearn-compatible).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Anomaly scores (negative outlier factor).
        """
        X = np.asarray(X, dtype=np.float64)
        lof_scores = self._compute_lof_scores_for_new_data(X)
        return -lof_scores
