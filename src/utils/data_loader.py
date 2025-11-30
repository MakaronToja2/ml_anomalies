"""
Data Loading Utilities for Anomaly Detection Project
=====================================================

This module provides functions to load and preprocess datasets:
- KDD Cup 99 (intrusion detection)
- Credit Card Fraud Detection
- Breast Cancer Wisconsin
- Synthetic data generation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Base path for data directory
DATA_DIR = Path(__file__).parent.parent.parent / 'data'


def load_kdd_cup_99(
    path: Optional[str] = None,
    sample_size: Optional[int] = None,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load KDD Cup 99 dataset for intrusion detection.

    Parameters
    ----------
    path : str, optional
        Path to kddcup.data.gz file. If None, uses default location.
    sample_size : int, optional
        Number of samples to load. If None, loads all data.
    random_state : int
        Random seed for sampling.

    Returns
    -------
    X : np.ndarray
        Feature matrix (standardized)
    y : np.ndarray
        Binary labels (0=normal, 1=anomaly/attack)
    """
    if path is None:
        path = DATA_DIR / 'kdd_cup_99' / 'kddcup.data.gz'

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"KDD Cup 99 dataset not found at {path}. "
            "Please download from https://www.kaggle.com/datasets/galaxyh/kdd-cup-1999-data "
            "and place kddcup.data.gz in data/kdd_cup_99/"
        )

    # Column names for KDD Cup 99 dataset
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
    ]

    # Read gzipped CSV
    df = pd.read_csv(path, names=columns, compression='gzip')

    # Sample if requested
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state)

    # Convert labels: normal=0, any attack=1
    y = (df['label'] != 'normal.').astype(int).values

    # Remove label column
    df = df.drop('label', axis=1)

    # Encode categorical features
    categorical_cols = ['protocol_type', 'service', 'flag']
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # Convert to numpy and standardize
    X = df.values.astype(np.float64)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


def load_credit_card_fraud(
    path: Optional[str] = None,
    sample_size: Optional[int] = None,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Credit Card Fraud Detection dataset.

    Parameters
    ----------
    path : str, optional
        Path to creditcard.csv file. If None, uses default location.
    sample_size : int, optional
        Number of samples to load. If None, loads all data.
    random_state : int
        Random seed for sampling.

    Returns
    -------
    X : np.ndarray
        Feature matrix (standardized)
    y : np.ndarray
        Binary labels (0=normal, 1=fraud)
    """
    if path is None:
        path = DATA_DIR / 'credit_card' / 'creditcard.csv'

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Credit Card Fraud dataset not found at {path}. "
            "Please download from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
            "and place creditcard.csv in data/credit_card/"
        )

    # Read CSV
    df = pd.read_csv(path)

    # Sample if requested
    if sample_size is not None and sample_size < len(df):
        # Stratified sampling to preserve class distribution
        df_normal = df[df['Class'] == 0]
        df_fraud = df[df['Class'] == 1]

        fraud_ratio = len(df_fraud) / len(df)
        n_fraud = max(1, int(sample_size * fraud_ratio))
        n_normal = sample_size - n_fraud

        df_normal_sampled = df_normal.sample(n=min(n_normal, len(df_normal)), random_state=random_state)
        df_fraud_sampled = df_fraud.sample(n=min(n_fraud, len(df_fraud)), random_state=random_state)

        df = pd.concat([df_normal_sampled, df_fraud_sampled])

    # Extract labels
    y = df['Class'].values

    # Features (V1-V28 are already PCA-transformed, also include Time and Amount)
    X = df.drop('Class', axis=1).values.astype(np.float64)

    # Standardize (especially for Time and Amount)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


def load_breast_cancer_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Breast Cancer Wisconsin dataset from sklearn.

    Returns
    -------
    X : np.ndarray
        Feature matrix (standardized)
    y : np.ndarray
        Binary labels (0=benign/normal, 1=malignant/anomaly)

    Notes
    -----
    Original sklearn labels: 0=malignant, 1=benign
    We flip them for anomaly detection convention: malignant=anomaly
    """
    data = load_breast_cancer()
    X = data.data

    # Flip labels: malignant (0) becomes anomaly (1), benign (1) becomes normal (0)
    y = 1 - data.target

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


def generate_synthetic(
    n_samples: int = 10000,
    n_features: int = 10,
    contamination: float = 0.05,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic dataset with controlled anomalies.

    Parameters
    ----------
    n_samples : int
        Total number of samples
    n_features : int
        Number of features
    contamination : float
        Fraction of anomalies (0 to 1)
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    X : np.ndarray
        Feature matrix (standardized)
    y : np.ndarray
        Binary labels (0=normal, 1=anomaly)
    """
    np.random.seed(random_state)

    n_outliers = int(n_samples * contamination)
    n_inliers = n_samples - n_outliers

    # Generate normal data (Gaussian, centered at origin)
    X_inliers = np.random.randn(n_inliers, n_features)

    # Generate anomalies (shifted Gaussian)
    X_outliers = np.random.randn(n_outliers, n_features) + 3

    # Combine
    X = np.vstack([X_inliers, X_outliers])
    y = np.hstack([np.zeros(n_inliers), np.ones(n_outliers)])

    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices].astype(int)

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


def preprocess_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train and test sets.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels
    test_size : float
        Fraction of data for testing
    random_state : int
        Random seed

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
        Split datasets
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def get_dataset_info(X: np.ndarray, y: np.ndarray, name: str) -> dict:
    """
    Get summary information about a dataset.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels
    name : str
        Dataset name

    Returns
    -------
    dict
        Dataset statistics
    """
    n_samples, n_features = X.shape
    n_anomalies = np.sum(y == 1)
    n_normal = np.sum(y == 0)
    contamination = n_anomalies / n_samples

    return {
        'name': name,
        'n_samples': n_samples,
        'n_features': n_features,
        'n_normal': n_normal,
        'n_anomalies': n_anomalies,
        'contamination': contamination,
        'contamination_pct': f"{contamination * 100:.2f}%"
    }
