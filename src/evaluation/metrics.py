"""
Evaluation Metrics for Anomaly Detection
========================================

This module provides evaluation functions for Report 4:
- Precision, Recall, F1-score, AUC-ROC computation
- ROC curves and confusion matrices visualization
- Comparison tables and heatmaps
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute evaluation metrics for anomaly detection.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0=normal, 1=anomaly)
    y_pred : np.ndarray
        Predicted binary labels
    y_scores : np.ndarray, optional
        Anomaly scores (higher = more anomalous)

    Returns
    -------
    dict
        Dictionary with precision, recall, f1, and optionally auc_roc
    """
    metrics = {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }

    if y_scores is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_scores)
        except ValueError:
            # Only one class present in y_true
            metrics['auc_roc'] = np.nan

    return metrics


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    title: str = "ROC Curve",
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None
) -> plt.Figure:
    """
    Plot ROC curve for a single algorithm.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_scores : np.ndarray
        Anomaly scores
    title : str
        Plot title
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    label : str, optional
        Legend label

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)

    label_text = f"{label} (AUC = {auc:.3f})" if label else f"AUC = {auc:.3f}"
    ax.plot(fpr, tpr, linewidth=2, label=label_text)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    return fig


def plot_roc_curves_comparison(
    results: Dict[str, Dict[str, np.ndarray]],
    title: str = "ROC Curves Comparison",
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot ROC curves for multiple algorithms on same plot.

    Parameters
    ----------
    results : dict
        Dictionary mapping algorithm name to {'y_true': ..., 'y_scores': ...}
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (name, data), color in zip(results.items(), colors):
        y_true = data['y_true']
        y_scores = data['y_scores']

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        try:
            auc = roc_auc_score(y_true, y_scores)
            label = f"{name} (AUC = {auc:.3f})"
        except ValueError:
            label = f"{name} (AUC = N/A)"

        ax.plot(fpr, tpr, linewidth=2, label=label, color=color)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    ax: Optional[plt.Axes] = None,
    labels: List[str] = ['Normal', 'Anomaly']
) -> plt.Figure:
    """
    Plot confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_pred : np.ndarray
        Predicted binary labels
    title : str
        Plot title
    ax : plt.Axes, optional
        Matplotlib axes
    labels : list
        Class labels

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    cm = confusion_matrix(y_true, y_pred)

    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        ax=ax, cbar=True
    )

    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    title: str = "Precision-Recall Curve",
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None
) -> plt.Figure:
    """
    Plot precision-recall curve.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_scores : np.ndarray
        Anomaly scores
    title : str
        Plot title
    ax : plt.Axes, optional
        Matplotlib axes
    label : str, optional
        Legend label

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    ax.plot(recall, precision, linewidth=2, label=label)
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    if label:
        ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    return fig


def create_comparison_table(
    results: Dict[str, Dict[str, Dict[str, float]]]
) -> pd.DataFrame:
    """
    Create comparison table of all algorithms across all datasets.

    Parameters
    ----------
    results : dict
        Nested dict: results[dataset][algorithm] = {'precision': ..., 'recall': ..., ...}

    Returns
    -------
    pd.DataFrame
        Comparison table with multi-index columns
    """
    rows = []

    for dataset, algo_results in results.items():
        for algo, metrics in algo_results.items():
            row = {
                'Dataset': dataset,
                'Algorithm': algo,
                **metrics
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # Pivot for better visualization
    df_pivot = df.pivot(index='Dataset', columns='Algorithm')

    return df, df_pivot


def plot_f1_heatmap(
    results: Dict[str, Dict[str, Dict[str, float]]],
    title: str = "F1-Score Comparison",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot heatmap of F1-scores across datasets and algorithms.

    Parameters
    ----------
    results : dict
        Nested dict: results[dataset][algorithm] = {'f1': ...}
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    # Extract F1 scores into matrix
    datasets = list(results.keys())
    algorithms = list(next(iter(results.values())).keys())

    f1_matrix = np.zeros((len(datasets), len(algorithms)))

    for i, dataset in enumerate(datasets):
        for j, algo in enumerate(algorithms):
            f1_matrix[i, j] = results[dataset][algo].get('f1', 0)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        f1_matrix,
        annot=True,
        fmt='.3f',
        cmap='YlGnBu',
        xticklabels=algorithms,
        yticklabels=datasets,
        ax=ax,
        cbar_kws={'label': 'F1-Score'}
    )

    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('Dataset', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_execution_time_comparison(
    times: Dict[str, Dict[str, float]],
    title: str = "Execution Time Comparison",
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot bar chart comparing execution times.

    Parameters
    ----------
    times : dict
        Nested dict: times[dataset][algorithm] = time_in_seconds
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    datasets = list(times.keys())
    algorithms = list(next(iter(times.values())).keys())

    x = np.arange(len(datasets))
    width = 0.8 / len(algorithms)

    fig, ax = plt.subplots(figsize=figsize)

    for i, algo in enumerate(algorithms):
        values = [times[ds][algo] for ds in datasets]
        offset = (i - len(algorithms) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=algo)

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(
                f'{val:.2f}s',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=8
            )

    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str] = ['Normal', 'Anomaly']
) -> str:
    """
    Print sklearn classification report.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    target_names : list
        Class names

    Returns
    -------
    str
        Classification report string
    """
    return classification_report(y_true, y_pred, target_names=target_names)
