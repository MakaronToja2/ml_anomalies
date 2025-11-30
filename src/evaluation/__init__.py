# Evaluation module
from .metrics import (
    compute_metrics,
    plot_roc_curve,
    plot_roc_curves_comparison,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    create_comparison_table,
    plot_f1_heatmap,
    plot_execution_time_comparison
)

__all__ = [
    'compute_metrics',
    'plot_roc_curve',
    'plot_roc_curves_comparison',
    'plot_confusion_matrix',
    'plot_precision_recall_curve',
    'create_comparison_table',
    'plot_f1_heatmap',
    'plot_execution_time_comparison'
]
