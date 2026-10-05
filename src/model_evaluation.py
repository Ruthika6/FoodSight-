"""
Model Evaluation and Metric Calculation Module.
Computes MAE, RMSE, and R² scores, performs residual analysis,
and programmatically identifies the optimal predictive model based on strict criteria.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class EvaluationMetrics:
    model_name: str
    mae: float
    rmse: float
    r2: float
    predictions: np.ndarray
    residuals: np.ndarray
    is_best: bool = False
    selection_reason: str = ""


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str
) -> EvaluationMetrics:
    """Compute standard regression performance metrics."""
    # Ensure non-negative predictions for food orders
    y_pred_clipped = np.clip(y_pred, 0, None)

    mae = float(mean_absolute_error(y_true, y_pred_clipped))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_clipped)))
    r2 = float(r2_score(y_true, y_pred_clipped))
    residuals = y_true - y_pred_clipped

    return EvaluationMetrics(
        model_name=model_name,
        mae=round(mae, 2),
        rmse=round(rmse, 2),
        r2=round(r2, 4),
        predictions=y_pred_clipped,
        residuals=residuals,
    )


def select_best_model(metrics_dict: Dict[str, EvaluationMetrics]) -> Tuple[str, pd.DataFrame]:
    """
    Select best model using deterministic hierarchical criteria:
    1. Primary: Lowest MAE
    2. Secondary: Lowest RMSE
    3. Tertiary: Highest R²

    Returns: (best_model_name, comparison_summary_dataframe)
    """
    if not metrics_dict:
        raise ValueError("Cannot select best model from empty metrics dictionary.")

    # Sort models according to criteria: lowest MAE, lowest RMSE, highest R2
    sorted_models = sorted(
        metrics_dict.items(),
        key=lambda item: (item[1].mae, item[1].rmse, -item[1].r2)
    )

    best_name, best_metric = sorted_models[0]
    best_metric.is_best = True
    best_metric.selection_reason = (
        f"Selected based on superior primary metric MAE ({best_metric.mae:.2f}) "
        f"and secondary RMSE ({best_metric.rmse:.2f}), R² ({best_metric.r2:.4f})"
    )

    summary_rows = []
    for rank, (name, m) in enumerate(sorted_models, start=1):
        summary_rows.append({
            "Rank": rank,
            "Model": name,
            "MAE": m.mae,
            "RMSE": m.rmse,
            "R² Score": m.r2,
            "Status": "★ Winner (Selected)" if name == best_name else "Candidate",
            "Selection Justification": m.selection_reason if name == best_name else "Sub-optimal error metric"
        })

    df_summary = pd.DataFrame(summary_rows)
    return best_name, df_summary
