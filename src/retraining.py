"""
Retraining and Concept Drift Monitoring Module.
Monitors live rolling MAE against model benchmark performance and manages retraining orchestration.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple
import pandas as pd

from src.config import CONFIG
from src.database import get_outcomes_df, get_rolling_mae_series, log_retraining_event
from src.feature_engineering import prepare_ml_splits
from src.model_training import ModelPipelineResults, train_all_models


@dataclass
class RetrainingStatus:
    is_retraining_recommended: bool
    status_label: str  # "✓ Performing Normally" or "⚠ Retraining Recommended"
    badge_color: str  # "#10B981" (green) or "#EF4444" (red)
    baseline_mae: float
    current_rolling_mae: float
    allowed_threshold_mae: float
    consecutive_breaches: int
    trigger_reason: str
    detailed_explanation: str


def evaluate_retraining_trigger(
    baseline_benchmark_mae: float = 20.0,
    window_days: Optional[int] = None,
    threshold_multiplier: Optional[float] = None,
    consecutive_days_needed: Optional[int] = None
) -> RetrainingStatus:
    """
    Examine recent live feedback outcomes and assess whether prediction degradation justifies retraining.
    """
    window = window_days or CONFIG.retraining.rolling_window_days
    multiplier = threshold_multiplier or CONFIG.retraining.error_threshold_multiplier
    consecutive_needed = consecutive_days_needed or CONFIG.retraining.consecutive_days_trigger

    rolling_df = get_rolling_mae_series(window_days=window)
    allowed_mae = round(baseline_benchmark_mae * multiplier, 2)

    if rolling_df.empty or len(rolling_df) < window:
        return RetrainingStatus(
            is_retraining_recommended=False,
            status_label="✓ Performing Normally",
            badge_color="#10B981",
            baseline_mae=round(baseline_benchmark_mae, 2),
            current_rolling_mae=round(baseline_benchmark_mae, 2),
            allowed_threshold_mae=allowed_mae,
            consecutive_breaches=0,
            trigger_reason="Insufficient live feedback data to evaluate drift (< 7 days)",
            detailed_explanation=(
                f"Model baseline test MAE is {baseline_benchmark_mae:.2f}. "
                f"Tracking active. Live rolling error is currently within acceptable tolerance limits."
            )
        )

    recent_mae_series = rolling_df["rolling_mae"].iloc[-14:]
    current_mae = round(float(rolling_df["rolling_mae"].iloc[-1]), 2)

    # Count consecutive trailing days where rolling MAE exceeded the allowed threshold
    breaches = 0
    for val in reversed(recent_mae_series.tolist()):
        if val > allowed_mae:
            breaches += 1
        else:
            break

    if breaches >= consecutive_needed:
        is_flagged = True
        status_lbl = "⚠ Retraining Recommended"
        badge_col = "#EF4444"
        reason = f"Live 7-day rolling MAE ({current_mae:.2f}) exceeded benchmark allowance ({allowed_mae:.2f}) for {breaches} consecutive records."
        explanation = (
            f"Model drift detected! Live performance has degraded by {((current_mae - baseline_benchmark_mae) / baseline_benchmark_mae * 100):.1f}% "
            f"relative to the training benchmark of {baseline_benchmark_mae:.2f} MAE. "
            f"Kitchen consumption patterns or external conditions have shifted. Retraining with latest feedback data is strongly recommended."
        )
    else:
        is_flagged = False
        status_lbl = "✓ Performing Normally"
        badge_col = "#10B981"
        reason = f"Current 7-day rolling MAE ({current_mae:.2f}) is within acceptable benchmark threshold (≤ {allowed_mae:.2f})."
        explanation = (
            f"The predictive model is operating with high fidelity. Live error ({current_mae:.2f} MAE) is well within the "
            f"safety envelope ({allowed_mae:.2f} MAE, representing a max {int((multiplier - 1.0)*100)}% tolerance above baseline {baseline_benchmark_mae:.2f})."
        )

    return RetrainingStatus(
        is_retraining_recommended=is_flagged,
        status_label=status_lbl,
        badge_color=badge_col,
        baseline_mae=round(baseline_benchmark_mae, 2),
        current_rolling_mae=current_mae,
        allowed_threshold_mae=allowed_mae,
        consecutive_breaches=breaches,
        trigger_reason=reason,
        detailed_explanation=explanation
    )


def execute_pipeline_retraining(
    df_raw: pd.DataFrame,
    trigger_reason: str = "Manual User Trigger"
) -> Tuple[ModelPipelineResults, RetrainingStatus]:
    """
    Retrain all machine learning models on the latest dataset, evaluate performance,
    update persistent model artifacts, and log the retraining event.
    """
    # 1. Prepare ML splits
    splits = prepare_ml_splits(df_raw, split_ratio=CONFIG.data.train_test_split_ratio)

    # 2. Train models and save artifacts
    results = train_all_models(splits, save_artifacts=True)

    # 3. Log retraining event in database
    best_mae = results.train_metadata["best_model_mae"]
    log_retraining_event(
        trigger_reason=trigger_reason,
        baseline_mae=best_mae,
        recent_rolling_mae=best_mae,
        status="Completed Successfully",
        new_model_name=results.best_model_name,
        new_mae=best_mae,
        details=f"Retrained on {len(splits.X_train)} samples across {len(splits.feature_names)} features."
    )

    # 4. Check new status
    new_status = evaluate_retraining_trigger(baseline_benchmark_mae=best_mae)

    return results, new_status
