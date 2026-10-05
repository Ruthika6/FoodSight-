"""
Closed-Loop Feedback and Live Accuracy Tracking Module.
Records post-service actual food consumption and preparation figures,
calculates actual waste and error metrics, and maintains rolling performance statistics.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import CONFIG
from src.database import get_outcomes_df, get_rolling_mae_series, log_actual_outcome, save_prediction


@dataclass
class FeedbackEntryResult:
    outcome_id: int
    target_date: str
    center_id: int
    meal_id: int
    predicted_demand: float
    actual_prepared: int
    actual_demand: int
    actual_waste: int
    shortage: int
    prediction_error: float
    abs_prediction_error: float
    percentage_error: float
    rolling_mae_7d: float
    summary_message: str


def record_feedback(
    target_date: str,
    center_id: int,
    meal_id: int,
    predicted_demand: float,
    actual_prepared: int,
    actual_demand: int,
    prediction_id: Optional[int] = None,
    notes: str = ""
) -> FeedbackEntryResult:
    """
    Log actual daily outcome, persist to SQLite, and re-compute rolling MAE.
    """
    outcome_id = log_actual_outcome(
        target_date=target_date,
        center_id=center_id,
        meal_id=meal_id,
        predicted_demand=predicted_demand,
        actual_prepared=actual_prepared,
        actual_demand=actual_demand,
        prediction_id=prediction_id,
        notes=notes
    )

    actual_waste = max(0, actual_prepared - actual_demand)
    shortage = max(0, actual_demand - actual_prepared)
    pred_error = round(float(predicted_demand - actual_demand), 2)
    abs_error = round(abs(float(predicted_demand - actual_demand)), 2)
    pct_error = round((abs_error / actual_demand * 100.0) if actual_demand > 0 else 0.0, 2)

    # Compute current rolling MAE
    rolling_df = get_rolling_mae_series(window_days=CONFIG.retraining.rolling_window_days)
    curr_rolling_mae = float(rolling_df["rolling_mae"].iloc[-1]) if not rolling_df.empty else abs_error

    msg = (
        f"Recorded outcome for Center {center_id}, Meal {meal_id} on {target_date}. "
        f"Actual Waste: {actual_waste} meals | Pred Error: {pred_error:+.1f} meals | 7D Rolling MAE: {curr_rolling_mae:.2f}"
    )

    return FeedbackEntryResult(
        outcome_id=outcome_id,
        target_date=target_date,
        center_id=center_id,
        meal_id=meal_id,
        predicted_demand=predicted_demand,
        actual_prepared=actual_prepared,
        actual_demand=actual_demand,
        actual_waste=actual_waste,
        shortage=shortage,
        prediction_error=pred_error,
        abs_prediction_error=abs_error,
        percentage_error=pct_error,
        rolling_mae_7d=curr_rolling_mae,
        summary_message=msg
    )


def seed_initial_feedback_history(
    num_days: int = 21,
    force_reseed: bool = False
) -> int:
    """
    Seed realistic initial historical feedback entries if SQLite database is empty.
    Provides immediate historical rolling MAE and waste trend visualizations.
    """
    existing = get_outcomes_df(limit=5)
    if not existing.empty and not force_reseed:
        return len(existing)

    rng = np.random.RandomState(42)
    start_date = datetime.now().date() - timedelta(days=num_days)
    records_count = 0

    # Common sample centers and meals
    sample_pairs = [(10, 1109), (10, 1062), (13, 1109), (24, 1993), (55, 1207)]

    for day_offset in range(num_days):
        cur_date = (start_date + timedelta(days=day_offset)).isoformat()
        for center_id, meal_id in sample_pairs:
            base_dem = 320 if meal_id == 1109 else (250 if meal_id == 1993 else 180)
            day_noise = rng.normal(0, 15)
            actual_dem = int(max(50, round(base_dem + day_noise)))

            # Forecast has realistic small error
            model_noise = rng.normal(0, 12)
            predicted_dem = float(max(40, round(actual_dem + model_noise)))

            # Kitchen prepares predicted + 5% buffer + human variation
            planned_buffer = int(np.ceil(predicted_dem * 0.05))
            actual_prep = int(predicted_dem + planned_buffer + rng.randint(-5, 10))

            pred_id = save_prediction(
                prediction_date=(datetime.strptime(cur_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d"),
                target_date=cur_date,
                center_id=center_id,
                meal_id=meal_id,
                predicted_demand=predicted_dem,
                recommended_prep=int(predicted_dem + planned_buffer),
                buffer_pct=5.0,
                model_used="XGBoost Regressor",
                weather_condition="Clear" if rng.uniform() > 0.3 else "Rainy",
                temperature=26.5 + rng.uniform(-4, 4),
                rainfall=0.0 if rng.uniform() > 0.3 else rng.uniform(2, 15),
                planned_price=280.0
            )

            log_actual_outcome(
                target_date=cur_date,
                center_id=center_id,
                meal_id=meal_id,
                predicted_demand=predicted_dem,
                actual_prepared=actual_prep,
                actual_demand=actual_dem,
                prediction_id=pred_id,
                notes="Standard service log"
            )
            records_count += 1

    return records_count
