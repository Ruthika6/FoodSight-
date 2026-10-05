"""
Unit tests for feedback loop, database persistence, and retraining drift evaluation.
"""

import pytest
from src.database import get_outcomes_df, get_predictions_df, init_db
from src.feedback_loop import record_feedback
from src.retraining import evaluate_retraining_trigger


@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "test_feedback.db"
    init_db(db_file)
    return db_file


def test_record_feedback_calculations(test_db, monkeypatch):
    from src.config import CONFIG
    monkeypatch.setattr(CONFIG.paths, "database_path", test_db)

    # Prepared 110, Actual demand 100, Predicted 105 -> Actual Waste = 10, Pred Error = +5
    res = record_feedback(
        target_date="2026-10-01",
        center_id=10,
        meal_id=1109,
        predicted_demand=105.0,
        actual_prepared=110,
        actual_demand=100,
        notes="Service test 1"
    )

    assert res.actual_waste == 10
    assert res.shortage == 0
    assert res.prediction_error == 5.0
    assert res.abs_prediction_error == 5.0
    assert res.percentage_error == 5.0

    df_outcomes = get_outcomes_df(test_db)
    assert len(df_outcomes) == 1
    assert df_outcomes.iloc[0]["actual_waste"] == 10


def test_retraining_trigger_drift_detection(test_db, monkeypatch):
    from src.config import CONFIG
    monkeypatch.setattr(CONFIG.paths, "database_path", test_db)

    # Initial state with normal performance
    status_normal = evaluate_retraining_trigger(baseline_benchmark_mae=15.0)
    assert not status_normal.is_retraining_recommended

    # Insert 10 records with high prediction error (e.g. 35 MAE vs baseline 15)
    for i in range(10):
        record_feedback(
            target_date=f"2026-10-{i+1:02d}",
            center_id=10,
            meal_id=1109,
            predicted_demand=140.0,
            actual_prepared=150,
            actual_demand=100,  # Error = 40
        )

    status_drift = evaluate_retraining_trigger(
        baseline_benchmark_mae=15.0,
        threshold_multiplier=1.20,
        consecutive_days_needed=3
    )
    assert status_drift.is_retraining_recommended
    assert "Retraining Recommended" in status_drift.status_label
    assert status_drift.consecutive_breaches >= 3
