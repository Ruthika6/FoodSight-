"""
Unit tests for feature engineering and chronological train/test splits.
"""

import pandas as pd
import pytest
from src.demo_generator import generate_demo_dataset
from src.feature_engineering import build_feature_pipeline, prepare_ml_splits


@pytest.fixture
def sample_daily_data(tmp_path):
    _, _, _, df_daily = generate_demo_dataset(output_dir=tmp_path, num_weeks=8, seed=42)
    return df_daily


def test_feature_pipeline_columns(sample_daily_data):
    df_feat = build_feature_pipeline(sample_daily_data, is_training=True, fill_lag_na=True)

    expected_features = [
        "day_of_week", "is_weekend", "is_holiday", "month", "season",
        "discount_amount", "discount_ratio", "promo_interaction",
        "demand_lag_1", "demand_lag_7", "demand_rolling_7_mean",
        "demand_rolling_7_std", "demand_trend_7", "is_heavy_rain", "is_extreme_weather"
    ]
    for feat in expected_features:
        assert feat in df_feat.columns, f"Missing engineered feature: {feat}"

    assert (df_feat["demand_rolling_7_mean"] >= 0).all()


def test_chronological_train_test_split_no_leakage(sample_daily_data):
    splits = prepare_ml_splits(sample_daily_data, split_ratio=0.75)

    assert len(splits.X_train) > 0
    assert len(splits.X_test) > 0
    assert len(splits.y_train) == len(splits.X_train)
    assert len(splits.y_test) == len(splits.X_test)

    # Verify chronological purity: max train date must be strictly earlier than min test date
    max_train_date = splits.train_dates.max()
    min_test_date = splits.test_dates.min()
    assert max_train_date < min_test_date, f"Temporal leakage detected: train max {max_train_date} >= test min {min_test_date}"
