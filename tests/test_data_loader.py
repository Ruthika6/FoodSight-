"""
Unit tests for data loading, schema validation, and disaggregation pipeline.
"""

import pandas as pd
import pytest
from src.data_loader import load_raw_or_demo_data, validate_schema
from src.demo_generator import generate_demo_dataset
from src.disaggregation import disaggregate_weekly_dataframe


def test_demo_dataset_generation(tmp_path):
    df_centers, df_meals, df_weekly, df_daily = generate_demo_dataset(output_dir=tmp_path, num_weeks=4)
    assert not df_centers.empty
    assert not df_meals.empty
    assert not df_weekly.empty
    assert not df_daily.empty
    assert "center_id" in df_daily.columns
    assert "meal_id" in df_daily.columns
    assert "num_orders" in df_daily.columns
    assert "temperature_c" in df_daily.columns


def test_schema_validation():
    df = pd.DataFrame({
        "week": [1, 2],
        "center_id": [10, 10],
        "meal_id": [1109, 1109],
        "num_orders": [200, 250]
    })
    is_valid, notes = validate_schema(df, {"week", "center_id", "meal_id", "num_orders"}, "test_df")
    assert is_valid
    assert len(notes) == 0

    is_invalid, missing_notes = validate_schema(df, {"week", "center_id", "missing_col"}, "test_df")
    assert not is_invalid
    assert any("missing required columns" in note for note in missing_notes)


def test_disaggregation_preserves_order_sums():
    df_weekly = pd.DataFrame([{
        "week": 1,
        "center_id": 10,
        "meal_id": 1109,
        "num_orders": 700,
        "checkout_price": 250.0,
        "base_price": 280.0,
        "emailer_for_promotion": 0,
        "homepage_featured": 0,
    }])
    df_daily = disaggregate_weekly_dataframe(df_weekly, start_date="2024-01-01", seed=42)
    assert len(df_daily) == 7
    # Sum of daily orders must match the weekly total
    assert df_daily["num_orders"].sum() == 700
    assert (df_daily["num_orders"] >= 0).all()


def test_load_raw_or_demo_data_fallback():
    df_daily, metadata = load_raw_or_demo_data(force_demo=True)
    assert not df_daily.empty
    assert metadata.source_mode == "DEMO DATA"
    assert metadata.num_rows > 0
    assert metadata.num_centers > 0
    assert metadata.num_meals > 0
