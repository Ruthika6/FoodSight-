"""
Unit tests for model training, evaluation, comparison, and feature importance.
"""

import numpy as np
import pytest
from src.demo_generator import generate_demo_dataset
from src.feature_engineering import prepare_ml_splits
from src.feature_importance import extract_feature_importance, compute_grouped_importance
from src.model_evaluation import compute_metrics, select_best_model
from src.model_training import train_all_models


@pytest.fixture
def trained_pipeline(tmp_path):
    _, _, _, df_daily = generate_demo_dataset(output_dir=tmp_path, num_weeks=10, seed=42)
    splits = prepare_ml_splits(df_daily, split_ratio=0.80)
    results = train_all_models(splits, models_dir=tmp_path / "models", save_artifacts=True)
    return results, splits


def test_metric_computation():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])
    m = compute_metrics(y_true, y_pred, "TestModel")
    assert m.mae == 10.0
    assert m.rmse == 10.0
    assert m.r2 > 0.95


def test_all_models_train_and_evaluate(trained_pipeline):
    results, splits = trained_pipeline

    assert "Naive Baseline (Last Week Same Day)" in results.models
    assert "Linear Regression (Ridge)" in results.models
    assert "Random Forest Regressor" in results.models
    assert "XGBoost Regressor" in results.models

    # Winner must be dynamically chosen and present in summary table
    assert results.best_model_name in results.models
    assert not results.comparison_table.empty
    assert results.comparison_table.iloc[0]["Rank"] == 1


def test_feature_importance_extraction(trained_pipeline):
    results, splits = trained_pipeline
    best_model = results.best_model
    df_imp = extract_feature_importance(best_model, splits.feature_names, results.best_model_name)

    assert not df_imp.empty
    assert "Feature" in df_imp.columns
    assert "Contribution_Pct" in df_imp.columns
    # Sum of percentages should be approximately 100%
    assert abs(df_imp["Contribution_Pct"].sum() - 100.0) < 1.0

    df_grouped = compute_grouped_importance(df_imp)
    assert not df_grouped.empty
    assert "Factor Category" in df_grouped.columns
    assert "Total Contribution (%)" in df_grouped.columns
