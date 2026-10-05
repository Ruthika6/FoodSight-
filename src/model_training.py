"""
Model Training and Experimentation Pipeline Module.
Implements Naive Baseline, Linear Regression, Random Forest, and XGBoost Regressor.
Performs training, evaluation, programmatic selection, and artifact serialization.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.config import CONFIG
from src.feature_engineering import DatasetSplits
from src.model_evaluation import EvaluationMetrics, compute_metrics, select_best_model


class NaiveBaselineRegressor:
    """
    Naive baseline predicting same-day-last-week demand (demand_lag_7).
    Falls back to demand_rolling_7_mean or series mean if lag_7 is missing.
    """
    def __init__(self):
        self.feature_name = "demand_lag_7"
        self.fallback_feature = "demand_rolling_7_mean"

    def fit(self, X: pd.DataFrame, y: pd.Series):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.feature_name in X.columns:
            preds = X[self.feature_name].copy()
            if self.fallback_feature in X.columns:
                preds = preds.fillna(X[self.fallback_feature])
            preds = preds.fillna(preds.median() if not preds.dropna().empty else 100.0)
            return preds.to_numpy()
        return np.full(len(X), 100.0)


@dataclass
class ModelPipelineResults:
    models: Dict[str, Any]
    metrics: Dict[str, EvaluationMetrics]
    comparison_table: pd.DataFrame
    best_model_name: str
    best_model: Any
    feature_names: list
    categorical_encoders: Dict[str, Dict[str, int]]
    train_metadata: Dict[str, Any]


def train_all_models(
    splits: DatasetSplits,
    models_dir: Optional[Path] = None,
    save_artifacts: bool = True
) -> ModelPipelineResults:
    """
    Train all 4 candidate models, evaluate strictly on chronological test split,
    determine winning model, and optionally serialize artifacts.
    """
    target_models_dir = models_dir or CONFIG.paths.models_dir
    target_models_dir.mkdir(parents=True, exist_ok=True)

    X_train = splits.X_train
    y_train = splits.y_train
    X_test = splits.X_test
    y_test = splits.y_test

    models: Dict[str, Any] = {}
    metrics: Dict[str, EvaluationMetrics] = {}

    # 1. Naive Baseline
    naive = NaiveBaselineRegressor()
    naive.fit(X_train, y_train)
    y_pred_naive = naive.predict(X_test)
    metrics["Naive Baseline (Last Week Same Day)"] = compute_metrics(
        y_test.to_numpy(), y_pred_naive, "Naive Baseline (Last Week Same Day)"
    )
    models["Naive Baseline (Last Week Same Day)"] = naive

    # 2. Linear Regression (Ridge with StandardScaler)
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=1.0, random_state=42))
    ])
    lr_pipe.fit(X_train, y_train)
    y_pred_lr = lr_pipe.predict(X_test)
    metrics["Linear Regression (Ridge)"] = compute_metrics(
        y_test.to_numpy(), y_pred_lr, "Linear Regression (Ridge)"
    )
    models["Linear Regression (Ridge)"] = lr_pipe

    # 3. Random Forest Regressor
    rf_reg = RandomForestRegressor(
        n_estimators=100,
        max_depth=14,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    rf_reg.fit(X_train, y_train)
    y_pred_rf = rf_reg.predict(X_test)
    metrics["Random Forest Regressor"] = compute_metrics(
        y_test.to_numpy(), y_pred_rf, "Random Forest Regressor"
    )
    models["Random Forest Regressor"] = rf_reg

    # 4. XGBoost Regressor
    xgb_reg = XGBRegressor(
        n_estimators=120,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1
    )
    xgb_reg.fit(X_train, y_train)
    y_pred_xgb = xgb_reg.predict(X_test)
    metrics["XGBoost Regressor"] = compute_metrics(
        y_test.to_numpy(), y_pred_xgb, "XGBoost Regressor"
    )
    models["XGBoost Regressor"] = xgb_reg

    # Programmatic Winner Selection
    best_name, comparison_table = select_best_model(metrics)
    best_model_obj = models[best_name]

    train_metadata = {
        "training_timestamp": datetime.now().isoformat(),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "split_cutoff_date": splits.split_date_threshold,
        "feature_count": len(splits.feature_names),
        "features": splits.feature_names,
        "best_model_name": best_name,
        "best_model_mae": metrics[best_name].mae,
        "best_model_rmse": metrics[best_name].rmse,
        "best_model_r2": metrics[best_name].r2,
        "all_metrics": {
            name: {"mae": m.mae, "rmse": m.rmse, "r2": m.r2}
            for name, m in metrics.items()
        }
    }

    if save_artifacts:
        # Save best model and metadata
        joblib.dump(best_model_obj, target_models_dir / "best_model.joblib")
        joblib.dump(models, target_models_dir / "all_models.joblib")

        with open(target_models_dir / "model_metadata.json", "w", encoding="utf-8") as f:
            json.dump(train_metadata, f, indent=2)

        with open(target_models_dir / "categorical_encoders.json", "w", encoding="utf-8") as f:
            json.dump(splits.categorical_encoders, f, indent=2)

    return ModelPipelineResults(
        models=models,
        metrics=metrics,
        comparison_table=comparison_table,
        best_model_name=best_name,
        best_model=best_model_obj,
        feature_names=splits.feature_names,
        categorical_encoders=splits.categorical_encoders,
        train_metadata=train_metadata
    )


def load_trained_artifacts(
    models_dir: Optional[Path] = None
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
    """
    Attempt to load serialized model bundle and metadata from disk.
    Returns (models_dict, metadata_dict, encoders_dict) or None if absent.
    """
    target_models_dir = models_dir or CONFIG.paths.models_dir
    meta_file = target_models_dir / "model_metadata.json"
    models_file = target_models_dir / "all_models.joblib"
    encoders_file = target_models_dir / "categorical_encoders.json"

    if not (meta_file.exists() and models_file.exists() and encoders_file.exists()):
        return None

    try:
        models_dict = joblib.load(models_file)
        with open(meta_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        with open(encoders_file, "r", encoding="utf-8") as f:
            encoders = json.load(f)
        return models_dict, metadata, encoders
    except Exception:
        return None
