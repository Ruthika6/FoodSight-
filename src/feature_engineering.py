"""
Feature Engineering Module.
Creates temporal lag features, rolling statistics, calendar/holiday indicators,
weather condition encoding, and pricing interaction terms without data leakage.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# Major fixed national and cultural holidays (Month-Day format)
FIXED_HOLIDAYS = {
    (1, 1): "New Year's Day",
    (1, 26): "Republic Day",
    (5, 1): "May Day",
    (8, 15): "Independence Day",
    (10, 2): "Gandhi Jayanti",
    (11, 1): "Karnataka Rajyotsava",
    (12, 25): "Christmas",
}


def assign_season(month: int) -> str:
    """Assign Indian meteorological season based on month."""
    if month in (12, 1, 2):
        return "Winter"
    elif month in (3, 4, 5):
        return "Summer"
    elif month in (6, 7, 8, 9):
        return "Monsoon"
    else:
        return "Autumn"


def build_feature_pipeline(
    df: pd.DataFrame,
    is_training: bool = True,
    fill_lag_na: bool = True
) -> pd.DataFrame:
    """
    Construct all engineered features for day-level food demand forecasting.
    Maintains strict time causality per (center_id, meal_id) series.
    """
    df_feat = df.copy()
    df_feat["date"] = pd.to_datetime(df_feat["date"])
    df_feat = df_feat.sort_values(by=["center_id", "meal_id", "date"]).reset_index(drop=True)

    # 1. Calendar Features
    df_feat["day_of_week"] = df_feat["date"].dt.weekday
    df_feat["is_weekend"] = df_feat["day_of_week"].isin([5, 6]).astype(int)
    df_feat["month"] = df_feat["date"].dt.month
    df_feat["day_of_month"] = df_feat["date"].dt.day
    df_feat["season"] = df_feat["month"].apply(assign_season)

    # Holiday Flag
    df_feat["is_holiday"] = df_feat["date"].apply(
        lambda d: 1 if (d.month, d.day) in FIXED_HOLIDAYS else 0
    )

    # 2. Pricing & Promotion Features
    if "base_price" in df_feat.columns and "checkout_price" in df_feat.columns:
        df_feat["discount_amount"] = (df_feat["base_price"] - df_feat["checkout_price"]).clip(lower=0.0)
        df_feat["discount_ratio"] = (df_feat["discount_amount"] / df_feat["base_price"].replace(0, np.nan)).fillna(0.0)
    else:
        df_feat["discount_amount"] = 0.0
        df_feat["discount_ratio"] = 0.0

    df_feat["emailer_for_promotion"] = df_feat.get("emailer_for_promotion", 0).astype(int)
    df_feat["homepage_featured"] = df_feat.get("homepage_featured", 0).astype(int)
    df_feat["promo_interaction"] = df_feat["emailer_for_promotion"] * df_feat["homepage_featured"]

    # 3. Weather Condition Features
    df_feat["temperature_c"] = df_feat.get("temperature_c", 25.0).astype(float)
    df_feat["rainfall_mm"] = df_feat.get("rainfall_mm", 0.0).astype(float)
    df_feat["rain_probability_pct"] = df_feat.get("rain_probability_pct", 10.0).astype(float)
    df_feat["is_heavy_rain"] = (df_feat["rainfall_mm"] > 10.0).astype(int)
    df_feat["is_extreme_weather"] = (
        (df_feat["rainfall_mm"] > 30.0) |
        (df_feat["temperature_c"] > 38.0) |
        (df_feat["temperature_c"] < 10.0)
    ).astype(int)

    # 4. Demand History Features (Lag & Rolling per series)
    grouped = df_feat.groupby(["center_id", "meal_id"])

    # Lag 1 (yesterday) and Lag 7 (same day last week)
    df_feat["demand_lag_1"] = grouped["num_orders"].shift(1)
    df_feat["demand_lag_7"] = grouped["num_orders"].shift(7)

    # Rolling 7-day mean and std (computed from shifted series to avoid leakage of today's target)
    shifted_orders = grouped["num_orders"].shift(1)
    df_feat["demand_rolling_7_mean"] = (
        shifted_orders.groupby([df_feat["center_id"], df_feat["meal_id"]])
        .transform(lambda s: s.rolling(window=7, min_periods=1).mean())
    )
    df_feat["demand_rolling_7_std"] = (
        shifted_orders.groupby([df_feat["center_id"], df_feat["meal_id"]])
        .transform(lambda s: s.rolling(window=7, min_periods=1).std())
    ).fillna(0.0)

    # 7-day Trend: rate of change from t-7 to t-1
    df_feat["demand_trend_7"] = ((df_feat["demand_lag_1"] - df_feat["demand_lag_7"]) / 7.0).fillna(0.0)

    # Fill earliest missing lags with series median if needed
    if fill_lag_na:
        series_medians = grouped["num_orders"].transform("median")
        df_feat["demand_lag_1"] = df_feat["demand_lag_1"].fillna(series_medians)
        df_feat["demand_lag_7"] = df_feat["demand_lag_7"].fillna(series_medians)
        df_feat["demand_rolling_7_mean"] = df_feat["demand_rolling_7_mean"].fillna(series_medians)

    # Sort back chronologically
    df_feat = df_feat.sort_values(by=["date", "center_id", "meal_id"]).reset_index(drop=True)
    return df_feat


# Categorical feature specifications
CATEGORICAL_FEATURES = ["category", "cuisine", "center_type", "weather_condition", "season"]
NUMERICAL_FEATURES = [
    "checkout_price", "base_price", "discount_amount", "discount_ratio",
    "emailer_for_promotion", "homepage_featured", "promo_interaction",
    "op_area", "temperature_c", "rainfall_mm", "rain_probability_pct",
    "is_heavy_rain", "is_extreme_weather", "day_of_week", "is_weekend",
    "is_holiday", "month", "day_of_month",
    "demand_lag_1", "demand_lag_7", "demand_rolling_7_mean",
    "demand_rolling_7_std", "demand_trend_7"
]


@dataclass
class DatasetSplits:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train_dates: pd.Series
    test_dates: pd.Series
    feature_names: List[str]
    df_test_raw: pd.DataFrame
    split_date_threshold: str
    categorical_encoders: Dict[str, Dict[str, int]]


def prepare_ml_splits(
    df: pd.DataFrame,
    split_ratio: float = 0.80
) -> DatasetSplits:
    """
    Perform chronological train/test split to prevent temporal data leakage.
    Encodes categoricals deterministically.
    """
    df_engineered = build_feature_pipeline(df, is_training=True, fill_lag_na=False)

    # Drop the first 7 warmup days per series where lag_7 is NaN
    df_clean = df_engineered.dropna(subset=["demand_lag_7", "demand_rolling_7_mean", "num_orders"]).copy()
    df_clean = df_clean.sort_values(by="date").reset_index(drop=True)

    # Determine chronological cutoff date
    unique_dates = df_clean["date"].sort_values().unique()
    split_idx = int(len(unique_dates) * split_ratio)
    split_date = unique_dates[split_idx]

    train_mask = df_clean["date"] < split_date
    test_mask = df_clean["date"] >= split_date

    df_train = df_clean[train_mask].copy()
    df_test = df_clean[test_mask].copy()

    # Build categorical label encoders from training data
    cat_encoders: Dict[str, Dict[str, int]] = {}
    for cat_col in CATEGORICAL_FEATURES:
        if cat_col in df_clean.columns:
            unique_vals = sorted(df_clean[cat_col].astype(str).unique())
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            cat_encoders[cat_col] = mapping

    def encode_df(data: pd.DataFrame) -> pd.DataFrame:
        data_encoded = data.copy()
        for cat_col, mapping in cat_encoders.items():
            if cat_col in data_encoded.columns:
                data_encoded[cat_col + "_encoded"] = data_encoded[cat_col].astype(str).map(mapping).fillna(0).astype(int)
        return data_encoded

    df_train_enc = encode_df(df_train)
    df_test_enc = encode_df(df_test)

    encoded_cat_cols = [c + "_encoded" for c in CATEGORICAL_FEATURES if c in df_clean.columns]
    feature_cols = [c for c in NUMERICAL_FEATURES if c in df_clean.columns] + encoded_cat_cols

    X_train = df_train_enc[feature_cols]
    y_train = df_train_enc["num_orders"]
    X_test = df_test_enc[feature_cols]
    y_test = df_test_enc["num_orders"]

    return DatasetSplits(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        train_dates=df_train["date"],
        test_dates=df_test["date"],
        feature_names=feature_cols,
        df_test_raw=df_test,
        split_date_threshold=pd.to_datetime(split_date).strftime("%Y-%m-%d"),
        categorical_encoders=cat_encoders
    )
