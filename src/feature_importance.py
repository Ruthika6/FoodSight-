"""
Feature Importance and Model Interpretability Module.
Computes model-based feature importances and rigorously aggregated domain group contributions.
Transparently calculates percentage contributions for Demand History, Weather, Pricing, Calendar, and Meal/Location.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd


FEATURE_GROUPS: Dict[str, List[str]] = {
    "Demand History": [
        "demand_lag_1", "demand_lag_7", "demand_rolling_7_mean",
        "demand_rolling_7_std", "demand_trend_7"
    ],
    "Weather Conditions": [
        "temperature_c", "rainfall_mm", "rain_probability_pct",
        "is_heavy_rain", "is_extreme_weather", "weather_condition_encoded"
    ],
    "Pricing & Promotions": [
        "checkout_price", "base_price", "discount_amount", "discount_ratio",
        "emailer_for_promotion", "homepage_featured", "promo_interaction"
    ],
    "Calendar & Seasonality": [
        "day_of_week", "is_weekend", "is_holiday", "month",
        "day_of_month", "season_encoded"
    ],
    "Meal & Center Attributes": [
        "category_encoded", "cuisine_encoded", "center_type_encoded", "op_area"
    ]
}


def extract_feature_importance(
    model: Any,
    feature_names: List[str],
    model_name: str
) -> pd.DataFrame:
    """
    Extract normalized feature importances based on model type:
    - XGBoost / Random Forest: Feature importance (MDI / Gini / Gain)
    - Linear Regression: Absolute value of normalized coefficients
    - Naive Baseline: Equal or lag-based distribution
    """
    importances = np.zeros(len(feature_names))

    if hasattr(model, "feature_importances_"):
        # Tree-based ensemble (Random Forest, XGBoost)
        importances = np.array(model.feature_importances_)
    elif hasattr(model, "named_steps") and "regressor" in model.named_steps:
        # Scikit-learn Pipeline with Ridge / Linear Regression
        regressor = model.named_steps["regressor"]
        if hasattr(regressor, "coef_"):
            importances = np.abs(regressor.coef_)
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        # Fallback for baseline or non-standard models
        importances = np.ones(len(feature_names))

    # Normalize to sum to 100%
    total_imp = importances.sum()
    if total_imp > 0:
        normalized_imp = (importances / total_imp) * 100.0
    else:
        normalized_imp = np.full(len(feature_names), 100.0 / len(feature_names))

    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance_Score": importances,
        "Contribution_Pct": np.round(normalized_imp, 2)
    })

    # Add feature group classification
    def get_group(feat: str) -> str:
        for grp, members in FEATURE_GROUPS.items():
            if feat in members:
                return grp
        return "Other"

    df_imp["Group"] = df_imp["Feature"].apply(get_group)
    df_imp = df_imp.sort_values(by="Contribution_Pct", ascending=False).reset_index(drop=True)
    return df_imp


def compute_grouped_importance(df_feature_imp: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate individual feature importances into domain groups.
    Calculates exact percentage contribution of each feature group.
    """
    grouped = df_feature_imp.groupby("Group")["Contribution_Pct"].sum().reset_index()
    grouped.columns = ["Factor Category", "Total Contribution (%)"]
    grouped["Total Contribution (%)"] = np.round(grouped["Total Contribution (%)"], 2)
    grouped = grouped.sort_values(by="Total Contribution (%)", ascending=False).reset_index(drop=True)
    return grouped
