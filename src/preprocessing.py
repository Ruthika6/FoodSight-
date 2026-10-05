"""
Preprocessing and Data Sanitization Module.
Handles type conversions, null imputation, outlier bounds, and duplicate removal.
"""

from typing import Tuple
import numpy as np
import pandas as pd


def clean_and_sanitize_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Sanitize raw data:
    1. Parse dates and sort chronologically.
    2. Remove duplicate (date, center_id, meal_id) entries.
    3. Validate and bound numerical variables.
    4. Fill missing values using domain-appropriate defaults.
    """
    report = {
        "initial_rows": len(df),
        "duplicates_removed": 0,
        "nulls_handled": {},
        "invalid_values_clipped": 0,
    }

    df_clean = df.copy()

    # 1. Date normalization
    if "date" in df_clean.columns:
        df_clean["date"] = pd.to_datetime(df_clean["date"])
    else:
        raise ValueError("Dataset missing required 'date' column for time-series processing.")

    # 2. Duplicate checking
    dedup_subset = ["date", "center_id", "meal_id"]
    initial_count = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=dedup_subset, keep="last")
    report["duplicates_removed"] = initial_count - len(df_clean)

    # 3. Handle nulls
    null_counts = df_clean.isnull().sum()
    for col, count in null_counts.items():
        if count > 0:
            report["nulls_handled"][col] = int(count)
            if col in ["num_orders", "checkout_price", "base_price", "temperature_c", "rainfall_mm"]:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            elif col in ["emailer_for_promotion", "homepage_featured", "is_weekend"]:
                df_clean[col] = df_clean[col].fillna(0)
            else:
                df_clean[col] = df_clean[col].fillna("Unknown")

    # 4. Numerical constraints (clip negative values)
    if "num_orders" in df_clean.columns:
        invalid_orders = (df_clean["num_orders"] < 0).sum()
        if invalid_orders > 0:
            report["invalid_values_clipped"] += int(invalid_orders)
            df_clean["num_orders"] = df_clean["num_orders"].clip(lower=0)

    if "checkout_price" in df_clean.columns:
        df_clean["checkout_price"] = df_clean["checkout_price"].clip(lower=10.0)

    if "base_price" in df_clean.columns:
        df_clean["base_price"] = df_clean["base_price"].clip(lower=10.0)

    # Sort strictly by date, center_id, meal_id
    df_clean = df_clean.sort_values(by=["date", "center_id", "meal_id"]).reset_index(drop=True)
    report["final_rows"] = len(df_clean)

    return df_clean, report
