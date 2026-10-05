"""
Data Loader Module.
Loads raw Kaggle / Genpact food demand data or realistic demo datasets.
Performs strict schema validation, provenance tagging, and error reporting.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd
from src.config import CONFIG
from src.demo_generator import generate_demo_dataset
from src.disaggregation import disaggregate_weekly_dataframe
from src.weather_service import DEFAULT_CENTER_COORDINATES, generate_fallback_weather


REQUIRED_TRAIN_COLUMNS = {"week", "center_id", "meal_id", "checkout_price", "base_price", "emailer_for_promotion", "homepage_featured", "num_orders"}
REQUIRED_CENTERS_COLUMNS = {"center_id", "city_code", "region_code", "center_type", "op_area"}
REQUIRED_MEALS_COLUMNS = {"meal_id", "category", "cuisine"}


@dataclass
class DatasetMetadata:
    source_mode: str  # "REAL DATA" or "DEMO DATA"
    granularity: str  # "SIMULATED DAY-LEVEL DATA (Disaggregated from Weekly Totals)"
    num_rows: int
    num_centers: int
    num_meals: int
    start_date: str
    end_date: str
    file_origin: Dict[str, str]
    is_valid: bool
    validation_notes: list


def check_raw_dataset_exists(raw_dir: Optional[Path] = None) -> bool:
    """Check if all required real dataset CSV files exist in raw_dir."""
    target_dir = raw_dir or CONFIG.paths.raw_data_dir
    train_file = target_dir / "train.csv"
    centers_file = target_dir / "fulfilment_center_info.csv"
    meals_file = target_dir / "meal_info.csv"
    return train_file.exists() and centers_file.exists() and meals_file.exists()


def validate_schema(
    df: pd.DataFrame,
    required_cols: set,
    dataset_name: str
) -> Tuple[bool, list]:
    """Validate DataFrame columns and check for basic structural integrity."""
    notes = []
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        notes.append(f"ERROR: {dataset_name} missing required columns: {missing_cols}")
        return False, notes

    # Check for empty dataframe
    if df.empty:
        notes.append(f"ERROR: {dataset_name} is empty.")
        return False, notes

    # Null value summary
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if not null_cols.empty:
        notes.append(f"WARNING: {dataset_name} contains null values in columns: {null_cols.to_dict()}")

    return True, notes


def load_raw_or_demo_data(
    force_demo: bool = False,
    raw_dir: Optional[Path] = None,
    demo_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, DatasetMetadata]:
    """
    Load food demand dataset.
    Prioritizes real raw data if available and force_demo=False.
    Otherwise loads or generates the benchmark demo dataset.
    Returns merged day-level DataFrame and rich metadata.
    """
    raw_path = raw_dir or CONFIG.paths.raw_data_dir
    demo_path = demo_dir or CONFIG.paths.demo_data_dir
    validation_notes = []

    has_real = check_raw_dataset_exists(raw_path) and not force_demo

    if has_real:
        source_mode = "REAL DATA"
        train_file = raw_path / "train.csv"
        centers_file = raw_path / "fulfilment_center_info.csv"
        meals_file = raw_path / "meal_info.csv"

        df_train_weekly = pd.read_csv(train_file)
        df_centers = pd.read_csv(centers_file)
        df_meals = pd.read_csv(meals_file)

        # Validate schema of real data
        v1, n1 = validate_schema(df_train_weekly, REQUIRED_TRAIN_COLUMNS, "train.csv")
        v2, n2 = validate_schema(df_centers, REQUIRED_CENTERS_COLUMNS, "fulfilment_center_info.csv")
        v3, n3 = validate_schema(df_meals, REQUIRED_MEALS_COLUMNS, "meal_info.csv")
        validation_notes.extend(n1 + n2 + n3)

        if not (v1 and v2 and v3):
            raise ValueError(f"Real dataset schema validation failed: {validation_notes}")

        # Map center city names if missing
        if "city" not in df_centers.columns:
            df_centers["city"] = df_centers["center_id"].apply(
                lambda cid: DEFAULT_CENTER_COORDINATES.get(cid, {}).get("city", f"Center-{cid}")
            )

        # Disaggregate weekly records into day-level data
        df_daily = disaggregate_weekly_dataframe(df_train_weekly, start_date="2024-01-01", seed=42)
        df_daily = df_daily.merge(df_centers, on="center_id", how="left")
        df_daily = df_daily.merge(df_meals, on="meal_id", how="left")

        # Synthesize realistic historical weather for the training period
        weather_records = []
        for _, row in df_daily.iterrows():
            c_id = int(row["center_id"])
            c_city = str(row.get("city", f"Center-{c_id}"))
            d_str = str(row["date"])
            w = generate_fallback_weather(c_id, d_str, c_city)
            weather_records.append({
                "temperature_c": w.temperature_c,
                "rainfall_mm": w.rainfall_mm,
                "rain_probability_pct": w.rain_probability_pct,
                "weather_condition": w.weather_condition,
            })
        df_weather = pd.DataFrame(weather_records)
        df_daily = pd.concat([df_daily, df_weather], axis=1)

        file_origins = {
            "train": str(train_file),
            "centers": str(centers_file),
            "meals": str(meals_file),
        }

    else:
        source_mode = "DEMO DATA"
        demo_benchmark_file = demo_path / "daily_demand_with_weather.csv"

        if not demo_benchmark_file.exists():
            generate_demo_dataset(demo_path)

        df_daily = pd.read_csv(demo_benchmark_file)
        file_origins = {
            "demo_daily_benchmark": str(demo_benchmark_file),
            "description": "Standardized Genpact benchmark with synthetic Indian climatic weather"
        }
        validation_notes.append("Using pre-generated benchmark demo dataset.")

    # Sort chronologically to preserve time structure
    df_daily["date"] = pd.to_datetime(df_daily["date"])
    df_daily = df_daily.sort_values(by=["date", "center_id", "meal_id"]).reset_index(drop=True)

    metadata = DatasetMetadata(
        source_mode=source_mode,
        granularity="SIMULATED DAY-LEVEL DATA (Disaggregated from Weekly Totals)",
        num_rows=len(df_daily),
        num_centers=int(df_daily["center_id"].nunique()),
        num_meals=int(df_daily["meal_id"].nunique()),
        start_date=df_daily["date"].min().strftime("%Y-%m-%d"),
        end_date=df_daily["date"].max().strftime("%Y-%m-%d"),
        file_origin=file_origins,
        is_valid=True,
        validation_notes=validation_notes
    )

    return df_daily, metadata
