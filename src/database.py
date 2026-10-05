"""
Database persistence module using SQLite.
Stores prediction records, actual kitchen outcomes, and retraining audit logs.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sqlite3
import pandas as pd
from src.config import CONFIG


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Get a SQLite database connection with row factory."""
    path = db_path or CONFIG.paths.database_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """Initialize SQLite database schema."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prediction_date TEXT NOT NULL,
        target_date TEXT NOT NULL,
        center_id INTEGER NOT NULL,
        meal_id INTEGER NOT NULL,
        predicted_demand REAL NOT NULL,
        recommended_prep INTEGER NOT NULL,
        buffer_pct REAL NOT NULL,
        model_used TEXT NOT NULL,
        weather_condition TEXT,
        temperature REAL,
        rainfall REAL,
        planned_price REAL,
        emailer_promotion INTEGER DEFAULT 0,
        homepage_featured INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS actual_outcomes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prediction_id INTEGER,
        target_date TEXT NOT NULL,
        center_id INTEGER NOT NULL,
        meal_id INTEGER NOT NULL,
        predicted_demand REAL NOT NULL,
        actual_prepared INTEGER NOT NULL,
        actual_demand INTEGER NOT NULL,
        actual_waste INTEGER NOT NULL,
        shortage INTEGER NOT NULL,
        prediction_error REAL NOT NULL,
        abs_prediction_error REAL NOT NULL,
        percentage_error REAL,
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(prediction_id) REFERENCES predictions(id)
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS retraining_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        trigger_reason TEXT NOT NULL,
        baseline_mae REAL,
        recent_rolling_mae REAL,
        status TEXT NOT NULL,
        new_model_name TEXT,
        new_mae REAL,
        details TEXT
    );
    """)

    # Indices for performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_target ON predictions(target_date, center_id, meal_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_outcomes_target ON actual_outcomes(target_date, center_id, meal_id);")

    conn.commit()
    conn.close()


def save_prediction(
    prediction_date: str,
    target_date: str,
    center_id: int,
    meal_id: int,
    predicted_demand: float,
    recommended_prep: int,
    buffer_pct: float,
    model_used: str,
    weather_condition: Optional[str] = None,
    temperature: Optional[float] = None,
    rainfall: Optional[float] = None,
    planned_price: Optional[float] = None,
    emailer_promotion: int = 0,
    homepage_featured: int = 0,
    db_path: Optional[Path] = None,
) -> int:
    """Save a next-day prediction record to database."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (
            prediction_date, target_date, center_id, meal_id,
            predicted_demand, recommended_prep, buffer_pct, model_used,
            weather_condition, temperature, rainfall, planned_price,
            emailer_promotion, homepage_featured
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        prediction_date, target_date, center_id, meal_id,
        predicted_demand, recommended_prep, buffer_pct, model_used,
        weather_condition, temperature, rainfall, planned_price,
        emailer_promotion, homepage_featured
    ))
    pred_id = cursor.lastrowid or 0
    conn.commit()
    conn.close()
    return pred_id


def log_actual_outcome(
    target_date: str,
    center_id: int,
    meal_id: int,
    predicted_demand: float,
    actual_prepared: int,
    actual_demand: int,
    prediction_id: Optional[int] = None,
    notes: str = "",
    db_path: Optional[Path] = None,
) -> int:
    """
    Log actual kitchen outcomes and calculate error metrics.
    Actual Waste = max(0, Actual Prepared - Actual Demand)
    Shortage = max(0, Actual Demand - Actual Prepared)
    Prediction Error = Predicted Demand - Actual Demand
    Absolute Prediction Error = |Predicted Demand - Actual Demand|
    """
    init_db(db_path)
    actual_waste = max(0, actual_prepared - actual_demand)
    shortage = max(0, actual_demand - actual_prepared)
    pred_error = round(float(predicted_demand - actual_demand), 2)
    abs_error = round(abs(float(predicted_demand - actual_demand)), 2)
    pct_error = round((abs_error / actual_demand * 100.0) if actual_demand > 0 else 0.0, 2)

    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO actual_outcomes (
            prediction_id, target_date, center_id, meal_id,
            predicted_demand, actual_prepared, actual_demand,
            actual_waste, shortage, prediction_error, abs_prediction_error,
            percentage_error, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        prediction_id, target_date, center_id, meal_id,
        predicted_demand, actual_prepared, actual_demand,
        actual_waste, shortage, pred_error, abs_error,
        pct_error, notes
    ))
    outcome_id = cursor.lastrowid or 0
    conn.commit()
    conn.close()
    return outcome_id


def get_predictions_df(db_path: Optional[Path] = None, limit: int = 1000) -> pd.DataFrame:
    """Retrieve predictions as a pandas DataFrame."""
    init_db(db_path)
    conn = get_connection(db_path)
    query = "SELECT * FROM predictions ORDER BY id DESC LIMIT ?"
    df = pd.read_sql_query(query, conn, params=(limit,))
    conn.close()
    return df


def get_outcomes_df(db_path: Optional[Path] = None, limit: int = 1000) -> pd.DataFrame:
    """Retrieve actual outcomes as a pandas DataFrame."""
    init_db(db_path)
    conn = get_connection(db_path)
    query = "SELECT * FROM actual_outcomes ORDER BY target_date ASC, id ASC LIMIT ?"
    df = pd.read_sql_query(query, conn, params=(limit,))
    conn.close()
    return df


def get_rolling_mae_series(
    db_path: Optional[Path] = None,
    window_days: int = 7,
    center_id: Optional[int] = None,
    meal_id: Optional[int] = None
) -> pd.DataFrame:
    """
    Compute rolling MAE and cumulative metrics over time from actual outcomes.
    """
    df = get_outcomes_df(db_path, limit=5000)
    if df.empty:
        return pd.DataFrame(columns=[
            "target_date", "center_id", "meal_id", "actual_demand",
            "predicted_demand", "abs_prediction_error", "rolling_mae",
            "actual_waste", "shortage"
        ])

    if center_id is not None:
        df = df[df["center_id"] == center_id]
    if meal_id is not None:
        df = df[df["meal_id"] == meal_id]

    if df.empty:
        return pd.DataFrame()

    df = df.sort_values(by=["target_date", "id"]).reset_index(drop=True)
    df["rolling_mae"] = df["abs_prediction_error"].rolling(window=window_days, min_periods=1).mean().round(2)
    df["rolling_waste"] = df["actual_waste"].rolling(window=window_days, min_periods=1).mean().round(2)
    return df


def log_retraining_event(
    trigger_reason: str,
    baseline_mae: Optional[float],
    recent_rolling_mae: Optional[float],
    status: str,
    new_model_name: Optional[str] = None,
    new_mae: Optional[float] = None,
    details: str = "",
    db_path: Optional[Path] = None
) -> int:
    """Record an audit entry for model retraining or status check."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO retraining_logs (
            trigger_reason, baseline_mae, recent_rolling_mae,
            status, new_model_name, new_mae, details
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        trigger_reason, baseline_mae, recent_rolling_mae,
        status, new_model_name, new_mae, details
    ))
    log_id = cursor.lastrowid or 0
    conn.commit()
    conn.close()
    return log_id


def get_retraining_logs_df(db_path: Optional[Path] = None, limit: int = 50) -> pd.DataFrame:
    """Retrieve retraining audit logs."""
    init_db(db_path)
    conn = get_connection(db_path)
    query = "SELECT * FROM retraining_logs ORDER BY id DESC LIMIT ?"
    df = pd.read_sql_query(query, conn, params=(limit,))
    conn.close()
    return df
