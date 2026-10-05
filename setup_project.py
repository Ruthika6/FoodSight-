"""
Setup and Initialization Script for the Adaptive Food Demand Forecasting System.
Performs full environment verification, benchmark dataset generation, ML model training,
artifact serialization, and SQLite database seeding.
"""

import sys
from pathlib import Path

# Add project root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import CONFIG
from src.database import get_outcomes_df, get_predictions_df, init_db
from src.data_loader import load_raw_or_demo_data
from src.feedback_loop import seed_initial_feedback_history
from src.feature_engineering import prepare_ml_splits
from src.model_training import train_all_models
from src.retraining import evaluate_retraining_trigger
from src.weather_service import fetch_open_meteo_forecast


def setup_all():
    print("=" * 65)
    print("ADAPTIVE FOOD DEMAND FORECASTING & WASTE RISK ESTIMATION SYSTEM")
    print("Manipal Institute of Technology, Bengaluru (CSE_3125)")
    print("Project Initializer & Environment Setup")
    print("=" * 65)

    # 1. Ensure project directories
    print("\n[1/5] Ensuring project directory structure...")
    CONFIG.paths.ensure_directories()
    print(f"  [OK] Raw Data Directory:       {CONFIG.paths.raw_data_dir}")
    print(f"  [OK] Demo Data Directory:      {CONFIG.paths.demo_data_dir}")
    print(f"  [OK] Processed Data Directory: {CONFIG.paths.processed_data_dir}")
    print(f"  [OK] Models Directory:         {CONFIG.paths.models_dir}")
    print(f"  [OK] SQLite Database:          {CONFIG.paths.database_path}")

    # 2. Ingest / Synthesize Benchmark Data
    print("\n[2/5] Loading and verifying demand dataset...")
    df_daily, metadata = load_raw_or_demo_data(force_demo=False)
    print(f"  [OK] Dataset Mode:             {metadata.source_mode}")
    print(f"  [OK] Granularity:              {metadata.granularity}")
    print(f"  [OK] Total Daily Records:      {metadata.num_rows:,} rows")
    print(f"  [OK] Distinct Centers:         {metadata.num_centers}")
    print(f"  [OK] Distinct Meals:           {metadata.num_meals}")
    print(f"  [OK] Date Range:               {metadata.start_date} to {metadata.end_date}")

    # 3. Train ML Model Suite
    print("\n[3/5] Engineering features and training ML model suite...")
    splits = prepare_ml_splits(df_daily, split_ratio=CONFIG.data.train_test_split_ratio)
    print(f"  [OK] Training Samples:         {len(splits.X_train):,} rows")
    print(f"  [OK] Chronological Test Set:   {len(splits.X_test):,} rows (Cutoff: {splits.split_date_threshold})")
    print(f"  [OK] Total Engineered Inputs:  {len(splits.feature_names)} features")

    results = train_all_models(splits, save_artifacts=True)
    print("\n  Candidate Model Evaluation Leaderboard:")
    print("  " + "-" * 55)
    for _, row in results.comparison_table.iterrows():
        status_marker = "*" if "Winner" in row["Status"] else " "
        print(f"  {status_marker} {row['Model']:<36} | MAE: {row['MAE']:>5.2f} | RMSE: {row['RMSE']:>5.2f} | R2: {row['R² Score']:>6.4f}")
    print("  " + "-" * 55)
    print(f"  [OK] Programmatic Selection:   {results.best_model_name}")

    # 4. Initialize Database & Seed Feedback History
    print("\n[4/5] Initializing SQLite database and seeding 21-day feedback history...")
    init_db()
    count = seed_initial_feedback_history(num_days=21, force_reseed=True)
    df_outcomes = get_outcomes_df()
    df_preds = get_predictions_df()
    print(f"  [OK] Initialized tables:       predictions, actual_outcomes, retraining_logs")
    print(f"  [OK] Seeded Service Outcomes:  {len(df_outcomes)} records")
    print(f"  [OK] Seeded Forecast Logs:     {len(df_preds)} records")

    # 5. Verify Weather & Drift System
    print("\n[5/5] Testing Open-Meteo weather service and drift monitor...")
    weather_sample = fetch_open_meteo_forecast(10)
    print(f"  [OK] Weather API Status:       {weather_sample.data_source} ({weather_sample.city}: {weather_sample.temperature_c} C, {weather_sample.weather_condition})")

    retrain_status = evaluate_retraining_trigger(baseline_benchmark_mae=results.train_metadata["best_model_mae"])
    clean_status = retrain_status.status_label.replace("✓", "[NORMAL]").replace("⚠", "[ALERT]")
    print(f"  [OK] Drift Monitor Status:     {clean_status}")
    print(f"  [OK] Baseline Benchmark MAE:   {retrain_status.baseline_mae:.2f}")
    print(f"  [OK] Retrain Error Limit:      <= {retrain_status.allowed_threshold_mae:.2f} MAE")

    print("\n" + "=" * 65)
    print("[SUCCESS] COMPLETE SYSTEM INITIALIZATION SUCCESSFUL!")
    print("Launch the dashboard at any time using:")
    print("    streamlit run app.py")
    print("=" * 65)


if __name__ == "__main__":
    setup_all()
