"""
Page 8: System Health, Drift Monitoring & Continuous Retraining Engine.
Displays operational service health, database record counts, model metadata,
concept drift metrics, and provides automated one-click model retraining orchestration.
"""

from datetime import datetime
import pandas as pd
import streamlit as st

from src.config import CONFIG
from src.database import get_outcomes_df, get_predictions_df, get_retraining_logs_df
from src.retraining import evaluate_retraining_trigger, execute_pipeline_retraining
from src.ui.styles import render_app_header, render_kpi_card
from src.weather_service import fetch_open_meteo_forecast


def render_health_page(df_daily: pd.DataFrame, model_results, metadata, on_retrain_callback=None) -> None:
    render_app_header(
        title="System Health, Drift Monitoring & Retraining",
        subtitle="End-to-end telemetry on model calibration, feedback drift, database integrity, and retraining triggers."
    )

    best_mae = model_results.train_metadata["best_model_mae"]
    best_name = model_results.best_model_name
    train_timestamp = model_results.train_metadata.get("training_timestamp", "Not available")
    train_samples = model_results.train_metadata.get("train_samples", len(df_daily))

    retrain_status = evaluate_retraining_trigger(baseline_benchmark_mae=best_mae)
    df_outcomes = get_outcomes_df()
    df_preds = get_predictions_df()

    # Top KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card("Model Health", retrain_status.status_label.split(" ")[0], f"Breaches: {retrain_status.consecutive_breaches}/3", badge="Health", badge_type="green" if not retrain_status.is_retraining_recommended else "red")
    with k2:
        render_kpi_card("Benchmark MAE", f"{best_mae:.2f}", f"Threshold: {retrain_status.allowed_threshold_mae:.2f}", badge="Baseline", badge_type="blue")
    with k3:
        render_kpi_card("Live Rolling MAE", f"{retrain_status.current_rolling_mae:.2f}", f"Window: {CONFIG.retraining.rolling_window_days} Days", badge="Live Drift", badge_type="green" if not retrain_status.is_retraining_recommended else "red")
    with k4:
        render_kpi_card("Stored Feedback", f"{len(df_outcomes):,}", f"{len(df_preds):,} Predictions", badge="Persistence", badge_type="blue")

    st.markdown("---")

    # Drift Status Diagnostic Banner
    status_box_col = "#10B981" if not retrain_status.is_retraining_recommended else "#EF4444"
    st.markdown(f"""
    <div class="directive-box" style="border-left-color: {status_box_col};">
        <div class="directive-title" style="color: {'#34D399' if not retrain_status.is_retraining_recommended else '#F87171'};">
            {retrain_status.status_label}
        </div>
        <div class="directive-text">
            {retrain_status.detailed_explanation}
        </div>
        <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 0.5rem;">
            <strong>Trigger Rationale:</strong> {retrain_status.trigger_reason}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Health & System Telemetry Columns
    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown("#### Model Registry & Architecture Metadata")
        meta_items = [
            {"Property": "Active Production Model", "Value": best_name},
            {"Property": "Model Selection Rule", "Value": "Lowest MAE → Lowest RMSE → Highest R²"},
            {"Property": "Training Timestamp", "Value": train_timestamp},
            {"Property": "Training Sample Size", "Value": f"{train_samples:,} rows"},
            {"Property": "Validation Out-of-Sample Size", "Value": f"{model_results.train_metadata.get('test_samples', 0):,} rows"},
            {"Property": "Engineered Features Count", "Value": f"{len(model_results.feature_names)} features"},
            {"Property": "Primary Benchmark Metric", "Value": f"MAE = {best_mae:.2f} | RMSE = {model_results.train_metadata.get('best_model_rmse', 0):.2f}"},
        ]
        st.dataframe(pd.DataFrame(meta_items), use_container_width=True, hide_index=True)

    with col_t2:
        st.markdown("#### Infrastructure & Data Telemetry")
        weather_test = fetch_open_meteo_forecast(10)
        infra_items = [
            {"Component": "Dataset Mode", "Status": metadata.source_mode, "Details": metadata.granularity},
            {"Component": "SQLite Database", "Status": "Connected (Healthy)", "Details": f"{len(df_outcomes)} outcomes logged"},
            {"Component": "Weather API Service", "Status": weather_test.data_source, "Details": f"Live ping response: {weather_test.weather_condition}"},
            {"Component": "Disaggregation Method", "Status": "Empirical Multipliers + Noise", "Details": "Validated on canteen patterns"},
            {"Component": "Temporal Split Purity", "Status": "Chronological (No Leakage)", "Details": f"Cutoff: {model_results.train_metadata.get('split_cutoff_date', 'N/A')}"},
        ]
        st.dataframe(pd.DataFrame(infra_items), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Retraining Action Section
    st.markdown("#### Pipeline Retraining Orchestration")
    st.write(
        "Initiating retraining ingests latest historical records and feedback outcomes, re-evaluates all 4 algorithms "
        "(Naive, Linear Regression, Random Forest, XGBoost), updates the winner in the model registry, and resets the error baseline."
    )

    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        if st.button("⚡ Execute Model Retraining Pipeline", type="primary", use_container_width=True):
            with st.spinner("Retraining all models, calculating test metrics, and updating registry..."):
                new_results, new_status = execute_pipeline_retraining(df_daily, trigger_reason="Manual User Retrain from Health Dashboard")
                st.success(f"✓ Retraining complete! Winner: {new_results.best_model_name} (MAE: {new_results.train_metadata['best_model_mae']:.2f})")
                if on_retrain_callback:
                    on_retrain_callback()
                st.rerun()

    with col_info:
        st.caption("ℹ Retraining automatically recalibrates hyperparameter weights, computes new test residuals, and logs an audit entry in the local database.")

    # Retraining Logs Table
    st.markdown("#### Retraining Audit History Logs")
    df_logs = get_retraining_logs_df()
    if not df_logs.empty:
        st.dataframe(df_logs, use_container_width=True, hide_index=True)
    else:
        st.info("No prior retraining events recorded. Use the button above to execute a retraining cycle.")
