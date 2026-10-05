"""
Page 4: Closed-Loop Feedback & Post-Service Validation Engine.
Allows kitchen managers to log actual preparation and consumption figures,
computes true physical waste and prediction errors, tracks 7-day rolling MAE,
and monitors performance drift.
"""

from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.database import get_outcomes_df, get_predictions_df, get_rolling_mae_series
from src.demo_generator import CENTERS_DATA, MEALS_DATA
from src.feedback_loop import record_feedback, seed_initial_feedback_history
from src.retraining import evaluate_retraining_trigger
from src.ui.styles import get_plotly_dark_layout, render_app_header, render_kpi_card


def render_feedback_page(df_daily: pd.DataFrame, model_results, metadata) -> None:
    render_app_header(
        title="Closed-Loop Feedback & Outcome Validation",
        subtitle="Log post-service actuals, calculate prediction errors and physical food waste, and track live rolling accuracy."
    )

    best_mae = model_results.train_metadata["best_model_mae"]
    retrain_status = evaluate_retraining_trigger(baseline_benchmark_mae=best_mae)

    # Top Metric Cards
    df_outcomes = get_outcomes_df(limit=2000)
    total_entries = len(df_outcomes)
    total_waste_recorded = int(df_outcomes["actual_waste"].sum()) if not df_outcomes.empty else 0
    mean_abs_error = round(float(df_outcomes["abs_prediction_error"].mean()), 2) if not df_outcomes.empty else 0.0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card("Feedback Records", f"{total_entries:,}", "Total logged service shifts", badge="Database", badge_type="blue")
    with k2:
        render_kpi_card("Mean Actual Error", f"{mean_abs_error:.1f} Meals", "Average absolute variance", badge="Live MAE", badge_type="green" if mean_abs_error <= retrain_status.allowed_threshold_mae else "red")
    with k3:
        render_kpi_card("Total Waste Logged", f"{total_waste_recorded:,} Meals", "Prepared - Consumed", badge="Physical Spoilage", badge_type="amber")
    with k4:
        render_kpi_card("Model Health", retrain_status.status_label.split(" ")[0], f"Limit: {retrain_status.allowed_threshold_mae:.1f} MAE", badge="Retrain Monitor", badge_type="green" if not retrain_status.is_retraining_recommended else "red")

    st.markdown("---")

    # Form to Enter Actual Service Outcomes
    st.markdown("#### Record Post-Service Kitchen Outcomes")
    with st.expander("📝 Open Outcome Submission Form", expanded=True):
        with st.form("feedback_form", clear_on_submit=False):
            f_col1, f_col2, f_col3 = st.columns(3)

            with f_col1:
                yesterday = datetime.now().date() - timedelta(days=1)
                log_date = st.date_input("Service Date:", value=yesterday, key="fb_date")
                center_opts = {c["center_id"]: f"Center {c['center_id']} ({c['city']})" for c in CENTERS_DATA}
                selected_center = st.selectbox("Fulfilment Center:", list(center_opts.keys()), format_func=lambda x: center_opts[x], key="fb_cid")

            with f_col2:
                meal_opts = {m["meal_id"]: f"Meal {m['meal_id']} ({m['category']})" for m in MEALS_DATA}
                selected_meal = st.selectbox("Meal Item:", list(meal_opts.keys()), format_func=lambda x: meal_opts[x], key="fb_mid")
                pred_input = st.number_input("System Predicted Demand (Meals):", min_value=0, max_value=5000, value=300, step=5, key="fb_pred")

            with f_col3:
                prep_input = st.number_input("Actual Quantity Prepared (Meals):", min_value=0, max_value=5000, value=315, step=5, key="fb_prep")
                actual_dem_input = st.number_input("Actual Consumption / Demand (Meals):", min_value=0, max_value=5000, value=295, step=5, key="fb_act")

            notes_input = st.text_input("Operational Shift Notes (Optional):", placeholder="e.g., Heavy sudden rainfall during dinner service", key="fb_notes")

            submit_btn = st.form_submit_button("✓ Commit Feedback Outcome to Database", type="primary", use_container_width=True)

            if submit_btn:
                res = record_feedback(
                    target_date=log_date.strftime("%Y-%m-%d"),
                    center_id=selected_center,
                    meal_id=selected_meal,
                    predicted_demand=float(pred_input),
                    actual_prepared=int(prep_input),
                    actual_demand=int(actual_dem_input),
                    notes=notes_input
                )
                st.success(f"Outcome successfully registered! Actual Waste: {res.actual_waste} meals | Error: {res.prediction_error:+.1f} | 7D Rolling MAE: {res.rolling_mae_7d:.2f}")
                st.rerun()

    # Seed demo feedback data button if database has few entries
    if total_entries < 10:
        if st.button("🌱 Populate 21-Day Benchmark Historical Feedback Dataset", help="Inserts 21 days of realistic service logs to demonstrate rolling error and waste trends."):
            seeded = seed_initial_feedback_history(num_days=21, force_reseed=True)
            st.success(f"Inserted {seeded} historical feedback observations!")
            st.rerun()

    st.markdown("---")

    # Time Series Analysis: Rolling MAE & Waste Accumulation
    st.markdown("#### Rolling Prediction Accuracy & Spoilage Trends")
    rolling_df = get_rolling_mae_series(window_days=7)

    if not rolling_df.empty:
        c_chart1, c_chart2 = st.columns(2)

        with c_chart1:
            fig_mae = go.Figure()
            fig_mae.add_trace(go.Scatter(
                x=rolling_df["target_date"],
                y=rolling_df["rolling_mae"],
                mode="lines+markers",
                name="7D Rolling MAE",
                line=dict(color="#3B82F6", width=2.5)
            ))
            fig_mae.add_hline(y=best_mae, line_dash="dot", line_color="#10B981", annotation_text=f"Benchmark ({best_mae:.1f})")
            fig_mae.add_hline(y=retrain_status.allowed_threshold_mae, line_dash="dash", line_color="#EF4444", annotation_text=f"Retrain Limit ({retrain_status.allowed_threshold_mae:.1f})")
            fig_mae.update_layout(
                **get_plotly_dark_layout(title="7-Day Rolling Mean Absolute Error (MAE)", height=320),
                xaxis_title="Date",
                yaxis_title="Error (Meals)"
            )
            st.plotly_chart(fig_mae, use_container_width=True)

        with c_chart2:
            fig_waste = go.Figure()
            fig_waste.add_trace(go.Bar(
                x=rolling_df["target_date"],
                y=rolling_df["actual_waste"],
                name="Daily Actual Waste (Meals)",
                marker_color="#EF4444"
            ))
            fig_waste.update_layout(
                **get_plotly_dark_layout(title="Daily Actual Food Spoilage / Over-Prep (Meals)", height=320),
                xaxis_title="Date",
                yaxis_title="Waste (Meals)"
            )
            st.plotly_chart(fig_waste, use_container_width=True)

    # Detailed Audit Log Table
    st.markdown("#### Complete Feedback Audit Log Table")
    if not df_outcomes.empty:
        display_df = df_outcomes[[
            "id", "target_date", "center_id", "meal_id", "predicted_demand",
            "actual_prepared", "actual_demand", "actual_waste", "shortage",
            "prediction_error", "abs_prediction_error", "percentage_error", "notes"
        ]].copy()
        display_df.columns = [
            "ID", "Date", "Center", "Meal", "Predicted", "Prepared", "Actual Demand",
            "Waste", "Shortage", "Pred Error", "Abs Error", "% Error", "Notes"
        ]
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        csv_data = display_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export Feedback History CSV", data=csv_data, file_name="closed_loop_feedback_history.csv", mime="text/csv")
    else:
        st.info("No feedback records logged yet.")
