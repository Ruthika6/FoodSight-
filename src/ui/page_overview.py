"""
Page 1: Executive Overview Dashboard.
Displays high-level KPIs, next-day demand estimates, buffer recommendations,
weather context, model health status, and live feedback metrics.
"""

from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.database import get_outcomes_df, get_predictions_df, get_rolling_mae_series
from src.retraining import evaluate_retraining_trigger
from src.ui.styles import get_plotly_dark_layout, render_app_header, render_kpi_card
from src.weather_service import DEFAULT_CENTER_COORDINATES, fetch_open_meteo_forecast


def render_overview_page(df_daily: pd.DataFrame, model_results, metadata) -> None:
    render_app_header(
        title="Executive Overview & Intelligence Dashboard",
        subtitle="Real-time food demand forecasting, operational buffer planning, and closed-loop waste risk tracking."
    )

    # Calculate Overview Metrics
    best_name = model_results.best_model_name
    best_mae = model_results.train_metadata["best_model_mae"]

    # Retraining Status
    retrain_status = evaluate_retraining_trigger(baseline_benchmark_mae=best_mae)
    rolling_df = get_rolling_mae_series(window_days=7)
    live_mae = float(rolling_df["rolling_mae"].iloc[-1]) if not rolling_df.empty else best_mae
    total_waste = int(rolling_df["actual_waste"].sum()) if not rolling_df.empty else 0

    # Top KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        render_kpi_card(
            label="Active Model",
            value=best_name.split(" ")[0],
            subtext=f"Benchmark MAE: {best_mae:.2f}",
            badge="Optimized",
            badge_type="blue"
        )
    with col2:
        render_kpi_card(
            label="Live 7D Rolling MAE",
            value=f"{live_mae:.2f}",
            subtext=f"Tolerance: ≤ {retrain_status.allowed_threshold_mae:.2f}",
            badge="Live Error",
            badge_type="green" if live_mae <= retrain_status.allowed_threshold_mae else "red"
        )
    with col3:
        status_badge_type = "green" if not retrain_status.is_retraining_recommended else "red"
        render_kpi_card(
            label="System Health",
            value="Normal" if not retrain_status.is_retraining_recommended else "Retrain Req.",
            subtext=f"Breaches: {retrain_status.consecutive_breaches}/3",
            badge=retrain_status.status_label.split(" ")[0],
            badge_type=status_badge_type
        )
    with col4:
        render_kpi_card(
            label="Logged Waste (Meals)",
            value=f"{total_waste:,}",
            subtext="From closed feedback loop",
            badge="Tracked",
            badge_type="amber"
        )
    with col5:
        render_kpi_card(
            label="Data Provenance",
            value=metadata.source_mode.split(" ")[0],
            subtext="Simulated Day-Level",
            badge=metadata.source_mode,
            badge_type="blue"
        )

    st.markdown("---")

    # Center & Meal Quick View Filters
    st.markdown("#### Demand Trend & Multi-Day Forecasting Timeline")
    c_filter_col1, c_filter_col2 = st.columns([1, 1])
    with c_filter_col1:
        centers_list = sorted(df_daily["center_id"].unique())
        selected_center = st.selectbox("Select Fulfilment Center:", centers_list, format_func=lambda x: f"Center {x} ({DEFAULT_CENTER_COORDINATES.get(x, {}).get('city', 'Regional')})", key="overview_center")
    with c_filter_col2:
        meals_list = sorted(df_daily["meal_id"].unique())
        selected_meal = st.selectbox("Select Meal ID:", meals_list, key="overview_meal")

    # Time series chart for selected pair
    pair_df = df_daily[(df_daily["center_id"] == selected_center) & (df_daily["meal_id"] == selected_meal)].copy()
    pair_df = pair_df.sort_values(by="date").tail(45)

    if not pair_df.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pair_df["date"],
            y=pair_df["num_orders"],
            mode="lines+markers",
            name="Actual / Disaggregated Demand",
            line=dict(color="#3B82F6", width=2.5),
            marker=dict(size=5)
        ))

        # Add 7-day rolling average trace
        if len(pair_df) >= 7:
            pair_df["roll_7"] = pair_df["num_orders"].rolling(7, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=pair_df["date"],
                y=pair_df["roll_7"],
                mode="lines",
                name="7-Day Moving Average",
                line=dict(color="#10B981", width=2, dash="dash")
            ))

        fig.update_layout(
            **get_plotly_dark_layout(
                title=f"Daily Demand Pattern — Center {selected_center} | Meal {selected_meal} (Last 45 Days)",
                height=360
            ),
            xaxis_title="Date",
            yaxis_title="Meal Demand (Orders)",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No historical records available for selected center and meal pair.")

    st.markdown("---")

    # Two Column Layout: Weather Snapshot & Feedback Accuracy
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### Regional Weather & Environmental Context")
        w_records = []
        for cid in DEFAULT_CENTER_COORDINATES.keys():
            w = fetch_open_meteo_forecast(cid)
            w_records.append({
                "Center": f"Center {cid} ({w.city})",
                "Condition": w.weather_condition,
                "Temp (°C)": f"{w.temperature_c}°C",
                "Rain (mm)": f"{w.rainfall_mm} mm",
                "Rain Prob": f"{w.rain_probability_pct}%",
                "Source": "Live Open-Meteo" if "Live" in w.data_source else "Offline Demo"
            })
        df_w_table = pd.DataFrame(w_records)
        st.dataframe(df_w_table, use_container_width=True, hide_index=True)
        st.caption("ℹ Weather data is dynamically queried from Open-Meteo API with offline deterministic fallback.")

    with col_right:
        st.markdown("#### Closed-Loop Rolling MAE Tracking (Last 14 Days)")
        if not rolling_df.empty:
            recent_roll = rolling_df.tail(20)
            fig_roll = go.Figure()
            fig_roll.add_trace(go.Scatter(
                x=recent_roll["target_date"],
                y=recent_roll["rolling_mae"],
                mode="lines+markers",
                name="7D Rolling MAE",
                line=dict(color="#6366F1", width=2.5)
            ))
            # Benchmark baseline line
            fig_roll.add_hline(
                y=best_mae,
                line_dash="dot",
                line_color="#10B981",
                annotation_text=f"Benchmark ({best_mae:.1f})",
                annotation_position="bottom right"
            )
            # Retraining threshold line
            fig_roll.add_hline(
                y=retrain_status.allowed_threshold_mae,
                line_dash="dash",
                line_color="#EF4444",
                annotation_text=f"Retrain Limit ({retrain_status.allowed_threshold_mae:.1f})",
                annotation_position="top right"
            )
            fig_roll.update_layout(
                **get_plotly_dark_layout(title="7-Day Rolling MAE vs Safety Threshold", height=270),
                xaxis_title="Date",
                yaxis_title="MAE"
            )
            st.plotly_chart(fig_roll, use_container_width=True)
        else:
            st.info("No feedback entries recorded yet. Enter actual outcomes in the Feedback Loop page.")
