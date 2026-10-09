"""
Page 5: Model Performance, Benchmarking & Comparative Analytics.
Presents rigorous evaluation metrics across Naive Baseline, Linear Regression,
Random Forest, and XGBoost, with residual diagnostics and programmatic winner justification.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.ui.styles import get_plotly_dark_layout, render_app_header, render_kpi_card


def render_models_page(df_daily: pd.DataFrame, model_results, metadata) -> None:
    render_app_header(
        title="Model Performance & Comparative Benchmarking",
        subtitle="Empirical evaluation across baseline, linear, bagging, and gradient-boosted regression algorithms."
    )

    comp_table = model_results.comparison_table
    metrics = model_results.metrics
    best_name = model_results.best_model_name
    best_metric = metrics[best_name]

    # Top KPI Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card("Selected Model", best_name.split(" ")[0], best_name, badge="★ Winner", badge_type="blue")
    with c2:
        render_kpi_card("Optimal MAE", f"{best_metric.mae:.2f}", "Mean Absolute Error", badge="Primary Metric", badge_type="green")
    with c3:
        render_kpi_card("Root MSE", f"{best_metric.rmse:.2f}", "Root Mean Squared Error", badge="Secondary Metric", badge_type="blue")
    with c4:
        render_kpi_card("R² Determination", f"{best_metric.r2:.4f}", f"Explains {best_metric.r2*100:.1f}% variance", badge="Goodness of Fit", badge_type="green")

    st.markdown("---")

    # Model Comparison Table
    st.markdown("#### Regression Algorithm Comparison Matrix")
    st.dataframe(
        comp_table,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(f"""
    <div class="directive-box" style="border-left-color: #10B981;">
        <div class="directive-title">Programmatic Selection Rule:</div>
        <div class="directive-text">
            Models are evaluated strictly on the out-of-sample chronological test set. 
            The system sorts candidate models hierarchically: <strong>1. Lowest MAE (Primary)</strong> → <strong>2. Lowest RMSE</strong> → <strong>3. Highest R²</strong>. 
            The current active winner is <strong style="color: #34D399;">{best_name}</strong> ({best_metric.selection_reason}).
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Diagnostic Visualizations: Bar Charts & Scatter Fit
    col_v1, col_v2 = st.columns(2)

    with col_v1:
        st.markdown("#### Error Metric Comparison (MAE & RMSE)")
        fig_bars = go.Figure()
        fig_bars.add_trace(go.Bar(
            x=comp_table["Model"],
            y=comp_table["MAE"],
            name="MAE (Lower is Better)",
            marker_color="#3B82F6"
        ))
        fig_bars.add_trace(go.Bar(
            x=comp_table["Model"],
            y=comp_table["RMSE"],
            name="RMSE (Lower is Better)",
            marker_color="#F59E0B"
        ))
        fig_bars.update_layout(
            **get_plotly_dark_layout(height=340, barmode="group")
        )
        st.plotly_chart(fig_bars, use_container_width=True)

    with col_v2:
        st.markdown(f"#### Actual vs Predicted Fit ({best_name})")
        y_true = best_metric.residuals + best_metric.predictions
        y_pred = best_metric.predictions

        df_scatter = pd.DataFrame({"Actual": y_true, "Predicted": y_pred}).sample(min(400, len(y_true)), random_state=42)

        fig_scat = px.scatter(
            df_scatter,
            x="Actual",
            y="Predicted",
            opacity=0.65,
            color_discrete_sequence=["#3B82F6"],
            labels={"Actual": "Actual Demand", "Predicted": "Model Forecast"}
        )
        min_val = min(df_scatter["Actual"].min(), df_scatter["Predicted"].min())
        max_val = max(df_scatter["Actual"].max(), df_scatter["Predicted"].max())
        fig_scat.add_shape(
            type="line", line=dict(dash="dash", color="#EF4444", width=2),
            x0=min_val, y0=min_val, x1=max_val, y1=max_val
        )
        fig_scat.update_layout(
            **get_plotly_dark_layout(height=340)
        )
        st.plotly_chart(fig_scat, use_container_width=True)

    # Residual Distribution Histogram
    st.markdown("#### Residual Error Diagnostics")
    res_df = pd.DataFrame({"Residuals": best_metric.residuals})
    fig_hist = px.histogram(
        res_df,
        x="Residuals",
        nbins=40,
        color_discrete_sequence=["#6366F1"],
        marginal="box"
    )
    fig_hist.update_layout(
        **get_plotly_dark_layout(title=f"Residual Error Distribution (Actual - Predicted) for {best_name}", height=300)
    )
    st.plotly_chart(fig_hist, use_container_width=True)
