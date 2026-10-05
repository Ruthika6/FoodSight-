"""
Page 6: Feature Importance & Interpretability Dashboard.
Visualizes tree-based/coefficient feature importances, computes grouped domain factor contributions,
and explores environmental weather sensitivities.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.feature_importance import compute_grouped_importance, extract_feature_importance
from src.ui.styles import get_plotly_dark_layout, render_app_header, render_kpi_card


def render_importance_page(df_daily: pd.DataFrame, model_results, metadata) -> None:
    render_app_header(
        title="Feature Importance & Interpretability Analysis",
        subtitle="Quantify the relative predictive influence of demand history, weather forecasts, pricing, promotions, and calendar seasonality."
    )

    best_model = model_results.best_model
    best_name = model_results.best_model_name
    feature_names = model_results.feature_names

    # Extract importance
    df_feat_imp = extract_feature_importance(best_model, feature_names, best_name)
    df_grouped = compute_grouped_importance(df_feat_imp)

    # Top KPI Cards for Group Contributions
    top_group = df_grouped.iloc[0]
    weather_group_row = df_grouped[df_grouped["Factor Category"] == "Weather Conditions"]
    weather_pct = float(weather_group_row["Total Contribution (%)"].iloc[0]) if not weather_group_row.empty else 0.0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card("Dominant Factor", top_group["Factor Category"], f"Contribution: {top_group['Total Contribution (%)']}%", badge="Primary Driver", badge_type="blue")
    with k2:
        render_kpi_card("Weather Contribution", f"{weather_pct:.1f}%", "Rainfall & Temperature impact", badge="Environmental", badge_type="amber")
    with k3:
        render_kpi_card("Total Features", f"{len(feature_names)}", "Engineered inputs", badge="Complexity", badge_type="blue")
    with k4:
        render_kpi_card("Interpreter Engine", "Tree Gini / Gain" if "Forest" in best_name or "XGBoost" in best_name else "Linear Coef", best_name.split(" ")[0], badge="Model-Based", badge_type="green")

    st.markdown("---")

    # Grouped Contribution Donut & Bar Charts
    col_g1, col_g2 = st.columns([1.1, 1.2])

    with col_g1:
        st.markdown("#### Grouped Factor Contribution Share")
        fig_donut = px.pie(
            df_grouped,
            names="Factor Category",
            values="Total Contribution (%)",
            hole=0.45,
            color_discrete_sequence=["#3B82F6", "#10B981", "#F59E0B", "#8B5CF6", "#EC4899"]
        )
        fig_donut.update_layout(
            **get_plotly_dark_layout(height=340),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_g2:
        st.markdown("#### Domain Group Contribution Breakdown")
        fig_grp_bar = px.bar(
            df_grouped,
            x="Total Contribution (%)",
            y="Factor Category",
            orientation="h",
            text="Total Contribution (%)",
            color="Factor Category",
            color_discrete_sequence=["#3B82F6", "#10B981", "#F59E0B", "#8B5CF6", "#EC4899"]
        )
        fig_grp_bar.update_layout(
            **get_plotly_dark_layout(height=340),
            showlegend=False,
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_grp_bar, use_container_width=True)

    st.markdown(f"""
    <div class="directive-box">
        <div class="directive-title">Methodology Note:</div>
        <div class="directive-text">
            Group contributions are computed by summing the normalized feature importances within each domain group:
            <code>Group Contribution (%) = ( Σ Importance(f_i) / Σ Total Importance ) × 100%</code>.
            Demand history represents the auto-regressive baseline, while weather conditions capture footfall variations induced by rain or extreme temperature.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Top Individual Features Horizontal Bar Chart
    st.markdown("#### Top 15 Individual Predictive Features")
    top_15 = df_feat_imp.head(15)

    fig_top = px.bar(
        top_15,
        x="Contribution_Pct",
        y="Feature",
        orientation="h",
        color="Group",
        text="Contribution_Pct",
        labels={"Contribution_Pct": "Relative Importance (%)", "Feature": "Engineered Feature"},
        color_discrete_map={
            "Demand History": "#3B82F6",
            "Weather Conditions": "#10B981",
            "Pricing & Promotions": "#F59E0B",
            "Calendar & Seasonality": "#8B5CF6",
            "Meal & Center Attributes": "#EC4899",
            "Other": "#94A3B8"
        }
    )
    fig_top.update_layout(
        **get_plotly_dark_layout(height=450),
        yaxis=dict(autorange="reversed")
    )
    st.plotly_chart(fig_top, use_container_width=True)

    # Detailed Table
    with st.expander("🔍 View Complete Feature Importance Table"):
        st.dataframe(df_feat_imp, use_container_width=True, hide_index=True)
