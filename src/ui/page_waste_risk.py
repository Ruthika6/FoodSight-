"""
Page 3: Waste & Shortage Risk Intelligence Engine.
Analyzes kitchen batch preparation schedules against AI forecasts,
computes over-preparation waste risk, shortage risk, economic cost impact,
and provides multi-item batch optimization simulations.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.demo_generator import CENTERS_DATA, MEALS_DATA
from src.prediction import predict_single_item
from src.ui.styles import get_plotly_dark_layout, render_app_header, render_kpi_card
from src.waste_risk import assess_waste_and_shortage_risk, calculate_preparation_recommendation


def render_waste_risk_page(df_daily: pd.DataFrame, model_results, metadata) -> None:
    render_app_header(
        title="Waste Risk Intelligence & Batch Optimization",
        subtitle="Evaluate kitchen batch allocations against predicted demand to eliminate surplus spoilage and stock-outs."
    )

    tab_single, tab_multi = st.tabs(["Single Item Risk Simulator", "Full Menu Batch Risk Audit"])

    with tab_single:
        st.markdown("#### Interactive Kitchen Batch Risk Simulator")
        col_ctrl, col_display = st.columns([1, 1.3], gap="large")

        with col_ctrl:
            sim_demand = st.number_input("Forecasted / Expected Demand (Meals):", min_value=10, max_value=2000, value=350, step=10, key="wr_dem")
            sim_prep = st.number_input("Planned Kitchen Preparation Batch (Meals):", min_value=10, max_value=2500, value=390, step=10, key="wr_prep")
            unit_cost = st.number_input("Raw Ingredient Unit Prep Cost (₹/meal):", min_value=20.0, max_value=500.0, value=85.0, step=5.0, key="wr_cost")

            st.markdown("##### Configurable Risk Thresholds")
            low_thresh = st.slider("Low Waste Surplus Limit (%):", 0.0, 10.0, 5.0, step=0.5, key="wr_low_th")
            med_thresh = st.slider("Medium Waste Surplus Limit (%):", 5.0, 30.0, 15.0, step=1.0, key="wr_med_th")
            crit_shortage = st.slider("Critical Shortage Deficit Limit (%):", 5.0, 25.0, 10.0, step=1.0, key="wr_crit_sh")

        with col_display:
            assessment = assess_waste_and_shortage_risk(
                planned_preparation=int(sim_prep),
                predicted_demand=float(sim_demand),
                low_waste_thresh=low_thresh,
                med_waste_thresh=med_thresh,
                crit_shortage_thresh=crit_shortage,
                unit_cost_inr=unit_cost
            )

            k1, k2, k3 = st.columns(3)
            with k1:
                render_kpi_card("Waste Risk Tier", assessment.waste_risk_level, f"Surplus: {assessment.surplus_percentage}%", badge="Over-Prep", badge_type="green" if assessment.waste_risk_level == "LOW" else ("amber" if assessment.waste_risk_level == "MEDIUM" else "red"))
            with k2:
                render_kpi_card("Shortage Risk Tier", assessment.shortage_risk_level, f"Deficit: {assessment.shortage_percentage}%", badge="Stock-Out", badge_type="green" if assessment.shortage_risk_level == "NONE" else "red")
            with k3:
                render_kpi_card("Expected Variance", f"{assessment.planned_preparation - assessment.predicted_demand:+d}", "Planned vs Forecast", badge="Delta", badge_type="blue")

            # Actionable directive box
            border_color = assessment.status_badge_color
            st.markdown(f"""
            <div class="directive-box" style="border-left-color: {border_color};">
                <div class="directive-title">Actionable Kitchen Advisory</div>
                <div class="directive-text">{assessment.actionable_recommendation}</div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 0.5rem;">{assessment.economic_impact_summary}</div>
            </div>
            """, unsafe_allow_html=True)

            # Gauge / Delta Bar Visual
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                name="Forecast Demand",
                x=["Batch Allocation"],
                y=[assessment.predicted_demand],
                marker_color="#3B82F6"
            ))
            fig_bar.add_trace(go.Bar(
                name="Planned Preparation",
                x=["Batch Allocation"],
                y=[assessment.planned_preparation],
                marker_color="#10B981" if assessment.waste_risk_level == "LOW" else ("#F59E0B" if assessment.waste_risk_level == "MEDIUM" else "#EF4444")
            ))
            fig_bar.update_layout(
                **get_plotly_dark_layout(
                    title="Batch Allocation: Preparation Plan vs ML Forecast",
                    height=300,
                    barmode="group",
                    margin=dict(t=50, b=25, l=20, r=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    with tab_multi:
        st.markdown("#### Full Menu Multi-Item Risk Audit")
        c_choice = st.selectbox("Select Fulfilment Center for Full Menu Audit:", [c["center_id"] for c in CENTERS_DATA], format_func=lambda x: f"Center {x} ({CENTERS_DATA[0]['city']})", key="wr_multi_c")

        # Generate menu audit table
        audit_rows = []
        for meal in MEALS_DATA:
            m_id = meal["meal_id"]
            m_name = f"{meal['category']} ({meal['cuisine']})"
            base_dem = meal["base_demand"]

            # Predict demand
            pred_obj = predict_single_item(
                model=model_results.best_model,
                model_name=model_results.best_model_name,
                feature_names=model_results.feature_names,
                categorical_encoders=model_results.categorical_encoders,
                center_id=c_choice,
                meal_id=m_id,
                target_date_str="2026-10-06",
                df_history=df_daily
            )
            pred_val = pred_obj.predicted_demand
            rec_obj = calculate_preparation_recommendation(pred_val, buffer_pct=5.0)
            rec_prep = rec_obj.recommended_preparation

            # Simulate arbitrary kitchen plan (some with surplus, some normal)
            sim_kitchen_plan = int(round(rec_prep * 1.08)) if m_id in [1062, 1207] else rec_prep

            risk_item = assess_waste_and_shortage_risk(sim_kitchen_plan, pred_val)

            audit_rows.append({
                "Meal ID": m_id,
                "Menu Item": m_name,
                "Predicted Demand": pred_val,
                "Recommended Prep (+5%)": rec_prep,
                "Simulated Prep Plan": sim_kitchen_plan,
                "Expected Surplus / Deficit": f"{sim_kitchen_plan - pred_val:+d}",
                "Waste Risk Level": risk_item.waste_risk_level,
                "Shortage Risk": risk_item.shortage_risk_level,
            })

        df_audit = pd.DataFrame(audit_rows)
        st.dataframe(df_audit, use_container_width=True, hide_index=True)

        # Plotly comparison across all items
        fig_multi = go.Figure()
        fig_multi.add_trace(go.Bar(
            x=df_audit["Menu Item"],
            y=df_audit["Predicted Demand"],
            name="Predicted Demand",
            marker_color="#3B82F6"
        ))
        fig_multi.add_trace(go.Bar(
            x=df_audit["Menu Item"],
            y=df_audit["Simulated Prep Plan"],
            name="Planned Preparation",
            marker_color="#F59E0B"
        ))
        fig_multi.update_layout(
            **get_plotly_dark_layout(
                title=f"Menu-Wide Preparation Plan vs Predicted Demand (Center {c_choice})",
                height=340,
                barmode="group"
            )
        )
        st.plotly_chart(fig_multi, use_container_width=True)
