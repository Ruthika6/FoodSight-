"""
Page 2: Next-Day Prediction & Preparation Engine.
Provides interactive inference controls, weather integration, buffer-adjusted preparation recommendations,
and direct persistence of forecast records into SQLite.
"""

from datetime import datetime, timedelta
import pandas as pd
import streamlit as st

from src.database import save_prediction
from src.demo_generator import CENTERS_DATA, MEALS_DATA
from src.prediction import get_meal_info_dict, predict_single_item
from src.ui.styles import render_app_header, render_kpi_card
from src.waste_risk import assess_waste_and_shortage_risk, calculate_preparation_recommendation
from src.weather_service import WeatherData, fetch_open_meteo_forecast


def render_forecast_page(df_daily: pd.DataFrame, model_results, metadata) -> None:
    render_app_header(
        title="Next-Day Demand Forecasting & Preparation Planner",
        subtitle="Generate AI-powered point forecasts, operational safety buffers, and waste risk assessments."
    )

    best_model = model_results.best_model
    best_model_name = model_results.best_model_name
    feature_names = model_results.feature_names
    cat_encoders = model_results.categorical_encoders

    # Layout: Control Panel & Live Output
    col_input, col_output = st.columns([1.1, 1.4], gap="large")

    with col_input:
        st.markdown("#### Forecast Configuration")

        # 1. Location & Meal
        center_options = {c["center_id"]: f"Center {c['center_id']} — {c['city']} ({c['center_type']})" for c in CENTERS_DATA}
        selected_cid = st.selectbox("Fulfilment Center:", list(center_options.keys()), format_func=lambda x: center_options[x], key="f_center")

        meal_options = {m["meal_id"]: f"Meal {m['meal_id']} — {m['category']} ({m['cuisine']})" for m in MEALS_DATA}
        selected_mid = st.selectbox("Meal Item:", list(meal_options.keys()), format_func=lambda x: meal_options[x], key="f_meal")

        meal_meta = get_meal_info_dict(selected_mid)
        default_price = float(meal_meta.get("base_price", 250.0))

        # 2. Date Selection
        next_day = datetime.now().date() + timedelta(days=1)
        target_date = st.date_input("Target Forecast Date:", value=next_day, key="f_date")
        target_date_str = target_date.strftime("%Y-%m-%d")

        # 3. Pricing & Promotions
        st.markdown("##### Commercial & Promotion Planning")
        price_col1, price_col2 = st.columns(2)
        with price_col1:
            planned_price = st.number_input(
                "Planned Price (₹):",
                min_value=50.0,
                max_value=1000.0,
                value=default_price,
                step=5.0,
                key="f_price"
            )
        with price_col2:
            st.metric("Base Benchmark Price", f"₹{default_price:.2f}")

        promo_col1, promo_col2 = st.columns(2)
        with promo_col1:
            emailer_promo = st.checkbox("Promotional Email Blast", value=False, key="f_email")
        with promo_col2:
            homepage_feat = st.checkbox("Featured on App Homepage", value=False, key="f_home")

        # 4. Weather Controls
        st.markdown("##### Weather & Environmental Conditions")
        weather_mode = st.radio("Weather Mode:", ["Auto-Fetch Live Forecast", "Manual Environmental Simulation"], horizontal=True, key="f_wmode")

        if weather_mode == "Auto-Fetch Live Forecast":
            weather_obj = fetch_open_meteo_forecast(selected_cid, target_date_str)
            st.info(f"Source: {weather_obj.data_source} | Condition: {weather_obj.weather_condition} ({weather_obj.temperature_c}°C, {weather_obj.rainfall_mm}mm rain)")
        else:
            w_col1, w_col2, w_col3 = st.columns(3)
            with w_col1:
                custom_temp = st.slider("Temperature (°C)", 10.0, 45.0, 28.0, step=0.5, key="f_ctemp")
            with w_col2:
                custom_rain = st.slider("Rainfall (mm)", 0.0, 60.0, 0.0, step=1.0, key="f_crain")
            with w_col3:
                custom_cond = st.selectbox("Condition", ["Clear", "Cloudy", "Rainy", "Stormy", "Extreme Heat"], key="f_ccond")

            weather_obj = WeatherData(
                center_id=selected_cid,
                city=CENTERS_DATA[0]["city"],
                target_date=target_date_str,
                temperature_c=custom_temp,
                rainfall_mm=custom_rain,
                rain_probability_pct=90.0 if custom_rain > 0 else 10.0,
                weather_condition=custom_cond,
                condition_description=f"Manual Simulation ({custom_cond})",
                is_extreme_weather=(custom_rain > 30.0 or custom_temp > 38.0 or custom_cond == "Stormy"),
                data_source="Manual User Simulation",
                status_message="Manual scenario override applied."
            )

        # 5. Operational Safety Buffer Slider
        st.markdown("##### Operational Kitchen Buffer")
        buffer_pct = st.slider("Safety Buffer (% extra prep)", min_value=0.0, max_value=25.0, value=5.0, step=0.5, key="f_buffer")

        generate_btn = st.button("🚀 Generate Next-Day Forecast", type="primary", use_container_width=True)

    with col_output:
        st.markdown("#### Forecast Results & Decision Support")

        # Execute prediction on load or button click
        forecast_result = predict_single_item(
            model=best_model,
            model_name=best_model_name,
            feature_names=feature_names,
            categorical_encoders=cat_encoders,
            center_id=selected_cid,
            meal_id=selected_mid,
            target_date_str=target_date_str,
            checkout_price=planned_price,
            emailer_promotion=1 if emailer_promo else 0,
            homepage_featured=1 if homepage_feat else 0,
            weather_override=weather_obj,
            df_history=df_daily
        )

        prep_rec = calculate_preparation_recommendation(
            predicted_demand=forecast_result.predicted_demand,
            buffer_pct=buffer_pct
        )

        # Primary Output Cards
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            render_kpi_card(
                label="Predicted Demand",
                value=f"{forecast_result.predicted_demand:,} Meals",
                subtext=f"Raw Model Estimate: {forecast_result.raw_model_prediction:.1f}",
                badge="ML Forecast",
                badge_type="blue"
            )
        with r_col2:
            render_kpi_card(
                label="Recommended Preparation",
                value=f"{prep_rec.recommended_preparation:,} Meals",
                subtext=f"Includes +{prep_rec.buffer_units} buffer meals ({buffer_pct:.1f}%)",
                badge="Actionable Target",
                badge_type="green"
            )

        # Operational Recommendation Card
        st.markdown(f"""
        <div class="directive-box">
            <div class="directive-title">Operational Decision Directive</div>
            <div class="directive-text">{prep_rec.operational_reason}</div>
        </div>
        """, unsafe_allow_html=True)

        # Risk Assessment
        risk = assess_waste_and_shortage_risk(
            planned_preparation=prep_rec.recommended_preparation,
            predicted_demand=forecast_result.predicted_demand
        )

        st.markdown("##### Risk Analysis & Buffer Calibration")
        risk_col1, risk_col2, risk_col3 = st.columns(3)
        with risk_col1:
            render_kpi_card("Waste Risk Tier", risk.waste_risk_level, f"Surplus: +{risk.expected_surplus} meals", badge="Perishability", badge_type="green" if risk.waste_risk_level == "LOW" else "amber")
        with risk_col2:
            render_kpi_card("Shortage Risk Tier", risk.shortage_risk_level, f"Deficit: {risk.expected_shortage} meals", badge="Stock-Out", badge_type="green" if risk.shortage_risk_level == "NONE" else "red")
        with risk_col3:
            render_kpi_card("Active Algorithm", best_model_name.split(" ")[0], f"Trained on {len(df_daily):,} records", badge="Best ML", badge_type="blue")

        # Feature Influence Context Table
        st.markdown("##### Key Influencing Factor Snapshot")
        feat_df = pd.DataFrame([
            {"Feature Dimension": "Previous Day Demand (Lag 1)", "Value": f"{forecast_result.historical_lag_1:.0f} meals", "Impact": "High Auto-regressive Anchor"},
            {"Feature Dimension": "Same Day Last Week (Lag 7)", "Value": f"{forecast_result.historical_lag_7:.0f} meals", "Impact": "Weekly Cyclical Pattern"},
            {"Feature Dimension": "7-Day Moving Average", "Value": f"{forecast_result.historical_rolling_7:.1f} meals", "Impact": "Baseline Demand Level"},
            {"Feature Dimension": "Weather Condition", "Value": f"{forecast_result.weather.weather_condition} ({forecast_result.weather.temperature_c}°C)", "Impact": "Footfall Sensitivity"},
            {"Feature Dimension": "Promotions Active", "Value": "Yes (+Email/Featured)" if forecast_result.is_promo else "No Promotion", "Impact": "Demand Elasticity Uplift"},
        ])
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

        # Database Save Action
        if st.button("💾 Save Forecast to History Database", use_container_width=True):
            pid = save_prediction(
                prediction_date=datetime.now().strftime("%Y-%m-%d"),
                target_date=target_date_str,
                center_id=selected_cid,
                meal_id=selected_mid,
                predicted_demand=forecast_result.predicted_demand,
                recommended_prep=prep_rec.recommended_preparation,
                buffer_pct=buffer_pct,
                model_used=best_model_name,
                weather_condition=forecast_result.weather.weather_condition,
                temperature=forecast_result.weather.temperature_c,
                rainfall=forecast_result.weather.rainfall_mm,
                planned_price=planned_price,
                emailer_promotion=1 if emailer_promo else 0,
                homepage_featured=1 if homepage_feat else 0
            )
            st.success(f"✓ Forecast record successfully registered in SQLite database with ID #{pid}!")
