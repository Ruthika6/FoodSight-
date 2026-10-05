"""
Page 9: Academic Synopsis, Theoretical Methodology & Project Documentation.
Contains complete project details for Manipal Institute of Technology Bengaluru V Semester Mini Project.
"""

import streamlit as st
from src.ui.styles import render_app_header


def render_about_page(df_daily, model_results, metadata) -> None:
    render_app_header(
        title="Project Synopsis & Academic Methodology",
        subtitle="Adaptive Food Demand Forecasting and Waste Risk Estimation System: A Closed-Loop Approach"
    )

    # Project Metadata Card
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); color: #F8FAFC; padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 15px;">
            <div>
                <h3 style="margin: 0; color: #60A5FA; font-size: 1.3rem;">MANIPAL INSTITUTE OF TECHNOLOGY, BENGALURU</h3>
                <div style="color: #94A3B8; font-size: 0.9rem;">(A constituent unit of MAHE, Manipal) — School of Computer Engineering</div>
                <div style="color: #E2E8F0; font-weight: 600; margin-top: 0.5rem;">Machine Learning (CSE_3125) — V Semester Mini Project</div>
            </div>
            <div style="text-align: right;">
                <span style="background: rgba(59, 130, 246, 0.2); border: 1px solid #3B82F6; color: #93C5FD; padding: 0.3rem 0.8rem; border-radius: 6px; font-size: 0.8rem; font-weight: 600;">
                    September 2026
                </span>
            </div>
        </div>
        <hr style="border: 0; border-top: 1px solid rgba(255, 255, 255, 0.15); margin: 1rem 0;">
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 15px; font-size: 0.88rem;">
            <div>
                <div style="color: #94A3B8;">Project Authors:</div>
                <div style="color: #FFFFFF; font-weight: 600;">• K Sri Praneetha (245805006)</div>
                <div style="color: #FFFFFF; font-weight: 600;">• K Ruthika Reddy (245805344)</div>
                <div style="color: #CBD5E1; font-size: 0.8rem;">CSE Core - A</div>
            </div>
            <div>
                <div style="color: #94A3B8;">Project Guide / Supervisor:</div>
                <div style="color: #FFFFFF; font-weight: 600;">Shreya Banerjee</div>
                <div style="color: #CBD5E1; font-size: 0.8rem;">School of Computer Engineering, MIT Bengaluru</div>
            </div>
            <div>
                <div style="color: #94A3B8;">Active Production Model:</div>
                <div style="color: #34D399; font-weight: 600;">""" + model_results.best_model_name + """</div>
                <div style="color: #CBD5E1; font-size: 0.8rem;">Out-of-sample Test MAE: """ + f"{model_results.train_metadata['best_model_mae']:.2f}" + """</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Abstract & Objectives", "Methodology & Disaggregation", "Closed-Loop Architecture", "Math & Formulas", "Limitations & Stack"
    ])

    with tab1:
        st.markdown("### 1. Abstract")
        st.write(
            "Food service operations such as canteens, hostels, and catering units routinely struggle to match next-day preparation quantities "
            "with actual demand, resulting in either food shortages or significant food waste. This project proposes a **Smart Food Demand "
            "Forecasting and Waste Risk Estimation system** that predicts the number of meals likely to be required the following day, using recent "
            "demand history, meal characteristics, pricing, promotions, location, calendar effects, and next-day weather forecasts."
        )
        st.write(
            "Three machine learning models — **Linear Regression**, **Random Forest**, and **XGBoost** — are trained and compared against a "
            "naive historical-average baseline. The best-performing model is selected using standard regression metrics. "
            "Unlike a one-shot forecasting tool, the system **closes the loop**: actual demand and preparation figures are logged post-service, "
            "allowing the system to track its live 7-day rolling MAE and flag itself for retraining when concept drift or performance degradation occurs."
        )

        st.markdown("### 2. Core Project Objectives")
        st.markdown("""
        * **Day-Level Disaggregation:** Collect weekly food order, pricing, promotion, and center data and disaggregate to realistic daily resolution.
        * **Weather Integration:** Integrate real-time historical and forecasted weather data (temperature, rainfall, conditions) via Open-Meteo API.
        * **Feature Engineering:** Construct causal lag features ($t-1, t-7$), 7-day rolling averages, trends, promotional flags, and calendar indicators without data leakage.
        * **Multi-Model Benchmark:** Train Linear Regression, Random Forest, and XGBoost against a Naive Baseline.
        * **Operational Preparation Recommendation:** Compute safety buffers and translate point forecasts into kitchen batch directives.
        * **Waste & Shortage Risk Estimation:** Quantify surplus spoilage risk and stock-out probabilities with economic costing.
        * **Closed-Loop Feedback:** Log post-service actuals, track live 7-day rolling MAE, and trigger automated retraining alerts upon drift.
        """)

    with tab2:
        st.markdown("### 3. Proposed Methodology & Disaggregation")
        st.write(
            "The system utilizes the publicly available Food Demand Forecasting dataset (Genpact / Analytics Vidhya hosted on Kaggle) "
            "as the baseline. Weekly totals are disaggregated into day-level demand estimates using realistic empirical day-of-week ratios "
            "(higher footfall on Fridays and weekends, promotional elasticity uplifts, and weather modulations) combined with controlled noise."
        )
        st.markdown("""
        ```
        Historical Weekly Orders (Genpact / Kaggle)
                            ↓
        Daily Disaggregation (Calibrated Multipliers + Controlled Noise)
                            ↓
        Weather Fusion (Open-Meteo Live / Historical Climate)
                            ↓
        Chronological Train / Test Split (Strict Time Causality, No Leakage)
                            ↓
        Multi-Model Benchmark (Baseline, Ridge, Random Forest, XGBoost)
                            ↓
        Programmatic Selection (Lowest MAE → Lowest RMSE → Highest R²)
                            ↓
        Next-Day Preparation Recommendation (+ Configurable Buffer %)
                            ↓
        Post-Service Actual Outcome Entry (Feedback Loop)
                            ↓
        Live 7-Day Rolling MAE Drift Monitor → Retraining Alert
        ```
        """)

    with tab3:
        st.markdown("### 4. Closed-Loop Continuous Improvement")
        st.write(
            "Traditional demand forecasting systems generate one-off static predictions without ever verifying whether the prediction was "
            "accurate in real kitchen operations. Our closed-loop architecture introduces a post-service audit layer:"
        )
        st.markdown(r"""
        1. **Evening Forecast:** Predicts demand $\hat{y}_{t+1}$ and suggests buffer-adjusted batch size $P_{t+1}$.
        2. **Kitchen Service:** Kitchen prepares actual quantity $P_{\text{actual}}$ and serves customers.
        3. **Post-Service Logging:** Manager logs actual consumed meals $y_{\text{actual}}$ and actual preparation $P_{\text{actual}}$.
        4. **Automatic Variance Computation:** Computes actual food waste, prediction error, absolute error, and shortage.
        5. **7-Day Rolling Accuracy Tracking:** Tracks running MAE over time.
        6. **Concept Drift & Retraining Flag:** If rolling MAE exceeds the training benchmark by $\ge 25\%$ for 3 consecutive days, the system triggers a **Retraining Recommended** flag.
        """)

    with tab4:
        st.markdown("### 5. Mathematical Formulations")
        st.latex(r"\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|")
        st.latex(r"\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}")
        st.latex(r"R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}")
        st.latex(r"\text{Recommended Preparation} = \left\lceil \hat{y} \times \left(1 + \frac{\text{Buffer \%}}{100}\right) \right\rceil")
        st.latex(r"\text{Actual Food Waste} = \max(0, \text{Actual Prepared} - \text{Actual Demand})")
        st.latex(r"\text{Actual Shortage} = \max(0, \text{Actual Demand} - \text{Actual Prepared})")
        st.latex(r"\text{Group Contribution \%} = \frac{\sum_{f \in \text{Group}} \text{Importance}(f)}{\sum_{\text{all } f} \text{Importance}(f)} \times 100\%")

    with tab5:
        st.markdown("### 6. System Limitations & Honesty In Data")
        st.markdown("""
        * **Simulated Day-Level Granularity:** While based on real Genpact weekly fulfillment data, daily distribution is disaggregated via empirical multipliers.
        * **Weather Forecast Dependency:** Real-world day-ahead accuracy is naturally bounded by the accuracy of the underlying meteorological forecast.
        * **Data Provenance Transparency:** The system explicitly distinguishes between **REAL DATA**, **DEMO BENCHMARK DATA**, and **SIMULATED WEATHER FALLBACKS**.
        """)

        st.markdown("### 7. Technology Stack")
        st.markdown("""
        * **Core Language:** Python 3.13 / 3.10+
        * **Machine Learning:** Scikit-learn, XGBoost, NumPy, SciPy
        * **Data Processing:** Pandas, PyYAML, Joblib
        * **User Interface & Visualizations:** Streamlit, Plotly Express & Graph Objects
        * **Weather Integration:** Open-Meteo API (Free tier, no key required)
        * **Database Persistence:** SQLite 3 with WAL indexing
        * **Testing Suite:** PyTest (100% test pass rate)
        """)
