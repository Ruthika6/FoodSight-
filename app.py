"""
Adaptive Food Demand Forecasting and Waste Risk Estimation System:
A Closed-Loop Approach

Main Streamlit Application Controller & SaaS Dashboard Entry Point.
Manipal Institute of Technology, Bengaluru (V Semester Mini Project - CSE_3125)
Authors: K Sri Praneetha (245805006), K Ruthika Reddy (245805344)
Supervisor: Shreya Banerjee
"""

from pathlib import Path
import streamlit as st

# Set Streamlit Page Configuration must be first Streamlit command
st.set_page_config(
    page_title="Food Demand & Waste Risk Intelligence",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.config import CONFIG
from src.database import init_db
from src.data_loader import check_raw_dataset_exists, load_raw_or_demo_data
from src.feedback_loop import seed_initial_feedback_history
from src.feature_engineering import prepare_ml_splits
from src.model_training import train_all_models
from src.retraining import evaluate_retraining_trigger
from src.ui.page_about import render_about_page
from src.ui.page_explorer import render_explorer_page
from src.ui.page_feedback import render_feedback_page
from src.ui.page_forecast import render_forecast_page
from src.ui.page_health import render_health_page
from src.ui.page_importance import render_importance_page
from src.ui.page_models import render_models_page
from src.ui.page_overview import render_overview_page
from src.ui.page_waste_risk import render_waste_risk_page
from src.ui.styles import apply_custom_styles


# Initialize SQLite Database and seed baseline feedback on first launch
init_db()
seed_initial_feedback_history(num_days=21, force_reseed=False)

# Apply Modern SaaS Styling
apply_custom_styles()


@st.cache_data(show_spinner="Loading and validating food demand dataset...")
def get_cached_dataset(force_demo: bool):
    """Load dataset with Streamlit caching."""
    return load_raw_or_demo_data(force_demo=force_demo)


@st.cache_resource(show_spinner="Training machine learning model suite (Baseline, Ridge, Random Forest, XGBoost)...")
def get_cached_models(df_raw):
    """Train ML models with Streamlit resource caching."""
    splits = prepare_ml_splits(df_raw, split_ratio=CONFIG.data.train_test_split_ratio)
    results = train_all_models(splits, save_artifacts=True)
    return results


def main():
    # Sidebar Navigation & System Controls
    st.sidebar.markdown("""
    <div class="sidebar-header-box">
        <h2 class="sidebar-title">🍽️ Demand Intel</h2>
        <div class="sidebar-subtitle">Closed-Loop Forecasting System</div>
    </div>
    <hr style="border: 0; border-top: 1px solid rgba(255, 255, 255, 0.08); margin: 0.5rem 0 1rem 0;">
    """, unsafe_allow_html=True)

    # 1. Dataset Mode Selection in Sidebar
    has_real_data = check_raw_dataset_exists()
    dataset_mode_options = ["Demo Benchmark Dataset (Validated)"]
    if has_real_data:
        dataset_mode_options.append("Real Kaggle Dataset (data/raw)")

    selected_mode = st.sidebar.radio(
        "📁 Dataset Source Mode:",
        dataset_mode_options,
        index=0,
        help="Switch between pre-packaged validated benchmark data and custom raw Kaggle datasets."
    )
    force_demo = ("Demo" in selected_mode)

    # Load Data & Train Models
    df_daily, metadata = get_cached_dataset(force_demo=force_demo)
    model_results = get_cached_models(df_daily)

    # Navigation Menu
    st.sidebar.markdown("### 📌 Navigation")
    pages = {
        "Executive Overview": "📊 Overview",
        "Next-Day Forecast": "🔮 Next-Day Forecast",
        "Waste & Shortage Risk": "⚠️ Waste & Shortage Risk",
        "Closed-Loop Feedback": "🔄 Feedback Loop",
        "Model Performance": "📈 Model Benchmarks",
        "Feature Importance": "🧠 Feature Importance",
        "Data Explorer": "🔍 Data Explorer",
        "System Health & Retrain": "🛡️ System Health",
        "Academic Synopsis & About": "📖 Synopsis & About"
    }

    choice = st.sidebar.radio(
        "Select Page:",
        list(pages.keys()),
        format_func=lambda x: pages[x],
        label_visibility="collapsed"
    )

    # Sidebar System Status Pill
    best_mae = model_results.train_metadata["best_model_mae"]
    retrain_status = evaluate_retraining_trigger(baseline_benchmark_mae=best_mae)
    status_color = "#10B981" if not retrain_status.is_retraining_recommended else "#EF4444"
    status_text = "System Healthy" if not retrain_status.is_retraining_recommended else "Retrain Recommended"

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    <div class="sidebar-info-box">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.4rem;">
            <span style="font-weight: 700; font-size: 0.75rem; text-transform: uppercase; color: #94A3B8; letter-spacing: 0.05em;">System Telemetry</span>
            <span style="height: 9px; width: 9px; background-color: {status_color}; border-radius: 50%; display: inline-block; box-shadow: 0 0 8px {status_color};"></span>
        </div>
        <div style="font-weight: 700; color: #F8FAFC; font-size: 0.9rem;">{status_text}</div>
        <div style="color: #94A3B8; font-size: 0.8rem; margin-top: 0.35rem;">Active: <strong style="color: #60A5FA;">{model_results.best_model_name.split(' ')[0]}</strong></div>
        <div style="color: #94A3B8; font-size: 0.8rem;">Test MAE: <strong style="color: #34D399;">{best_mae:.2f}</strong></div>
        <div style="color: #94A3B8; font-size: 0.8rem;">Data: <strong style="color: #CBD5E1;">{metadata.source_mode.split(' ')[0]}</strong></div>
    </div>
    """, unsafe_allow_html=True)

    def on_retrain():
        st.cache_resource.clear()

    # Route to Selected Page
    if choice == "Executive Overview":
        render_overview_page(df_daily, model_results, metadata)
    elif choice == "Next-Day Forecast":
        render_forecast_page(df_daily, model_results, metadata)
    elif choice == "Waste & Shortage Risk":
        render_waste_risk_page(df_daily, model_results, metadata)
    elif choice == "Closed-Loop Feedback":
        render_feedback_page(df_daily, model_results, metadata)
    elif choice == "Model Performance":
        render_models_page(df_daily, model_results, metadata)
    elif choice == "Feature Importance":
        render_importance_page(df_daily, model_results, metadata)
    elif choice == "Data Explorer":
        render_explorer_page(df_daily, model_results, metadata)
    elif choice == "System Health & Retrain":
        render_health_page(df_daily, model_results, metadata, on_retrain_callback=on_retrain)
    elif choice == "Academic Synopsis & About":
        render_about_page(df_daily, model_results, metadata)


if __name__ == "__main__":
    main()
