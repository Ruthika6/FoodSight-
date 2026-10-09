"""
FoodSight — Smart Food Demand Forecasting and Zero-Waste Operational Intelligence.
Main Streamlit Application Controller & Enterprise Dashboard Entry Point.
"""

from pathlib import Path
import streamlit as st

# Set Streamlit Page Configuration must be first Streamlit command
st.set_page_config(
    page_title="FoodSight — Demand & Waste Intelligence",
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
from src.ui.styles import apply_custom_styles, render_top_navbar


# Initialize SQLite Database and seed baseline feedback on first launch
init_db()
seed_initial_feedback_history(num_days=21, force_reseed=False)

# Apply Modern SaaS Light Styling & Theme
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
    # 1. Dataset Mode Configuration in Sidebar
    has_real_data = check_raw_dataset_exists()
    dataset_mode_options = ["Demo Benchmark Dataset (Validated)"]
    if has_real_data:
        dataset_mode_options.append("Real Kaggle Dataset (data/raw)")

    st.sidebar.markdown('<div class="sidebar-section-heading">DATA SOURCE</div>', unsafe_allow_html=True)
    selected_mode = st.sidebar.selectbox(
        "Data Mode",
        dataset_mode_options,
        index=0,
        label_visibility="collapsed",
        help="Switch between pre-packaged validated benchmark data and custom raw Kaggle datasets."
    )
    force_demo = ("Demo" in selected_mode)

    # Load Data & Train Models
    df_daily, metadata = get_cached_dataset(force_demo=force_demo)
    model_results = get_cached_models(df_daily)

    # 2. Sidebar Operations Navigation Menu
    st.sidebar.markdown('<div class="sidebar-section-heading">OPERATIONS NAVIGATION</div>', unsafe_allow_html=True)
    pages = {
        "1. Overview": "📊 1. Overview",
        "2. Demand Forecast": "📈 2. Demand Forecast",
        "3. Waste & Shortage Risk": "⚠️ 3. Waste & Shortage Risk",
        "4. Model Performance": "📦 4. Model Performance",
        "5. Feature Importance": "🧠 5. Feature Importance",
        "6. Feedback Loop": "🔄 6. Feedback Loop",
        "7. Model Health & Retrain": "🛡️ 7. Model Health & Retrain",
        "8. Data Explorer": "🔍 8. Data Explorer"
    }

    choice = st.sidebar.radio(
        "Operations Navigation",
        list(pages.keys()),
        format_func=lambda x: pages[x],
        label_visibility="collapsed"
    )

    # 3. Model Telemetry & Status
    best_mae = model_results.train_metadata["best_model_mae"]
    retrain_status = evaluate_retraining_trigger(baseline_benchmark_mae=best_mae)
    is_healthy = not retrain_status.is_retraining_recommended
    status_label = "MODEL HEALTHY" if is_healthy else "RETRAIN RECOMMENDED"

    # 4. Top Global App Bar (matching reference layout)
    render_top_navbar(
        model_name=model_results.best_model_name,
        is_healthy=is_healthy,
        status_text=status_label
    )

    def on_retrain():
        st.cache_resource.clear()

    # Route to Selected Page
    if choice == "1. Overview":
        render_overview_page(df_daily, model_results, metadata)
    elif choice == "2. Demand Forecast":
        render_forecast_page(df_daily, model_results, metadata)
    elif choice == "3. Waste & Shortage Risk":
        render_waste_risk_page(df_daily, model_results, metadata)
    elif choice == "4. Model Performance":
        render_models_page(df_daily, model_results, metadata)
    elif choice == "5. Feature Importance":
        render_importance_page(df_daily, model_results, metadata)
    elif choice == "6. Feedback Loop":
        render_feedback_page(df_daily, model_results, metadata)
    elif choice == "7. Model Health & Retrain":
        render_health_page(df_daily, model_results, metadata, on_retrain_callback=on_retrain)
    elif choice == "8. Data Explorer":
        render_explorer_page(df_daily, model_results, metadata)


if __name__ == "__main__":
    main()
