# Adaptive Food Demand Forecasting & Waste Risk Estimation System: A Closed-Loop Approach

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![ML](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20XGBoost-F7931E.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Academic Project](https://img.shields.io/badge/Academic-MIT%20Bengaluru%20CSE-8B5CF6.svg)](#academic-project-details)

---

## 📖 Project Overview

Food service operations such as university canteens, hostels, hospitals, and catering units routinely struggle to match next-day preparation quantities with actual consumption demand. This operational mismatch leads to two persistent and costly failure modes:
1. **Over-preparation:** Generates avoidable perishable food waste, direct ingredient cost loss, and unnecessary environmental burden.
2. **Under-preparation:** Causes sudden stock-outs, customer dissatisfaction, and lost revenue.

Traditional commercial forecasting tools generate one-off predictions and stop there. They never verify whether their forecasts were accurate in actual kitchen operations.

**This project delivers a complete, production-grade closed-loop forecasting and waste-risk intelligence system.** It integrates demand history, promotional campaigns, calendar seasonality, and day-ahead weather forecasts to predict next-day meal orders, translates predictions into operational preparation targets with configurable safety buffers, estimates surplus/shortage risk, and **closes the loop** by logging post-service actual outcomes to monitor prediction error and automatically flag model retraining when concept drift occurs.

---

## 🏛️ Academic Project Details

* **Institution:** Manipal Institute of Technology (MIT), Bengaluru (A constituent unit of MAHE, Manipal)
* **Department:** School of Computer Engineering
* **Course:** Machine Learning (`CSE_3125`) — V Semester Mini Project
* **Academic Term:** September 2026
* **Project Team:**
  * **K Sri Praneetha** (Registration No: `245805006`)
  * **K Ruthika Reddy** (Registration No: `245805344`)
  * *Section:* CSE Core - A
* **Project Supervisor / Guide:** **Shreya Banerjee**, School of Computer Engineering, MIT Bengaluru

---

## 🔄 The Closed-Loop System Architecture

```
Historical Order Data (Genpact / Kaggle)
               ↓
Daily Disaggregation (Calibrated Multipliers + Controlled Noise)
               ↓
Weather Fusion (Open-Meteo Live API / Historical Climate)
               ↓
Feature Engineering (Lags, 7D Moving Avg, Calendar, Pricing)
               ↓
Chronological Train / Test Split (Strict Time Causality, No Leakage)
               ↓
Multi-Model Benchmark (Baseline, Ridge, Random Forest, XGBoost)
               ↓
Programmatic Model Selection (Lowest MAE → Lowest RMSE → Highest R²)
               ↓
Next-Day Forecasting & Preparation Recommendation (+ Configurable Buffer %)
               ↓
Waste & Shortage Risk Estimation (Low / Medium / High Tiers + Cost Impact)
               ↓
Post-Service Actual Outcome Entry (Feedback Loop)
               ↓
Live 7-Day Rolling MAE Tracking & Concept Drift Engine
               ↓
Automated Retraining Flag & 1-Click Pipeline Update
```

---

## ✨ Key Features

1. **Multi-Model Regression Suite:**
   * **Naive Baseline:** Last-week-same-day historical baseline ($t-7$).
   * **Linear Regression (Ridge):** Scaled, regularized linear baseline with coefficient interpretability.
   * **Random Forest Regressor:** Bagging ensemble capturing complex non-linear pricing/weather interactions.
   * **XGBoost Regressor:** Gradient-boosted decision tree ensemble optimized for predictive accuracy.

2. **Programmatic Best-Model Selection:**
   * Models are ranked on the out-of-sample chronological test set.
   * Winner is selected deterministically: Primary (Lowest MAE) $\rightarrow$ Secondary (Lowest RMSE) $\rightarrow$ Tertiary (Highest $R^2$).

3. **Weather Integration (Open-Meteo API):**
   * Real-time queries for temperature, rainfall, rain probability, and WMO weather condition codes for Indian cities (Bengaluru, Manipal, Mumbai, Delhi, Hyderabad).
   * Robust, deterministic offline fallback labeled honestly as simulated weather.

4. **Actionable Operational Preparation Buffer:**
   * Formula: $\text{Recommended Preparation} = \lceil \hat{y} \times (1 + \text{Buffer \%} / 100) \rceil$.
   * Configurable slider ($0\% - 25\%$) with transparent operational reasoning.

5. **Waste & Shortage Risk Assessment:**
   * Evaluates planned batch vs forecast. Categorizes into **LOW**, **MEDIUM**, and **HIGH** waste risk, alongside **NONE**, **MODERATE**, and **CRITICAL** shortage risk.
   * Quantifies expected raw material financial exposure in INR (₹).

6. **Closed-Loop Feedback & SQLite Persistence:**
   * Kitchen staff log post-service actual prepared and consumed meals.
   * Automatic calculation of **Actual Food Waste**, **Shortage**, **Prediction Error**, and **Absolute Error**.
   * Persists all predictions, outcomes, and retraining audits in a local SQLite database (`data/feedback_history.db`).

7. **Concept Drift & Retraining Flag:**
   * Computes **7-day Rolling MAE**.
   * If live rolling MAE exceeds the training benchmark by $\ge 25\%$ for 3+ consecutive records, the system triggers a `⚠ Retraining Recommended` status.
   * One-click retraining re-ingests fresh data, re-tunes models, updates artifacts in `models/`, and resets the baseline.

8. **Interpretability & Feature Importance:**
   * Tree-based Gini/Gain importance and Linear coefficient analysis.
   * Grouped factor contributions across **Demand History**, **Weather Conditions**, **Pricing & Promotions**, **Calendar & Seasonality**, and **Meal/Center Attributes**.

---

## 📊 Mathematical Formulations

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

$$\text{Actual Food Waste} = \max(0, \text{Actual Prepared} - \text{Actual Demand})$$

$$\text{Actual Shortage} = \max(0, \text{Actual Demand} - \text{Actual Prepared})$$

$$\text{Group Contribution \%} = \frac{\sum_{f \in \text{Group}} \text{Importance}(f)}{\sum_{\text{all } f} \text{Importance}(f)} \times 100\%$$

---

## 📁 Project Directory Structure

```
PROJECT/
├── app.py                      # Main Streamlit SaaS application controller
├── config.yaml                 # System configurations, hyperparameters, thresholds
├── requirements.txt            # Python package dependencies
├── README.md                   # Comprehensive system documentation
├── .gitignore                  # Git tracking exclusion rules
├── .env.example                # Environment variables template
│
├── src/                        # Core Python source modules
│   ├── __init__.py
│   ├── config.py               # YAML config loader and path resolver
│   ├── database.py             # SQLite persistence engine
│   ├── data_loader.py          # Data ingestion and schema validation
│   ├── disaggregation.py       # Weekly-to-daily disaggregation engine
│   ├── demo_generator.py       # Realistic benchmark demo dataset synthesizer
│   ├── preprocessing.py        # Data sanitization and cleaning
│   ├── feature_engineering.py  # Lag features, rolling stats, chronological split
│   ├── weather_service.py      # Open-Meteo API integration & offline fallback
│   ├── model_training.py       # Multi-model training and serialization
│   ├── model_evaluation.py     # Metrics calculation and selection rules
│   ├── prediction.py           # Day-ahead inference engine
│   ├── waste_risk.py           # Buffer planning and risk calculation
│   ├── feedback_loop.py        # Outcome logging and rolling MAE tracker
│   ├── retraining.py           # Concept drift monitoring and retraining
│   ├── feature_importance.py   # Model interpretability and factor contributions
│   └── ui/                     # UI components and page renderers
│       ├── styles.py           # SaaS design system CSS and KPI cards
│       ├── page_overview.py    # Page 1: Executive Overview
│       ├── page_forecast.py    # Page 2: Next-Day Forecast
│       ├── page_waste_risk.py  # Page 3: Waste & Shortage Risk
│       ├── page_feedback.py    # Page 4: Closed-Loop Feedback
│       ├── page_models.py      # Page 5: Model Benchmarks
│       ├── page_importance.py  # Page 6: Feature Importance
│       ├── page_explorer.py    # Page 7: Data Explorer
│       ├── page_health.py      # Page 8: System Health & Retraining
│       └── page_about.py       # Page 9: Academic Synopsis & Methodology
│
├── data/                       # Data storage directory
│   ├── raw/                    # Directory for user-provided Kaggle CSVs
│   ├── demo/                   # Pre-generated benchmark demo datasets
│   ├── processed/              # Feature-engineered cache
│   └── feedback_history.db     # SQLite database for predictions and feedback
│
├── models/                     # Serialized model artifacts (.joblib, metadata.json)
└── tests/                      # PyTest automated test suite
    ├── test_data_loader.py
    ├── test_feature_engineering.py
    ├── test_models.py
    ├── test_waste_risk.py
    ├── test_feedback_loop.py
    └── test_weather_service.py
```

---

## 🚀 Installation and Setup

### 1. Prerequisites
* Python 3.10, 3.11, 3.12, or 3.13 installed.
* Git installed.

### 2. Clone the Repository
```bash
git clone <repo-url>
cd PROJECT
```

### 3. Create and Activate a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🧪 Running the Test Suite

Execute the comprehensive automated test suite to verify data loading, feature engineering, model training, risk logic, and database persistence:

```bash
pytest -v
```

*All 17 test modules run with 100% pass rate.*

---

## 🖥️ Launching the Application

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

The application will launch in your default web browser at `http://localhost:8501`.

---

## 📂 Using Real Kaggle Food Demand Dataset

The system includes a pre-packaged, validated benchmark demo dataset so it is immediately runnable out of the box.

To use the official **Genpact / Analytics Vidhya Food Demand Forecasting Dataset** from Kaggle:
1. Download the dataset from Kaggle (`food-demand-forecasting`).
2. Place the following three CSV files into the `data/raw/` directory:
   * `data/raw/train.csv`
   * `data/raw/fulfilment_center_info.csv`
   * `data/raw/meal_info.csv`
3. Open the Streamlit application. The sidebar will automatically detect the presence of the real dataset and enable the `Real Kaggle Dataset (data/raw)` mode.

---

## ⚠️ Honesty in Data & Limitations

1. **Simulated Day-Level Granularity:** The base Genpact Kaggle dataset is provided at weekly granularity. In accordance with the project synopsis, daily demand is disaggregated using realistic, literature-validated day-of-week multipliers combined with controlled noise.
2. **Weather Forecast Upper Bound:** Day-ahead predictive accuracy is naturally bounded by the accuracy of the underlying meteorological weather forecast.
3. **No Fabricated Data:** The dashboard clearly marks data provenance (`REAL DATA`, `DEMO DATA`, and `SIMULATED WEATHER FALLBACK`).

---

## 👥 Authors & Supervision

Developed at **Manipal Institute of Technology, Bengaluru** (MAHE) for the V Semester Machine Learning Mini Project (`CSE_3125`):
* **K Sri Praneetha** (`245805006`)
* **K Ruthika Reddy** (`245805344`)
* **Project Guide:** **Shreya Banerjee** (School of Computer Engineering, MIT Bengaluru)
