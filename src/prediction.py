"""
Inference and Next-Day Food Demand Prediction Engine.
Constructs full feature vector for arbitrary user queries, merges recent time-series lags,
encodes metadata, and computes point estimates using trained model pipelines.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from src.demo_generator import CENTERS_DATA, MEALS_DATA
from src.feature_engineering import assign_season, FIXED_HOLIDAYS
from src.weather_service import WeatherData, fetch_open_meteo_forecast


@dataclass
class SingleForecastResult:
    target_date: str
    center_id: int
    center_city: str
    meal_id: int
    meal_name: str
    category: str
    cuisine: str
    predicted_demand: int
    raw_model_prediction: float
    model_name: str
    weather: WeatherData
    historical_lag_1: float
    historical_lag_7: float
    historical_rolling_7: float
    checkout_price: float
    discount_pct: float
    is_promo: bool
    feature_row: Dict[str, Any]


def get_center_info_dict(center_id: int) -> Dict[str, Any]:
    """Retrieve metadata for given center_id."""
    for c in CENTERS_DATA:
        if c["center_id"] == center_id:
            return c
    return {"center_id": center_id, "city_code": 500, "region_code": 50, "center_type": "TYPE_A", "op_area": 4.0, "city": f"Center-{center_id}"}


def get_meal_info_dict(meal_id: int) -> Dict[str, Any]:
    """Retrieve metadata for given meal_id."""
    for m in MEALS_DATA:
        if m["meal_id"] == meal_id:
            return m
    return {"meal_id": meal_id, "category": "Rice Bowl", "cuisine": "Indian", "base_price": 250.0, "base_demand": 300}


def predict_single_item(
    model: Any,
    model_name: str,
    feature_names: list,
    categorical_encoders: Dict[str, Dict[str, int]],
    center_id: int,
    meal_id: int,
    target_date_str: str,
    checkout_price: Optional[float] = None,
    emailer_promotion: int = 0,
    homepage_featured: int = 0,
    weather_override: Optional[WeatherData] = None,
    df_history: Optional[pd.DataFrame] = None
) -> SingleForecastResult:
    """
    Perform next-day demand prediction for a specific (center, meal, date) tuple.
    """
    target_dt = pd.to_datetime(target_date_str)
    center_meta = get_center_info_dict(center_id)
    meal_meta = get_meal_info_dict(meal_id)

    base_price = float(meal_meta.get("base_price", 250.0))
    if checkout_price is None or checkout_price <= 0:
        checkout_price = base_price

    # Weather: Use override or fetch forecast
    if weather_override is not None:
        weather = weather_override
    else:
        weather = fetch_open_meteo_forecast(center_id, target_date_str)

    # Historical Lags: Look up from history DataFrame if available, else derive sensible defaults
    lag_1 = float(meal_meta.get("base_demand", 250))
    lag_7 = float(meal_meta.get("base_demand", 250))
    rolling_7_mean = float(meal_meta.get("base_demand", 250))
    rolling_7_std = 15.0

    if df_history is not None and not df_history.empty:
        series_mask = (df_history["center_id"] == center_id) & (df_history["meal_id"] == meal_id)
        series_df = df_history[series_mask].sort_values(by="date")
        if not series_df.empty:
            recent_orders = series_df["num_orders"].dropna()
            if len(recent_orders) >= 1:
                lag_1 = float(recent_orders.iloc[-1])
            if len(recent_orders) >= 7:
                lag_7 = float(recent_orders.iloc[-7])
                rolling_7_mean = float(recent_orders.iloc[-7:].mean())
                rolling_7_std = float(recent_orders.iloc[-7:].std() or 10.0)
            else:
                rolling_7_mean = float(recent_orders.mean())
                rolling_7_std = float(recent_orders.std() or 10.0)

    trend_7 = round((lag_1 - lag_7) / 7.0, 2)

    # Calendar features
    day_of_week = target_dt.weekday()
    is_weekend = 1 if day_of_week in (5, 6) else 0
    is_holiday = 1 if (target_dt.month, target_dt.day) in FIXED_HOLIDAYS else 0
    month = target_dt.month
    day_of_month = target_dt.day
    season_str = assign_season(month)

    # Pricing & promo features
    discount_amt = max(0.0, base_price - checkout_price)
    discount_ratio = (discount_amt / base_price) if base_price > 0 else 0.0
    promo_interaction = emailer_promotion * homepage_featured

    # Weather numerical & boolean
    is_heavy_rain = 1 if weather.rainfall_mm > 10.0 else 0
    is_extreme = 1 if (weather.rainfall_mm > 30.0 or weather.temperature_c > 38.0 or weather.temperature_c < 10.0) else 0

    # Categorical encodings
    def enc(col_name: str, raw_val: str) -> int:
        mapping = categorical_encoders.get(col_name, {})
        return mapping.get(str(raw_val), 0)

    category_enc = enc("category", meal_meta["category"])
    cuisine_enc = enc("cuisine", meal_meta["cuisine"])
    center_type_enc = enc("center_type", center_meta["center_type"])
    weather_cond_enc = enc("weather_condition", weather.weather_condition)
    season_enc = enc("season", season_str)

    raw_feature_dict = {
        "checkout_price": float(checkout_price),
        "base_price": float(base_price),
        "discount_amount": float(discount_amt),
        "discount_ratio": float(discount_ratio),
        "emailer_for_promotion": int(emailer_promotion),
        "homepage_featured": int(homepage_featured),
        "promo_interaction": int(promo_interaction),
        "op_area": float(center_meta["op_area"]),
        "temperature_c": float(weather.temperature_c),
        "rainfall_mm": float(weather.rainfall_mm),
        "rain_probability_pct": float(weather.rain_probability_pct),
        "is_heavy_rain": int(is_heavy_rain),
        "is_extreme_weather": int(is_extreme),
        "day_of_week": int(day_of_week),
        "is_weekend": int(is_weekend),
        "is_holiday": int(is_holiday),
        "month": int(month),
        "day_of_month": int(day_of_month),
        "demand_lag_1": float(lag_1),
        "demand_lag_7": float(lag_7),
        "demand_rolling_7_mean": float(rolling_7_mean),
        "demand_rolling_7_std": float(rolling_7_std),
        "demand_trend_7": float(trend_7),
        "category_encoded": int(category_enc),
        "cuisine_encoded": int(cuisine_enc),
        "center_type_encoded": int(center_type_enc),
        "weather_condition_encoded": int(weather_cond_enc),
        "season_encoded": int(season_enc)
    }

    # Construct dataframe with exact column order expected by model
    X_single = pd.DataFrame([{col: raw_feature_dict.get(col, 0.0) for col in feature_names}])

    # Model inference
    raw_pred = float(model.predict(X_single)[0])
    point_pred = int(max(0, round(raw_pred)))

    return SingleForecastResult(
        target_date=target_date_str,
        center_id=center_id,
        center_city=center_meta["city"],
        meal_id=meal_id,
        meal_name=f"{meal_meta['category']} ({meal_meta['cuisine']})",
        category=meal_meta["category"],
        cuisine=meal_meta["cuisine"],
        predicted_demand=point_pred,
        raw_model_prediction=round(raw_pred, 2),
        model_name=model_name,
        weather=weather,
        historical_lag_1=lag_1,
        historical_lag_7=lag_7,
        historical_rolling_7=rolling_7_mean,
        checkout_price=checkout_price,
        discount_pct=round(discount_ratio * 100.0, 1),
        is_promo=bool(emailer_promotion or homepage_featured),
        feature_row=raw_feature_dict
    )
