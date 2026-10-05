"""
Demo Dataset Generator.
Generates realistic benchmark data mirroring the Kaggle / Genpact Food Demand Forecasting schema.
Produces:
1. fulfilment_center_info.csv
2. meal_info.csv
3. train.csv (weekly format)
4. daily_demand_with_weather.csv (disaggregated day-level benchmark with historical weather)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from src.config import CONFIG
from src.disaggregation import disaggregate_weekly_dataframe
from src.weather_service import DEFAULT_CENTER_COORDINATES, generate_fallback_weather


# Standard Center Registry
CENTERS_DATA = [
    {"center_id": 10, "city_code": 590, "region_code": 56, "center_type": "TYPE_A", "op_area": 3.8, "city": "Bengaluru"},
    {"center_id": 13, "city_code": 614, "region_code": 56, "center_type": "TYPE_B", "op_area": 4.5, "city": "Manipal"},
    {"center_id": 24, "city_code": 526, "region_code": 85, "center_type": "TYPE_A", "op_area": 5.1, "city": "Mumbai"},
    {"center_id": 55, "city_code": 647, "region_code": 34, "center_type": "TYPE_C", "op_area": 2.9, "city": "Delhi"},
    {"center_id": 86, "city_code": 562, "region_code": 77, "center_type": "TYPE_B", "op_area": 4.0, "city": "Hyderabad"},
]

# Standard Meals Registry
MEALS_DATA = [
    {"meal_id": 1062, "category": "Beverages", "cuisine": "Italian", "base_price": 180.0, "base_demand": 280},
    {"meal_id": 1109, "category": "Rice Bowl", "cuisine": "Indian", "base_price": 285.0, "base_demand": 420},
    {"meal_id": 1207, "category": "Pasta", "cuisine": "Italian", "base_price": 320.0, "base_demand": 190},
    {"meal_id": 1230, "category": "Beverages", "cuisine": "Continental", "base_price": 150.0, "base_demand": 240},
    {"meal_id": 1525, "category": "Starters", "cuisine": "Thai", "base_price": 240.0, "base_demand": 160},
    {"meal_id": 1778, "category": "Sandwich", "cuisine": "Italian", "base_price": 190.0, "base_demand": 310},
    {"meal_id": 1885, "category": "Salad", "cuisine": "Continental", "base_price": 220.0, "base_demand": 175},
    {"meal_id": 1993, "category": "Rice Bowl", "cuisine": "Indian", "base_price": 310.0, "base_demand": 460},
]


def generate_demo_dataset(
    output_dir: Optional[Path] = None,
    num_weeks: int = 52,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate complete demo dataset suite and save to target directory.
    Returns: (df_centers, df_meals, df_weekly_train, df_daily_processed)
    """
    target_dir = output_dir or CONFIG.paths.demo_data_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)

    # 1. Fulfilment Centers
    df_centers = pd.DataFrame(CENTERS_DATA)
    df_centers.to_csv(target_dir / "fulfilment_center_info.csv", index=False)

    # 2. Meal Info
    df_meals = pd.DataFrame(MEALS_DATA)
    df_meals.to_csv(target_dir / "meal_info.csv", index=False)

    # 3. Weekly Orders (train.csv format)
    weekly_rows = []
    record_id = 1000000

    for week in range(1, num_weeks + 1):
        # Seasonal cycle factor (52 weeks)
        season_cycle = 1.0 + 0.15 * np.sin(2 * np.pi * week / 52.0)

        for center in CENTERS_DATA:
            c_id = center["center_id"]
            op_area_mult = center["op_area"] / 4.0

            for meal in MEALS_DATA:
                m_id = meal["meal_id"]
                base_price = meal["base_price"]
                base_dem = meal["base_demand"]

                # Promotion probabilities
                emailer_promo = 1 if rng.uniform() < 0.10 else 0
                homepage_feat = 1 if rng.uniform() < 0.12 else 0

                # Pricing calculation with realistic promotional discount
                discount_pct = 0.0
                if emailer_promo or homepage_feat:
                    discount_pct = rng.uniform(0.10, 0.25)
                else:
                    discount_pct = rng.uniform(-0.03, 0.08)

                checkout_price = round(base_price * (1.0 - discount_pct), 2)

                # Demand calculation with price elasticity (-0.8) and promo uplifts
                price_elasticity = (checkout_price / base_price) ** (-0.85)
                promo_uplift = 1.0 + (0.35 * emailer_promo) + (0.45 * homepage_feat)
                noise_mult = rng.lognormal(mean=0.0, sigma=0.10)

                expected_weekly_orders = int(
                    base_dem * 7 * op_area_mult * season_cycle * price_elasticity * promo_uplift * noise_mult
                )
                expected_weekly_orders = max(20, expected_weekly_orders)

                weekly_rows.append({
                    "id": record_id,
                    "week": week,
                    "center_id": c_id,
                    "meal_id": m_id,
                    "checkout_price": checkout_price,
                    "base_price": base_price,
                    "emailer_for_promotion": emailer_promo,
                    "homepage_featured": homepage_feat,
                    "num_orders": expected_weekly_orders,
                })
                record_id += 1

    df_weekly_train = pd.DataFrame(weekly_rows)
    df_weekly_train.to_csv(target_dir / "train.csv", index=False)

    # 4. Disaggregate to Day-Level
    df_daily = disaggregate_weekly_dataframe(df_weekly_train, start_date="2024-01-01", seed=seed)

    # 5. Merge Metadata & Add Weather Features
    df_daily = df_daily.merge(df_centers[["center_id", "city", "center_type", "op_area", "city_code", "region_code"]], on="center_id", how="left")
    df_daily = df_daily.merge(df_meals[["meal_id", "category", "cuisine"]], on="meal_id", how="left")

    # Generate daily weather features
    weather_records = []
    for _, row in df_daily.iterrows():
        c_id = int(row["center_id"])
        c_city = str(row["city"])
        d_str = str(row["date"])
        w_obj = generate_fallback_weather(c_id, d_str, c_city)
        weather_records.append({
            "temperature_c": w_obj.temperature_c,
            "rainfall_mm": w_obj.rainfall_mm,
            "rain_probability_pct": w_obj.rain_probability_pct,
            "weather_condition": w_obj.weather_condition,
        })

    df_weather = pd.DataFrame(weather_records)
    df_daily = pd.concat([df_daily, df_weather], axis=1)

    # Weather impact on day-level demand: Heavy rain reduces dine-in/canteen footfall by ~15%
    rain_mask = df_daily["rainfall_mm"] > 5.0
    df_daily.loc[rain_mask, "num_orders"] = np.round(
        df_daily.loc[rain_mask, "num_orders"] * (1.0 - (df_daily.loc[rain_mask, "rainfall_mm"] / 100.0).clip(0.05, 0.25))
    ).astype(int)

    # Save processed demo day-level benchmark
    df_daily.to_csv(target_dir / "daily_demand_with_weather.csv", index=False)

    return df_centers, df_meals, df_weekly_train, df_daily
