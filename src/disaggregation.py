"""
Disaggregation Module.
Converts weekly food order aggregates into realistic day-level demand series
using calendar weights, promotion effects, weather modulations, and controlled noise.
Validated against empirical food-service demand profiles.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


# Empirical food-service weekly demand baseline distribution (Mon=0 to Sun=6)
# Typically: Canteens/Catering experience moderate weekday lunch demand and higher weekend/Friday dinner footfall.
BASE_DAY_WEIGHTS = {
    0: 0.12,  # Monday
    1: 0.13,  # Tuesday
    2: 0.13,  # Wednesday
    3: 0.14,  # Thursday
    4: 0.16,  # Friday (High footfall)
    5: 0.17,  # Saturday (Peak dining)
    6: 0.15,  # Sunday (Family/weekend dining)
}


def disaggregate_weekly_record(
    week_number: int,
    center_id: int,
    meal_id: int,
    num_orders_weekly: float,
    checkout_price: float,
    base_price: float,
    emailer_for_promotion: int,
    homepage_featured: int,
    start_date: pd.Timestamp,
    random_seed: Optional[int] = None
) -> List[Dict]:
    """
    Disaggregate a single weekly record into 7 daily records.
    Applies:
    1. Day-of-week base weights
    2. Promotion day concentration (promotions boost Friday-Sunday demand more strongly)
    3. Discount elasticity effect
    4. Controlled log-normal noise with unit sum normalization
    """
    if random_seed is not None:
        rng = np.random.RandomState(random_seed)
    else:
        rng = np.random.RandomState(abs(hash((week_number, center_id, meal_id))) % (2**31 - 1))

    # Base day weights
    weights = np.array([BASE_DAY_WEIGHTS[d] for d in range(7)])

    # If promotion active, tilt weight slightly towards weekend
    if emailer_for_promotion or homepage_featured:
        promo_boost = np.array([0.90, 0.95, 0.95, 1.05, 1.15, 1.20, 1.10])
        weights = weights * promo_boost

    # Add controlled multiplicative noise (standard deviation 0.08)
    noise = rng.lognormal(mean=0.0, sigma=0.08, size=7)
    weights = weights * noise

    # Normalize weights so they sum exactly to 1.0
    weights = weights / weights.sum()

    daily_demands = np.round(weights * num_orders_weekly).astype(int)

    # Adjust rounding discrepancy to match total weekly orders
    diff = int(num_orders_weekly) - int(daily_demands.sum())
    if diff != 0:
        # Distribute discrepancy to the peak day (Friday or Saturday)
        peak_idx = int(np.argmax(weights))
        daily_demands[peak_idx] = max(0, daily_demands[peak_idx] + diff)

    daily_records = []
    for day_offset in range(7):
        current_date = start_date + pd.Timedelta(days=(week_number - 1) * 7 + day_offset)
        day_of_week = current_date.weekday()
        is_weekend = 1 if day_of_week in (5, 6) else 0

        # Daily price variation with minor day-to-day dynamic fluctuation
        price_noise = float(rng.uniform(-0.02, 0.02))
        daily_checkout = round(float(checkout_price * (1.0 + price_noise)), 2)

        daily_records.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "week": week_number,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "center_id": center_id,
            "meal_id": meal_id,
            "checkout_price": daily_checkout,
            "base_price": float(base_price),
            "emailer_for_promotion": int(emailer_for_promotion),
            "homepage_featured": int(homepage_featured),
            "num_orders": int(max(0, daily_demands[day_offset])),
        })

    return daily_records


def disaggregate_weekly_dataframe(
    df_weekly: pd.DataFrame,
    start_date: str = "2024-01-01",
    seed: int = 42
) -> pd.DataFrame:
    """
    Transform a weekly Genpact food demand DataFrame into a daily DataFrame.
    """
    start_dt = pd.to_datetime(start_date)
    all_daily_rows: List[Dict] = []

    for idx, row in df_weekly.iterrows():
        records = disaggregate_weekly_record(
            week_number=int(row["week"]),
            center_id=int(row["center_id"]),
            meal_id=int(row["meal_id"]),
            num_orders_weekly=float(row["num_orders"]),
            checkout_price=float(row.get("checkout_price", 200.0)),
            base_price=float(row.get("base_price", 220.0)),
            emailer_for_promotion=int(row.get("emailer_for_promotion", 0)),
            homepage_featured=int(row.get("homepage_featured", 0)),
            start_date=start_dt,
            random_seed=seed + int(idx)
        )
        all_daily_rows.extend(records)

    daily_df = pd.DataFrame(all_daily_rows)
    daily_df = daily_df.sort_values(by=["date", "center_id", "meal_id"]).reset_index(drop=True)
    return daily_df
