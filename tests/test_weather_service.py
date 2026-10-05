"""
Unit tests for weather service and deterministic fallback behavior.
"""

from src.weather_service import fetch_open_meteo_forecast, generate_fallback_weather, get_center_metadata


def test_center_metadata():
    meta = get_center_metadata(10)
    assert meta["city"] == "Bengaluru"
    assert "lat" in meta
    assert "lon" in meta


def test_fallback_weather_generation():
    w = generate_fallback_weather(10, "2026-07-15", "Bengaluru")
    assert w.center_id == 10
    assert w.city == "Bengaluru"
    assert w.temperature_c > 0
    assert w.rainfall_mm >= 0
    assert "Fallback" in w.data_source or "Demo" in w.data_source


def test_fetch_open_meteo_returns_weather_data():
    # Will either fetch live forecast or cleanly fallback without throwing exception
    w = fetch_open_meteo_forecast(10, "2026-10-06", timeout=3)
    assert w.center_id == 10
    assert w.weather_condition in ["Clear", "Cloudy", "Rainy", "Stormy", "Extreme Heat", "Snowy", "Foggy"]
    assert w.data_source in ["Live Open-Meteo API", "Simulated Demo Weather (Offline Fallback)"]
