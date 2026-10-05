"""
Unit tests for preparation recommendations and waste/shortage risk calculations.
"""

import pytest
from src.waste_risk import assess_waste_and_shortage_risk, calculate_preparation_recommendation


def test_preparation_recommendation_buffer():
    # 100 predicted demand + 5% buffer = 105
    rec = calculate_preparation_recommendation(predicted_demand=100, buffer_pct=5.0)
    assert rec.predicted_demand == 100
    assert rec.recommended_preparation == 105
    assert rec.buffer_units == 5
    assert "safety buffer of 5.0%" in rec.operational_reason

    # Zero buffer test
    rec_zero = calculate_preparation_recommendation(predicted_demand=250, buffer_pct=0.0)
    assert rec_zero.recommended_preparation == 250
    assert rec_zero.buffer_units == 0


def test_waste_risk_levels():
    # Low waste: 105 planned vs 100 predicted (5% surplus)
    risk_low = assess_waste_and_shortage_risk(planned_preparation=105, predicted_demand=100)
    assert risk_low.waste_risk_level == "LOW"
    assert risk_low.expected_surplus == 5
    assert risk_low.expected_shortage == 0

    # Medium waste: 112 planned vs 100 predicted (12% surplus)
    risk_med = assess_waste_and_shortage_risk(planned_preparation=112, predicted_demand=100)
    assert risk_med.waste_risk_level == "MEDIUM"
    assert risk_med.expected_surplus == 12

    # High waste: 130 planned vs 100 predicted (30% surplus)
    risk_high = assess_waste_and_shortage_risk(planned_preparation=130, predicted_demand=100)
    assert risk_high.waste_risk_level == "HIGH"
    assert risk_high.expected_surplus == 30


def test_shortage_risk_levels():
    # Moderate shortage: 95 planned vs 100 predicted (5% shortage)
    shortage_mod = assess_waste_and_shortage_risk(planned_preparation=95, predicted_demand=100)
    assert shortage_mod.shortage_risk_level == "MODERATE"
    assert shortage_mod.expected_shortage == 5
    assert shortage_mod.expected_surplus == 0

    # Critical shortage: 80 planned vs 100 predicted (20% shortage)
    shortage_crit = assess_waste_and_shortage_risk(planned_preparation=80, predicted_demand=100)
    assert shortage_crit.shortage_risk_level == "CRITICAL"
    assert shortage_crit.expected_shortage == 20
