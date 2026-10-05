"""
Waste Risk and Operational Preparation Recommendation Module.
Calculates actionable preparation quantities, buffer policies,
and transparent surplus/shortage risk metrics.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
from src.config import CONFIG


@dataclass
class PreparationRecommendation:
    predicted_demand: int
    recommended_preparation: int
    buffer_percentage: float
    buffer_units: int
    operational_reason: str


@dataclass
class WasteRiskAssessment:
    planned_preparation: int
    predicted_demand: int
    expected_surplus: int
    expected_shortage: int
    surplus_percentage: float
    shortage_percentage: float
    waste_risk_level: str  # "LOW", "MEDIUM", "HIGH"
    shortage_risk_level: str  # "NONE", "MODERATE", "CRITICAL"
    status_badge_color: str
    actionable_recommendation: str
    economic_impact_summary: str


def calculate_preparation_recommendation(
    predicted_demand: float,
    buffer_pct: Optional[float] = None,
) -> PreparationRecommendation:
    """
    Compute operational food preparation recommendation.
    Formula: Recommended Prep = ceil(Predicted Demand * (1 + Buffer% / 100))
    """
    if buffer_pct is None:
        buffer_pct = CONFIG.waste_risk.default_buffer_percentage

    buffer_pct = max(0.0, float(buffer_pct))
    base_demand = int(max(0, round(predicted_demand)))
    buffer_units = int(np.ceil(base_demand * (buffer_pct / 100.0)))
    rec_prep = base_demand + buffer_units

    reason = (
        f"Base ML Demand Forecast: {base_demand} meals. "
        f"Applied operational safety buffer of {buffer_pct:.1f}% (+{buffer_units} meals) "
        f"to prevent service stock-outs while minimizing perishable waste."
    )

    return PreparationRecommendation(
        predicted_demand=base_demand,
        recommended_preparation=rec_prep,
        buffer_percentage=buffer_pct,
        buffer_units=buffer_units,
        operational_reason=reason
    )


def assess_waste_and_shortage_risk(
    planned_preparation: int,
    predicted_demand: float,
    low_waste_thresh: Optional[float] = None,
    med_waste_thresh: Optional[float] = None,
    crit_shortage_thresh: Optional[float] = None,
    unit_cost_inr: float = 85.0
) -> WasteRiskAssessment:
    """
    Evaluate waste (over-preparation) and shortage (under-preparation) risk
    when comparing planned kitchen batch against machine learning forecast.
    """
    low_thresh = low_waste_thresh or CONFIG.waste_risk.low_waste_pct
    med_thresh = med_waste_thresh or CONFIG.waste_risk.medium_waste_pct
    crit_thresh = crit_shortage_thresh or CONFIG.waste_risk.shortage_critical_pct

    base_demand = max(1, int(round(predicted_demand)))
    prep = max(0, int(planned_preparation))

    if prep >= base_demand:
        expected_surplus = prep - base_demand
        expected_shortage = 0
        surplus_pct = round((expected_surplus / base_demand) * 100.0, 1)
        shortage_pct = 0.0

        if surplus_pct <= low_thresh:
            waste_risk = "LOW"
            color = "#10B981"  # Green
            advice = (
                f"Planned preparation ({prep}) is well-calibrated with expected demand ({base_demand}). "
                f"Minimal surplus risk ({surplus_pct}%)."
            )
        elif surplus_pct <= med_thresh:
            waste_risk = "MEDIUM"
            color = "#F59E0B"  # Amber
            advice = (
                f"Surplus buffer is elevated ({expected_surplus} excess meals, +{surplus_pct}%). "
                f"Consider staggering batch cooking to prevent end-of-service discard."
            )
        else:
            waste_risk = "HIGH"
            color = "#EF4444"  # Red
            advice = (
                f"High over-preparation alert! Planned quantity exceeds forecast by {expected_surplus} meals (+{surplus_pct}%). "
                f"Immediate action recommended: Reduce batch size to avoid significant food wastage."
            )
        shortage_risk = "NONE"

    else:
        expected_surplus = 0
        expected_shortage = base_demand - prep
        surplus_pct = 0.0
        shortage_pct = round((expected_shortage / base_demand) * 100.0, 1)
        waste_risk = "LOW"

        if shortage_pct <= crit_thresh:
            shortage_risk = "MODERATE"
            color = "#F59E0B"  # Amber
            advice = (
                f"Planned preparation is slightly under forecasted demand by {expected_shortage} meals ({shortage_pct}% deficit). "
                f"Monitor footfall closely or prepare supplementary quick-cook options."
            )
        else:
            shortage_risk = "CRITICAL"
            color = "#EF4444"  # Red
            advice = (
                f"Severe stock-out risk! Preparation plan ({prep}) is {expected_shortage} meals below predicted demand ({base_demand}). "
                f"High probability of customer refusal and unmet demand."
            )

    potential_loss = expected_surplus * unit_cost_inr
    economic_summary = (
        f"Estimated potential raw material waste cost: ₹{potential_loss:,.2f} "
        f"(assuming baseline meal prep cost ₹{unit_cost_inr:.0f}/plate)."
    )

    return WasteRiskAssessment(
        planned_preparation=prep,
        predicted_demand=base_demand,
        expected_surplus=expected_surplus,
        expected_shortage=expected_shortage,
        surplus_percentage=surplus_pct,
        shortage_percentage=shortage_pct,
        waste_risk_level=waste_risk,
        shortage_risk_level=shortage_risk,
        status_badge_color=color,
        actionable_recommendation=advice,
        economic_impact_summary=economic_summary
    )
