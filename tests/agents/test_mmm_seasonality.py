# Test: MMM Seasonality Controls

import pytest
import numpy as np
from agents.mmm_agent import MMMAgent
from datetime import date, timedelta


def test_holiday_controls_included():
    """Q_006/A_009: Verify MMM includes holiday dummy variables"""
    agent = MMMAgent()
    
    # Train model with date range spanning major holidays
    start_date = date(2024, 1, 1)
    end_date = date(2024, 12, 31)
    
    model = agent.train_model(
        start_date=start_date,
        end_date=end_date,
        channels=["meta", "google"],
        include_seasonality=True
    )
    
    # Verify holiday controls are present in model
    assert "holiday_christmas" in model.controls
    assert "holiday_new_year" in model.controls
    assert "holiday_golden_week" in model.controls  # Japan
    
    # Verify coefficients exist
    assert model.coefficients["holiday_christmas"] is not None
    assert model.coefficients["holiday_new_year"] is not None


def test_holiday_statistical_significance():
    """Q_006/A_009: Verify holiday controls have p < 0.05"""
    agent = MMMAgent()
    
    # Generate synthetic data with holiday effects
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(365)]
    revenue = np.random.normal(100000, 10000, 365)
    
    # Add Christmas spike
    christmas_idx = [i for i, d in enumerate(dates) if d.month == 12 and d.day == 25][0]
    revenue[christmas_idx-3:christmas_idx+3] += 50000
    
    model = agent.train_model_from_data(
        dates=dates,
        revenue=revenue,
        channels={"meta": np.random.normal(10000, 1000, 365)},
        include_seasonality=True
    )
    
    # Verify holiday coefficients are statistically significant
    holiday_p_values = model.get_p_values(controls=["holiday_christmas"])
    assert holiday_p_values["holiday_christmas"] < 0.05


def test_seasonality_controls_comprehensive():
    """Q_006/A_009: Verify full seasonality control set"""
    agent = MMMAgent()
    
    model = agent.train_model(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
        channels=["meta"],
        include_seasonality=True
    )
    
    # Required seasonality controls
    required_controls = [
        "month_sin",  # Cyclical month encoding
        "month_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "holiday_christmas",
        "holiday_new_year",
        "holiday_golden_week",
        "is_weekend"
    ]
    
    for control in required_controls:
        assert control in model.controls, f"Missing seasonality control: {control}"
