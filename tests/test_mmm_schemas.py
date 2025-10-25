"""
Tests for Marketing Mix Modeling Schemas — Validates Bayesian attribution contracts.
"""

import pytest
from datetime import date
from contracts.mmm_schemas import (
    ROICurve,
    MMMEstimate,
    AllocationPlan,
    SaturationCurve,
    AdstockTransform
)


class TestROICurveSchema:
    """Test ROICurve monotonicity validation."""
    
    def test_roi_curve_monotonic_validation(self):
        """Valid monotonic ROI curve passes validation."""
        curve = ROICurve(
            spend=[100000, 200000, 300000, 400000],
            revenue_incremental=[220000, 360000, 460000, 540000]
        )
        
        assert len(curve.spend) == 4
        assert curve.spend[0] < curve.spend[1] < curve.spend[2] < curve.spend[3]
    
    def test_negative_roi_curve_rejects(self):
        """Decreasing spend values are rejected."""
        with pytest.raises(ValueError, match="monotonically increasing"):
            ROICurve(
                spend=[100000, 200000, 150000, 400000],  # INVALID: 150k < 200k
                revenue_incremental=[220000, 360000, 280000, 540000]
            )
    
    def test_roi_curve_equal_values_rejects(self):
        """Equal consecutive spend values are rejected."""
        with pytest.raises(ValueError, match="monotonically increasing"):
            ROICurve(
                spend=[100000, 200000, 200000, 400000],  # INVALID: duplicate
                revenue_incremental=[220000, 360000, 360000, 540000]
            )
    
    def test_roi_curve_interpolation(self):
        """Interpolation works correctly for intermediate values."""
        curve = ROICurve(
            spend=[0, 100000, 200000],
            revenue_incremental=[0, 200000, 350000]
        )
        
        # Linear interpolation
        mid_revenue = curve.interpolate(50000)
        assert 90000 <= mid_revenue <= 110000  # Approximate check
    
    def test_roi_curve_length_mismatch_rejects(self):
        """Mismatched spend/revenue array lengths are rejected."""
        with pytest.raises(ValueError, match="same length"):
            ROICurve(
                spend=[100000, 200000, 300000],
                revenue_incremental=[220000, 360000]  # INVALID: missing value
            )


class TestMMMEstimateSchema:
    """Test MMMEstimate schema validation."""
    
    def test_mmm_estimate_schema(self):
        """Valid MMM estimate passes validation."""
        curve = ROICurve(
            spend=[0, 100000, 200000, 300000],
            revenue_incremental=[0, 220000, 380000, 500000]
        )
        
        estimate = MMMEstimate(
            channel="meta",
            alpha=0.68,
            beta=0.85,
            theta=250000.0,
            adstock_alpha=0.42,
            half_life_days=12,
            roi_curve=curve,
            base_contribution=0.35,
            incremental_contribution=0.25,
            confidence_interval={"lower": 0.20, "upper": 0.30},
            trained_at=date(2025, 10, 15)
        )
        
        assert estimate.channel == "meta"
        assert estimate.half_life_days == 12
        assert 0.20 <= estimate.incremental_contribution <= 0.30
    
    def test_mmm_confidence_interval_validation(self):
        """Confidence interval must have lower/upper bounds."""
        curve = ROICurve(
            spend=[0, 100000],
            revenue_incremental=[0, 200000]
        )
        
        with pytest.raises(ValueError, match="must have 'lower' and 'upper' keys"):
            MMMEstimate(
                channel="google",
                alpha=0.5,
                beta=0.8,
                theta=150000.0,
                adstock_alpha=0.3,
                half_life_days=10,
                roi_curve=curve,
                base_contribution=0.4,
                incremental_contribution=0.2,
                confidence_interval={"min": 0.1, "max": 0.3},  # INVALID: wrong keys
                trained_at=date(2025, 10, 15)
            )
    
    def test_mmm_ci_bounds_validation(self):
        """CI lower cannot exceed upper."""
        curve = ROICurve(
            spend=[0, 100000],
            revenue_incremental=[0, 200000]
        )
        
        with pytest.raises(ValueError, match="lower bound cannot exceed upper"):
            MMMEstimate(
                channel="tiktok",
                alpha=0.6,
                beta=0.9,
                theta=180000.0,
                adstock_alpha=0.35,
                half_life_days=8,
                roi_curve=curve,
                base_contribution=0.3,
                incremental_contribution=0.15,
                confidence_interval={"lower": 0.25, "upper": 0.10},  # INVALID
                trained_at=date(2025, 10, 15)
            )


class TestAllocationPlanSchema:
    """Test AllocationPlan schema validation."""
    
    def test_allocation_plan_valid(self):
        """Valid allocation plan passes validation."""
        plan = AllocationPlan(
            week_start=date(2025, 10, 21),
            total_budget=500000.0,
            allocations={
                "meta": 175000.0,
                "google": 140000.0,
                "tiktok": 75000.0,
                "youtube": 60000.0,
                "organic": 50000.0
            },
            expected_revenue=1250000.0,
            expected_roas=2.5,
            confidence=0.82,
            requires_approval=False,
            constraints_applied=["min_roas_2.0", "max_shift_25pct"]
        )
        
        assert plan.total_budget == 500000.0
        assert plan.expected_roas == 2.5
    
    def test_allocation_sum_validation(self):
        """Allocations must sum to total_budget."""
        with pytest.raises(ValueError, match="must equal total_budget"):
            AllocationPlan(
                week_start=date(2025, 10, 21),
                total_budget=500000.0,
                allocations={
                    "meta": 200000.0,
                    "google": 150000.0,
                    "tiktok": 100000.0  # Sum = 450k, not 500k
                },
                expected_revenue=1125000.0,
                expected_roas=2.25,
                confidence=0.75
            )
    
    def test_allocation_rounding_tolerance(self):
        """Small rounding errors (<1%) are allowed."""
        # Total: 500000, Allocated: 499950 (50 difference = 0.01%)
        plan = AllocationPlan(
            week_start=date(2025, 10, 21),
            total_budget=500000.0,
            allocations={
                "meta": 174950.0,
                "google": 140000.0,
                "tiktok": 75000.0,
                "youtube": 60000.0,
                "organic": 50000.0
            },
            expected_revenue=1250000.0,
            expected_roas=2.5,
            confidence=0.82
        )
        
        assert sum(plan.allocations.values()) == 499950.0


class TestSaturationCurveSchema:
    """Test SaturationCurve (Hill function) evaluation."""
    
    def test_saturation_curve_evaluate(self):
        """Hill function evaluates correctly."""
        curve = SaturationCurve(
            alpha=1.0,
            beta=1.0,
            theta=100000.0
        )
        
        # At theta, should be ~0.5 * alpha
        effect_at_theta = curve.evaluate(100000.0)
        assert 0.49 <= effect_at_theta <= 0.51
        
        # At 0, should be ~0
        effect_at_zero = curve.evaluate(0.0)
        assert effect_at_zero < 0.01


class TestAdstockTransformSchema:
    """Test AdstockTransform parameters."""
    
    def test_adstock_transform_valid(self):
        """Valid adstock parameters pass validation."""
        adstock = AdstockTransform(
            decay_rate=0.42,
            half_life_days=12
        )
        
        assert 0 <= adstock.decay_rate <= 1
        assert adstock.half_life_days > 0
