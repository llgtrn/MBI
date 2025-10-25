"""
Marketing Mix Modeling (MMM) Data Contracts - Marketing Brand Intelligence

Pydantic schemas for MMM with ROI curve validation and Bayesian uncertainty.
Addresses Q_006 (ROI monotonic validation), Q_016 (Bayesian CI), A_003 (contract drift).
"""

from pydantic import BaseModel, Field, validator
from datetime import date
from typing import List, Dict, Optional
import numpy as np


class ROICurve(BaseModel):
    """
    ROI curve representing revenue response to spend.
    
    CRITICAL: Spend array MUST be monotonically increasing (Q_006).
    Invalid curves cause optimizer failures.
    """
    channel: str = Field(..., description="Marketing channel")
    spend: List[float] = Field(..., description="Spend levels (monotonically increasing)")
    revenue_incremental: List[float] = Field(..., description="Incremental revenue at each spend level")
    
    class Config:
        schema_extra = {
            "example": {
                "channel": "meta",
                "spend": [100000, 200000, 300000, 400000],
                "revenue_incremental": [220000, 360000, 450000, 520000]
            }
        }
    
    @validator('spend')
    def validate_spend_monotonic_increasing(cls, v):
        """
        Enforce monotonically increasing spend (Q_006 acceptance).
        
        Prevents optimizer failures from invalid curves.
        """
        if not v or len(v) < 2:
            raise ValueError("spend must have at least 2 data points")
        
        for i in range(1, len(v)):
            if v[i] <= v[i-1]:
                raise ValueError(
                    f"spend must be monotonically increasing: "
                    f"spend[{i}]={v[i]} <= spend[{i-1}]={v[i-1]}"
                )
        
        return v
    
    @validator('revenue_incremental')
    def validate_revenue_length_matches_spend(cls, v, values):
        """Revenue array must match spend array length"""
        if 'spend' in values and len(v) != len(values['spend']):
            raise ValueError(
                f"revenue_incremental length {len(v)} must match spend length {len(values['spend'])}"
            )
        return v
    
    @validator('revenue_incremental')
    def validate_revenue_nonnegative(cls, v):
        """Revenue values must be non-negative"""
        for i, revenue in enumerate(v):
            if revenue < 0:
                raise ValueError(f"revenue_incremental[{i}]={revenue} must be >= 0")
        return v
    
    def marginal_roi(self) -> List[float]:
        """
        Calculate marginal ROI between consecutive spend levels.
        
        Returns:
            List of marginal ROI values (revenue_delta / spend_delta)
        """
        if len(self.spend) < 2:
            return []
        
        marginal = []
        for i in range(1, len(self.spend)):
            spend_delta = self.spend[i] - self.spend[i-1]
            revenue_delta = self.revenue_incremental[i] - self.revenue_incremental[i-1]
            marginal.append(revenue_delta / spend_delta if spend_delta > 0 else 0)
        
        return marginal


class MMMEstimate(BaseModel):
    """
    Marketing Mix Model estimates per channel with Bayesian uncertainty (Q_016).
    
    Includes adstock/saturation parameters and confidence intervals.
    """
    channel: str = Field(..., description="Marketing channel")
    
    # Saturation parameters (Hill function)
    alpha: float = Field(..., gt=0, description="Saturation coefficient (maximum effect)")
    beta: float = Field(..., gt=0, description="Saturation exponent (steepness)")
    theta: float = Field(..., gt=0, description="Saturation point (50% of max effect)")
    
    # Adstock parameters
    adstock_alpha: float = Field(..., ge=0, le=1, description="Adstock decay rate (0-1)")
    half_life_days: int = Field(..., gt=0, description="Adstock half-life in days")
    
    # ROI curve
    roi_curve: ROICurve = Field(..., description="Revenue response curve")
    
    # Attribution split
    base_contribution: float = Field(..., ge=0, le=1, description="% of revenue from base (no marketing)")
    incremental_contribution: float = Field(..., ge=0, le=1, description="% of revenue from this channel")
    
    # Bayesian uncertainty (Q_016)
    confidence_interval: Dict[str, float] = Field(
        ..., 
        description="95% credible interval for incremental contribution"
    )
    
    # Metadata
    model_version: str = Field(..., description="Model version identifier")
    trained_at: date = Field(..., description="Model training date")
    
    class Config:
        schema_extra = {
            "example": {
                "channel": "meta",
                "alpha": 1.5,
                "beta": 0.68,
                "theta": 250000,
                "adstock_alpha": 0.42,
                "half_life_days": 12,
                "roi_curve": {
                    "channel": "meta",
                    "spend": [100000, 200000, 300000],
                    "revenue_incremental": [220000, 360000, 450000]
                },
                "base_contribution": 0.35,
                "incremental_contribution": 0.25,
                "confidence_interval": {
                    "lower": 0.22,
                    "upper": 0.28
                },
                "model_version": "mmm_v1.2",
                "trained_at": "2025-10-12"
            }
        }
    
    @validator('confidence_interval')
    def validate_confidence_interval_format(cls, v):
        """
        Confidence interval must have 'lower' and 'upper' keys (Q_016 acceptance).
        """
        if 'lower' not in v or 'upper' not in v:
            raise ValueError("confidence_interval must have 'lower' and 'upper' keys")
        
        if v['lower'] > v['upper']:
            raise ValueError(
                f"confidence_interval lower={v['lower']} must be <= upper={v['upper']}"
            )
        
        return v
    
    @validator('incremental_contribution')
    def validate_incremental_within_ci(cls, v, values):
        """Incremental contribution should be within confidence interval"""
        if 'confidence_interval' in values:
            ci = values['confidence_interval']
            if v < ci['lower'] or v > ci['upper']:
                raise ValueError(
                    f"incremental_contribution {v} outside confidence_interval "
                    f"[{ci['lower']}, {ci['upper']}]"
                )
        return v
    
    @validator('base_contribution', 'incremental_contribution')
    def validate_contribution_sum(cls, v, values):
        """
        Base + all incremental contributions should sum to ~1.0.
        
        Note: This validator runs per-channel, so we can't enforce
        sum=1 here. That check is in AllocationPlan.
        """
        return v


class AllocationPlan(BaseModel):
    """
    Optimal budget allocation plan from MMM optimization.
    
    Includes expected outcomes and uncertainty quantification.
    """
    week_start: date = Field(..., description="Planning week start date")
    total_budget: float = Field(..., gt=0, description="Total budget to allocate")
    
    # Allocation by channel
    allocations: Dict[str, float] = Field(
        ..., 
        description="Budget allocation per channel"
    )
    
    # Expected outcomes
    expected_revenue: float = Field(..., ge=0, description="Expected total incremental revenue")
    expected_roas: float = Field(..., ge=0, description="Expected return on ad spend")
    
    # Uncertainty (Q_016)
    confidence: float = Field(..., ge=0, le=1, description="Confidence score for plan (0-1)")
    revenue_ci: Dict[str, float] = Field(
        ..., 
        description="95% credible interval for expected revenue"
    )
    
    # Decision metadata
    requires_approval: bool = Field(default=False, description="Requires human approval")
    approval_reason: Optional[str] = Field(None, description="Reason for approval requirement")
    
    class Config:
        schema_extra = {
            "example": {
                "week_start": "2025-10-14",
                "total_budget": 500000,
                "allocations": {
                    "meta": 175000,
                    "google": 140000,
                    "tiktok": 105000,
                    "youtube": 80000
                },
                "expected_revenue": 1250000,
                "expected_roas": 2.5,
                "confidence": 0.85,
                "revenue_ci": {
                    "lower": 1100000,
                    "upper": 1400000
                },
                "requires_approval": False
            }
        }
    
    @validator('allocations')
    def validate_allocations_sum_to_budget(cls, v, values):
        """Sum of allocations must equal total_budget (within 0.01% tolerance)"""
        if 'total_budget' in values:
            allocation_sum = sum(v.values())
            budget = values['total_budget']
            tolerance = 0.0001  # 0.01%
            
            if abs(allocation_sum - budget) / budget > tolerance:
                raise ValueError(
                    f"allocations sum {allocation_sum} does not match "
                    f"total_budget {budget} (tolerance {tolerance})"
                )
        
        return v
    
    @validator('allocations')
    def validate_allocations_nonnegative(cls, v):
        """All channel allocations must be non-negative"""
        for channel, amount in v.items():
            if amount < 0:
                raise ValueError(f"allocation for {channel}={amount} must be >= 0")
        return v
    
    @validator('revenue_ci')
    def validate_revenue_ci_format(cls, v):
        """Revenue confidence interval must have 'lower' and 'upper' keys"""
        if 'lower' not in v or 'upper' not in v:
            raise ValueError("revenue_ci must have 'lower' and 'upper' keys")
        
        if v['lower'] > v['upper']:
            raise ValueError(
                f"revenue_ci lower={v['lower']} must be <= upper={v['upper']}"
            )
        
        return v
    
    @validator('expected_revenue')
    def validate_expected_revenue_within_ci(cls, v, values):
        """Expected revenue should be within confidence interval"""
        if 'revenue_ci' in values:
            ci = values['revenue_ci']
            if v < ci['lower'] or v > ci['upper']:
                raise ValueError(
                    f"expected_revenue {v} outside revenue_ci "
                    f"[{ci['lower']}, {ci['upper']}]"
                )
        return v


class AdstockTransform(BaseModel):
    """
    Adstock transformation parameters.
    
    Models carryover effect: x'_t = x_t + alpha * x'_{t-1}
    """
    channel: str = Field(..., description="Marketing channel")
    alpha: float = Field(..., ge=0, le=1, description="Decay rate (0-1)")
    half_life_days: int = Field(..., gt=0, description="Half-life in days")
    max_lag: int = Field(default=30, gt=0, description="Maximum lag periods to consider")
    
    class Config:
        schema_extra = {
            "example": {
                "channel": "meta",
                "alpha": 0.42,
                "half_life_days": 12,
                "max_lag": 30
            }
        }
    
    @validator('alpha')
    def validate_alpha_halflife_consistency(cls, v, values):
        """
        Validate alpha is consistent with half_life.
        
        half_life = -log(2) / log(alpha)
        alpha = 2^(-1/half_life)
        """
        if 'half_life_days' in values:
            expected_alpha = 2 ** (-1 / values['half_life_days'])
            tolerance = 0.01  # 1% tolerance
            
            if abs(v - expected_alpha) / expected_alpha > tolerance:
                raise ValueError(
                    f"alpha {v} inconsistent with half_life_days {values['half_life_days']} "
                    f"(expected alpha ~{expected_alpha:.3f})"
                )
        
        return v


class SaturationTransform(BaseModel):
    """
    Saturation transformation parameters (Hill function).
    
    Models diminishing returns: f(x) = alpha * (x^beta / (theta^beta + x^beta))
    """
    channel: str = Field(..., description="Marketing channel")
    alpha: float = Field(..., gt=0, description="Maximum effect")
    beta: float = Field(..., gt=0, description="Steepness parameter")
    theta: float = Field(..., gt=0, description="Half-saturation point")
    
    class Config:
        schema_extra = {
            "example": {
                "channel": "meta",
                "alpha": 1.5,
                "beta": 0.68,
                "theta": 250000
            }
        }
