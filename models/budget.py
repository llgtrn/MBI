"""
Budget Allocation Models - Schema with Idempotency & LWW
"""
from datetime import date, datetime
from typing import Dict, Optional
from pydantic import BaseModel, Field, validator
import hashlib
import json


class BudgetAllocation(BaseModel):
    """
    Budget allocation recommendation with idempotency & LWW merge support
    """
    allocation_id: str = Field(..., description="Unique allocation identifier")
    allocation_id_hash: str = Field(
        ..., 
        description="SHA256 hash of (allocations_json + total_budget + constraints_json + week_start) for idempotency"
    )
    week_start: date = Field(..., description="Start date of allocation week (Monday)")
    total_budget: float = Field(..., ge=0, description="Total budget to allocate")
    allocations: Dict[str, float] = Field(
        ..., 
        description="Channel allocations {channel: amount}"
    )
    expected_revenue: Optional[float] = Field(None, description="Expected revenue from this allocation")
    expected_roas: Optional[float] = Field(None, description="Expected ROAS")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Confidence score [0,1]")
    requires_approval: bool = Field(False, description="Requires human approval")
    created_at: datetime = Field(..., description="Allocation creation timestamp")
    lww_timestamp: datetime = Field(
        ..., 
        description="Last-write-wins timestamp for conflict resolution"
    )
    approved: Optional[bool] = Field(None, description="Approval status")
    approved_by: Optional[str] = Field(None, description="Approver user ID")
    approved_at: Optional[datetime] = Field(None, description="Approval timestamp")
    
    @validator('allocations')
    def validate_allocations_sum(cls, v, values):
        """Validate allocations sum to total_budget (within 0.01% tolerance)"""
        if 'total_budget' in values:
            total = sum(v.values())
            expected = values['total_budget']
            tolerance = expected * 0.0001  # 0.01%
            if abs(total - expected) > tolerance:
                raise ValueError(f"Allocations sum {total} != total_budget {expected}")
        return v
    
    @validator('allocation_id_hash')
    def validate_hash_format(cls, v):
        """Validate hash is 64-char hex (SHA256)"""
        if not (len(v) == 64 and all(c in '0123456789abcdef' for c in v)):
            raise ValueError("allocation_id_hash must be 64-char hex SHA256")
        return v
    
    @staticmethod
    def compute_hash(
        allocations: Dict[str, float],
        total_budget: float,
        constraints: Dict,
        week_start: date
    ) -> str:
        """
        Compute deterministic allocation_id_hash for idempotency
        
        Formula: SHA256(allocations_json | total_budget | constraints_json | week_start)
        """
        allocations_json = json.dumps(allocations, sort_keys=True)
        constraints_json = json.dumps(constraints, sort_keys=True)
        week_start_str = week_start.isoformat()
        
        input_str = f"{allocations_json}|{total_budget}|{constraints_json}|{week_start_str}"
        return hashlib.sha256(input_str.encode()).hexdigest()
    
    class Config:
        json_schema_extra = {
            "example": {
                "allocation_id": "alloc_20251019_001",
                "allocation_id_hash": "a1b2c3d4e5f6...",
                "week_start": "2025-10-14",
                "total_budget": 500000,
                "allocations": {
                    "meta": 175000,
                    "google": 125000,
                    "tiktok": 100000,
                    "youtube": 100000
                },
                "expected_revenue": 1250000,
                "expected_roas": 2.5,
                "confidence": 0.85,
                "requires_approval": False,
                "created_at": "2025-10-19T15:00:00Z",
                "lww_timestamp": "2025-10-19T15:00:00Z"
            }
        }


class BudgetConstraints(BaseModel):
    """Budget allocation constraints"""
    min_roas: Optional[float] = Field(None, ge=0, description="Minimum ROAS target")
    max_cac: Optional[float] = Field(None, ge=0, description="Maximum CAC target")
    max_shift_pct: float = Field(
        250, 
        ge=0, 
        le=500, 
        description="Maximum relative budget shift per channel (percentage, default 250%)"
    )
    channel_limits: Optional[Dict[str, Dict[str, float]]] = Field(
        None,
        description="Per-channel min/max limits {channel: {min: x, max: y}}"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "min_roas": 2.0,
                "max_cac": 5000,
                "max_shift_pct": 250,
                "channel_limits": {
                    "meta": {"min": 50000, "max": 300000},
                    "google": {"min": 30000, "max": 200000}
                }
            }
        }


class BudgetAllocationPlan(BaseModel):
    """Budget allocation plan with metadata"""
    week_start: date
    total_budget: float
    allocations: Dict[str, float]
    expected_revenue: float
    expected_roas: float
    confidence: float
    requires_approval: bool
    created_at: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "week_start": "2025-10-14",
                "total_budget": 500000,
                "allocations": {"meta": 175000, "google": 125000},
                "expected_revenue": 1250000,
                "expected_roas": 2.5,
                "confidence": 0.85,
                "requires_approval": False,
                "created_at": "2025-10-19T15:00:00Z"
            }
        }
