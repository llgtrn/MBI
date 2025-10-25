"""
Budget Allocation Agent - Idempotency, Cap Enforcement, LWW Merge
"""
from datetime import date, datetime
from typing import Dict, List, Optional
import hashlib
import json
from models.budget import BudgetAllocation, BudgetConstraints
from prometheus_client import Counter


# Prometheus metrics
budget_allocation_cap_violations = Counter(
    'budget_allocation_cap_violations_total',
    'Total budget allocation cap violations',
    ['channel', 'reason']
)


class BudgetAllocationAgent:
    """
    Optimal budget allocation with:
    - Idempotency via allocation_id_hash
    - Relative cap enforcement (max 250% shift per channel)
    - LWW merge for concurrent allocations
    """
    
    def __init__(self, feature_store=None):
        self.feature_store = feature_store or get_feature_store()
    
    def optimize_allocation(
        self,
        total_budget: float,
        channels: List[str],
        week_start: date,
        constraints: Optional[Dict] = None,
        enforce_cap: bool = True
    ) -> BudgetAllocation:
        """
        Compute optimal budget allocation with idempotency and cap enforcement
        
        Args:
            total_budget: Total budget to allocate
            channels: List of channels to allocate across
            week_start: Start date of allocation week (Monday)
            constraints: Optional constraints (min_roas, max_cac, max_shift_pct, etc.)
            enforce_cap: Whether to enforce relative cap (default True)
        
        Returns:
            BudgetAllocation with allocation_id_hash for idempotency
        
        Raises:
            ValueError: If cap violation detected
        """
        constraints = constraints or {}
        constraints_obj = BudgetConstraints(**constraints)
        
        # Step 1: Optimize allocation (simplified linear optimization)
        allocations = self._optimize_internal(
            total_budget=total_budget,
            channels=channels,
            constraints=constraints_obj
        )
        
        # Step 2: Compute allocation_id_hash for idempotency
        allocation_id_hash = BudgetAllocation.compute_hash(
            allocations=allocations,
            total_budget=total_budget,
            constraints=constraints,
            week_start=week_start
        )
        
        # Step 3: Check if this allocation already exists (idempotency)
        existing = self.feature_store.get_allocation_by_hash(allocation_id_hash)
        if existing:
            return existing
        
        # Step 4: Enforce relative cap if enabled
        if enforce_cap:
            self._enforce_relative_cap(
                allocations=allocations,
                week_start=week_start,
                max_shift_pct=constraints_obj.max_shift_pct
            )
        
        # Step 5: Create new allocation
        now = datetime.utcnow()
        allocation = BudgetAllocation(
            allocation_id=f"alloc_{week_start.isoformat()}_{allocation_id_hash[:8]}",
            allocation_id_hash=allocation_id_hash,
            week_start=week_start,
            total_budget=total_budget,
            allocations=allocations,
            expected_revenue=self._estimate_revenue(allocations),
            expected_roas=self._estimate_roas(allocations),
            confidence=0.85,
            requires_approval=self._requires_approval(allocations, total_budget),
            created_at=now,
            lww_timestamp=now
        )
        
        # Step 6: Store allocation
        self.feature_store.save_allocation(allocation)
        
        return allocation
    
    def _optimize_internal(
        self,
        total_budget: float,
        channels: List[str],
        constraints: BudgetConstraints
    ) -> Dict[str, float]:
        """
        Internal optimization logic (simplified)
        
        In production, this would use scipy.optimize or MMM ROI curves.
        For now, equal split with channel limits applied.
        """
        allocations = {}
        per_channel = total_budget / len(channels)
        
        for channel in channels:
            amount = per_channel
            
            # Apply channel limits if specified
            if constraints.channel_limits and channel in constraints.channel_limits:
                limits = constraints.channel_limits[channel]
                if 'min' in limits:
                    amount = max(amount, limits['min'])
                if 'max' in limits:
                    amount = min(amount, limits['max'])
            
            allocations[channel] = amount
        
        # Normalize to total_budget (handle rounding)
        current_total = sum(allocations.values())
        if current_total != total_budget:
            scale = total_budget / current_total
            allocations = {k: v * scale for k, v in allocations.items()}
        
        return allocations
    
    def _enforce_relative_cap(
        self,
        allocations: Dict[str, float],
        week_start: date,
        max_shift_pct: float
    ):
        """
        Enforce relative budget cap: no channel can increase >max_shift_pct from baseline
        
        Raises:
            ValueError: If any channel exceeds cap
        """
        # Get previous week's allocation
        previous = self.feature_store.get_previous_allocation(week_start)
        
        if not previous:
            # No baseline, allow any allocation
            return
        
        violations = []
        
        for channel, new_amount in allocations.items():
            if channel not in previous:
                # New channel, no baseline
                continue
            
            baseline = previous[channel]
            if baseline == 0:
                # Avoid division by zero
                continue
            
            increase_pct = ((new_amount - baseline) / baseline) * 100
            
            if increase_pct > max_shift_pct:
                violations.append({
                    'channel': channel,
                    'baseline': baseline,
                    'new_amount': new_amount,
                    'increase_pct': increase_pct,
                    'max_shift_pct': max_shift_pct
                })
                
                # Emit metric
                budget_allocation_cap_violations.labels(
                    channel=channel,
                    reason='relative_cap_exceeded'
                ).inc()
        
        if violations:
            details = '; '.join([
                f"{v['channel']}: {v['baseline']:.0f}→{v['new_amount']:.0f} "
                f"({v['increase_pct']:.0f}% > {v['max_shift_pct']:.0f}%)"
                for v in violations
            ])
            raise ValueError(f"cap_violation: {details}")
    
    def _estimate_revenue(self, allocations: Dict[str, float]) -> float:
        """Estimate expected revenue from allocation (placeholder)"""
        # In production, use MMM ROI curves
        return sum(allocations.values()) * 2.5  # Assume 2.5x ROAS
    
    def _estimate_roas(self, allocations: Dict[str, float]) -> float:
        """Estimate expected ROAS (placeholder)"""
        return 2.5
    
    def _requires_approval(self, allocations: Dict[str, float], total_budget: float) -> bool:
        """Determine if allocation requires human approval"""
        # Require approval if total budget > $50K
        return total_budget > 50000
    
    def _lww_merge(self, allocations: List[BudgetAllocation]) -> BudgetAllocation:
        """
        Merge concurrent allocations using last-write-wins strategy
        
        Args:
            allocations: List of concurrent allocations
        
        Returns:
            Winning allocation (highest lww_timestamp)
        """
        if not allocations:
            raise ValueError("Cannot merge empty allocations list")
        
        if len(allocations) == 1:
            return allocations[0]
        
        # Sort by lww_timestamp descending, then by allocation_id descending (tie-breaker)
        sorted_allocs = sorted(
            allocations,
            key=lambda a: (a.lww_timestamp, a.allocation_id),
            reverse=True
        )
        
        return sorted_allocs[0]


def get_feature_store():
    """Get feature store instance (placeholder)"""
    return MockFeatureStore()


class MockFeatureStore:
    """Mock feature store for testing"""
    
    def __init__(self):
        self._allocations = {}
        self._previous = {}
    
    def get_allocation_by_hash(self, allocation_id_hash: str) -> Optional[BudgetAllocation]:
        """Get allocation by hash"""
        return self._allocations.get(allocation_id_hash)
    
    def save_allocation(self, allocation: BudgetAllocation):
        """Save allocation"""
        self._allocations[allocation.allocation_id_hash] = allocation
    
    def get_previous_allocation(self, week_start: date) -> Optional[Dict[str, float]]:
        """Get previous week's allocation (simplified)"""
        return self._previous.get(week_start)
    
    def set_previous_allocation(self, week_start: date, allocations: Dict[str, float]):
        """Set previous allocation (for testing)"""
        self._previous[week_start] = allocations


def prometheus_counter(name: str, labels: Dict[str, str]):
    """Emit Prometheus counter (wrapper for testing)"""
    budget_allocation_cap_violations.labels(**labels).inc()
