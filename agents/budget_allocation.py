"""
Budget Allocation Agent - Optimal budget distribution with idempotency
"""
import logging
from typing import Dict, List, Optional
from datetime import date, datetime
from prometheus_client import Counter
import uuid

from core.contracts import BudgetAllocation, AllocationResult
from core.database import Database
from core.exceptions import DuplicateAllocationError

logger = logging.getLogger(__name__)

# Prometheus metrics
allocation_duplicate_rejection = Counter(
    'budget_allocation_duplicate_rejection_total',
    'Duplicate allocation_id rejections',
    ['allocation_id']
)

allocation_applied = Counter(
    'budget_allocation_applied_total',
    'Successful allocations',
    ['week_start']
)

# Configuration
MAX_BUDGET_LIMIT = 5000000  # Maximum weekly budget in base currency
BUDGET_ALLOCATION_ENABLED = True  # Kill switch


class BudgetAllocationAgent:
    """
    Budget Allocation Agent with idempotency enforcement.
    Prevents double-spend on retries via allocation_id uniqueness.
    """
    
    def __init__(self):
        self.db = Database()
        self._allocation_cache = {}  # In-memory dedup cache
        
    def apply_allocation(
        self,
        allocation: BudgetAllocation
    ) -> AllocationResult:
        """
        Apply budget allocation with idempotency enforcement.
        
        Args:
            allocation: BudgetAllocation with unique allocation_id
            
        Returns:
            AllocationResult with success status
            
        Raises:
            DuplicateAllocationError: If allocation_id already exists
            ValueError: If validation fails
        """
        if not BUDGET_ALLOCATION_ENABLED:
            raise RuntimeError("Budget allocation is disabled (kill switch)")
        
        # Validate allocation
        self._validate_allocation(allocation)
        
        # Check idempotency: allocation_id must be unique
        if self._allocation_exists(allocation.allocation_id):
            logger.warning(
                f"allocation_duplicate_rejection: allocation_id={allocation.allocation_id} "
                f"already exists - rejecting duplicate"
            )
            allocation_duplicate_rejection.labels(
                allocation_id=allocation.allocation_id
            ).inc()
            
            raise DuplicateAllocationError(
                f"Allocation allocation_id={allocation.allocation_id} already exists. "
                f"This prevents double-spend on retries."
            )
        
        # Apply allocation to database
        try:
            self.db.execute(
                """
                INSERT INTO budget_allocations (
                    allocation_id, week_start, total_budget, allocations,
                    expected_revenue, expected_roas, confidence,
                    requires_approval, approved, created_at
                ) VALUES (
                    %(allocation_id)s, %(week_start)s, %(total_budget)s, %(allocations)s,
                    %(expected_revenue)s, %(expected_roas)s, %(confidence)s,
                    %(requires_approval)s, %(approved)s, %(created_at)s
                )
                """,
                {
                    'allocation_id': allocation.allocation_id,
                    'week_start': allocation.week_start,
                    'total_budget': allocation.total_budget,
                    'allocations': allocation.allocations,  # JSON
                    'expected_revenue': allocation.expected_revenue,
                    'expected_roas': allocation.expected_roas,
                    'confidence': allocation.confidence,
                    'requires_approval': allocation.requires_approval,
                    'approved': not allocation.requires_approval,
                    'created_at': datetime.utcnow()
                }
            )
            
            # Add to cache
            self._allocation_cache[allocation.allocation_id] = True
            
            # Emit success metric
            allocation_applied.labels(
                week_start=str(allocation.week_start)
            ).inc()
            
            logger.info(
                f"Budget allocation applied successfully: "
                f"allocation_id={allocation.allocation_id}, "
                f"total_budget={allocation.total_budget}"
            )
            
            return AllocationResult(
                success=True,
                allocation_id=allocation.allocation_id,
                applied_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(
                f"Failed to apply allocation: allocation_id={allocation.allocation_id}, "
                f"error={e}"
            )
            raise
    
    def _validate_allocation(self, allocation: BudgetAllocation):
        """
        Validate allocation before applying.
        
        Args:
            allocation: BudgetAllocation to validate
            
        Raises:
            ValueError: If validation fails
        """
        # Check budget limit
        if allocation.total_budget > MAX_BUDGET_LIMIT:
            raise ValueError(
                f"Allocation total_budget={allocation.total_budget} "
                f"exceeds budget limit={MAX_BUDGET_LIMIT}"
            )
        
        # Check allocations sum to total
        allocations_sum = sum(allocation.allocations.values())
        if abs(allocations_sum - allocation.total_budget) > 0.01:
            raise ValueError(
                f"Sum of allocations ({allocations_sum}) does not match "
                f"total_budget ({allocation.total_budget})"
            )
        
        # Check all channels have non-negative budgets
        for channel, amount in allocation.allocations.items():
            if amount < 0:
                raise ValueError(
                    f"Channel {channel} has negative budget: {amount}"
                )
        
        # Check allocation_id format
        if not allocation.allocation_id or len(allocation.allocation_id) < 5:
            raise ValueError(
                f"Invalid allocation_id: {allocation.allocation_id}"
            )
    
    def _allocation_exists(self, allocation_id: str) -> bool:
        """
        Check if allocation_id already exists (idempotency check).
        
        Args:
            allocation_id: Unique allocation identifier
            
        Returns:
            True if allocation exists, False otherwise
        """
        # Check cache first
        if allocation_id in self._allocation_cache:
            return True
        
        # Check database
        result = self.db.query_one(
            """
            SELECT allocation_id FROM budget_allocations
            WHERE allocation_id = %(allocation_id)s
            """,
            {'allocation_id': allocation_id}
        )
        
        exists = result is not None
        
        if exists:
            # Add to cache
            self._allocation_cache[allocation_id] = True
        
        return exists
    
    def get_allocation_schema(self) -> Dict:
        """
        Get database schema for allocations (for testing).
        
        Returns:
            Schema definition with constraints
        """
        return {
            'table': 'budget_allocations',
            'columns': {
                'allocation_id': 'STRING PRIMARY KEY',
                'week_start': 'DATE NOT NULL',
                'total_budget': 'FLOAT64 NOT NULL',
                'allocations': 'JSON NOT NULL',
                'expected_revenue': 'FLOAT64',
                'expected_roas': 'FLOAT64',
                'confidence': 'FLOAT64',
                'requires_approval': 'BOOL',
                'approved': 'BOOL',
                'approved_by': 'STRING',
                'approved_at': 'TIMESTAMP',
                'created_at': 'TIMESTAMP NOT NULL'
            },
            'unique_constraints': {
                'allocation_id': True  # UNIQUE constraint enforced
            },
            'indexes': [
                'week_start',
                'created_at'
            ]
        }
    
    async def optimize_allocation(
        self,
        total_budget: float,
        constraints: Dict,
        mmm_curves: Dict
    ) -> BudgetAllocation:
        """
        Optimize budget allocation using MMM curves.
        
        Args:
            total_budget: Total budget to allocate
            constraints: Channel constraints (min/max)
            mmm_curves: ROI curves from MMM agent
            
        Returns:
            Optimized BudgetAllocation
        """
        from scipy.optimize import minimize
        
        channels = list(mmm_curves.keys())
        
        def objective(allocations):
            """Maximize total incremental revenue"""
            return -sum(
                mmm_curves[ch].predict(alloc)
                for ch, alloc in zip(channels, allocations)
            )
        
        # Constraints
        bounds = [
            (constraints.get(ch, {}).get('min', 0),
             constraints.get(ch, {}).get('max', total_budget))
            for ch in channels
        ]
        
        constraint_sum = {
            'type': 'eq',
            'fun': lambda x: sum(x) - total_budget
        }
        
        # Initial guess: equal split
        x0 = [total_budget / len(channels)] * len(channels)
        
        # Solve
        result = minimize(
            objective,
            x0=x0,
            bounds=bounds,
            constraints=[constraint_sum],
            method='SLSQP'
        )
        
        if not result.success:
            logger.warning(f"Optimization did not converge: {result.message}")
        
        # Build allocation
        allocations = dict(zip(channels, result.x))
        expected_revenue = sum(
            mmm_curves[ch].predict(alloc)
            for ch, alloc in allocations.items()
        )
        
        # Generate unique allocation_id
        allocation_id = f"alloc_{uuid.uuid4().hex[:12]}"
        
        return BudgetAllocation(
            allocation_id=allocation_id,
            week_start=self._get_next_week_start(),
            total_budget=total_budget,
            allocations=allocations,
            expected_revenue=expected_revenue,
            expected_roas=expected_revenue / total_budget if total_budget > 0 else 0,
            confidence=0.85,  # Model confidence
            requires_approval=total_budget > 1000000  # Require approval for large budgets
        )
    
    def _get_next_week_start(self) -> date:
        """Get next Monday as week start date"""
        from datetime import timedelta
        today = date.today()
        days_until_monday = (7 - today.weekday()) % 7
        if days_until_monday == 0:
            days_until_monday = 7
        return today + timedelta(days=days_until_monday)
