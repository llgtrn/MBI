"""
Unit tests for Budget Allocation Agent - Idempotency enforcement
"""
import pytest
from datetime import date
from agents.budget_allocation import BudgetAllocationAgent
from core.contracts import BudgetAllocation
from core.exceptions import DuplicateAllocationError


class TestBudgetAllocationIdempotency:
    """Test allocation_id uniqueness enforcement"""
    
    @pytest.fixture
    def allocation_agent(self):
        return BudgetAllocationAgent()
    
    def test_allocation_idempotency_prevents_duplicate(self, allocation_agent):
        """Duplicate allocation_id must be rejected"""
        allocation = BudgetAllocation(
            allocation_id="alloc_001",
            week_start=date(2025, 10, 13),
            total_budget=500000,
            allocations={
                "meta": 175000,
                "google": 150000,
                "tiktok": 100000,
                "youtube": 75000
            },
            expected_revenue=1250000,
            expected_roas=2.5,
            confidence=0.85,
            requires_approval=False
        )
        
        # First execution succeeds
        result1 = allocation_agent.apply_allocation(allocation)
        assert result1.success is True
        
        # Second execution with same allocation_id must fail
        with pytest.raises(DuplicateAllocationError) as exc_info:
            allocation_agent.apply_allocation(allocation)
        
        assert "allocation_id=alloc_001" in str(exc_info.value)
        assert "already exists" in str(exc_info.value)
    
    def test_allocation_idempotency_logs_rejection(self, allocation_agent, caplog):
        """Duplicate rejection must be logged"""
        allocation = BudgetAllocation(
            allocation_id="alloc_002",
            week_start=date(2025, 10, 13),
            total_budget=500000,
            allocations={"meta": 500000},
            expected_revenue=1000000,
            expected_roas=2.0,
            confidence=0.80,
            requires_approval=False
        )
        
        # First apply
        allocation_agent.apply_allocation(allocation)
        
        # Second apply (duplicate)
        try:
            allocation_agent.apply_allocation(allocation)
        except DuplicateAllocationError:
            pass
        
        # Check log
        assert "allocation_duplicate_rejection" in caplog.text
        assert "alloc_002" in caplog.text
    
    def test_allocation_idempotency_allows_different_ids(self, allocation_agent):
        """Different allocation_ids should succeed"""
        alloc1 = BudgetAllocation(
            allocation_id="alloc_003",
            week_start=date(2025, 10, 13),
            total_budget=500000,
            allocations={"meta": 500000},
            expected_revenue=1000000,
            expected_roas=2.0,
            confidence=0.80,
            requires_approval=False
        )
        
        alloc2 = BudgetAllocation(
            allocation_id="alloc_004",  # Different ID
            week_start=date(2025, 10, 13),
            total_budget=600000,
            allocations={"google": 600000},
            expected_revenue=1200000,
            expected_roas=2.0,
            confidence=0.80,
            requires_approval=False
        )
        
        result1 = allocation_agent.apply_allocation(alloc1)
        result2 = allocation_agent.apply_allocation(alloc2)
        
        assert result1.success is True
        assert result2.success is True
    
    def test_allocation_db_constraint_enforcement(self, allocation_agent):
        """Database must enforce unique constraint on allocation_id"""
        # Verify schema has unique constraint
        schema = allocation_agent.get_allocation_schema()
        
        assert 'allocation_id' in schema['unique_constraints']
        assert schema['unique_constraints']['allocation_id'] is True


class TestBudgetAllocationRiskGates:
    """Test risk gates and validation"""
    
    def test_allocation_exceeds_budget_limit(self):
        """Allocation exceeding limit must be rejected"""
        agent = BudgetAllocationAgent()
        
        allocation = BudgetAllocation(
            allocation_id="alloc_005",
            week_start=date(2025, 10, 13),
            total_budget=10000000,  # Exceeds limit
            allocations={"meta": 10000000},
            expected_revenue=20000000,
            expected_roas=2.0,
            confidence=0.80,
            requires_approval=False
        )
        
        with pytest.raises(ValueError) as exc_info:
            agent.apply_allocation(allocation)
        
        assert "exceeds budget limit" in str(exc_info.value)
    
    def test_allocation_sum_validation(self):
        """Channel allocations must sum to total_budget"""
        agent = BudgetAllocationAgent()
        
        allocation = BudgetAllocation(
            allocation_id="alloc_006",
            week_start=date(2025, 10, 13),
            total_budget=500000,
            allocations={
                "meta": 200000,
                "google": 200000
                # Sum = 400000 != total_budget
            },
            expected_revenue=1000000,
            expected_roas=2.0,
            confidence=0.80,
            requires_approval=False
        )
        
        with pytest.raises(ValueError) as exc_info:
            agent.apply_allocation(allocation)
        
        assert "sum of allocations" in str(exc_info.value)
