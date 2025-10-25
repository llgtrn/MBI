"""
Test suite for budget allocation safety: shift limits and idempotency.
Validates Q_053 (≤25% shift limit) and Q_054 (allocation_id deduplication).
"""

import pytest
from datetime import date, datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from agents.budget_allocation_agent import BudgetAllocationAgent, BudgetShiftViolation
from core.idempotency import IdempotencyError


class TestBudgetShiftLimits:
    """Q_053: Budget shift limit ≤25% per week"""

    @pytest.fixture
    def budget_agent(self):
        """Create BudgetAllocationAgent with mocked dependencies"""
        agent = BudgetAllocationAgent(
            mmm_agent=AsyncMock(),
            mta_agent=AsyncMock(),
            feature_store=Mock(),
            metrics_client=Mock()
        )
        return agent

    @pytest.mark.asyncio
    async def test_budget_shift_limit_rejects_over_25(self, budget_agent):
        """
        ACCEPTANCE: Allocation rejected if shift >25%
        METRIC: budget_shift_violations (counter)
        """
        # Current allocation
        current_allocation = {
            "meta": 400000,
            "google": 300000,
            "tiktok": 200000,
            "youtube": 100000
        }
        
        # Proposed allocation with >25% shift on meta (400k -> 600k = +50%)
        proposed_allocation = {
            "meta": 600000,
            "google": 200000,
            "tiktok": 150000,
            "youtube": 50000
        }
        
        with pytest.raises(BudgetShiftViolation) as exc_info:
            await budget_agent.validate_allocation_shift(
                current=current_allocation,
                proposed=proposed_allocation,
                max_shift_pct=0.25
            )
        
        # Verify error details
        assert "meta" in str(exc_info.value)
        assert "50.0%" in str(exc_info.value) or "0.50" in str(exc_info.value)
        
        # Metric validation
        budget_agent.metrics_client.increment.assert_called_with(
            'budget_shift_violations',
            tags=['channel:meta', 'shift_pct:0.50']
        )

    @pytest.mark.asyncio
    async def test_budget_shift_within_25_accepted(self, budget_agent):
        """Verify allocation within ≤25% shift is accepted"""
        current = {
            "meta": 400000,
            "google": 300000,
            "tiktok": 200000,
            "youtube": 100000
        }
        
        # Proposed with exactly 25% shift on meta (400k -> 500k = +25%)
        proposed = {
            "meta": 500000,
            "google": 250000,
            "tiktok": 150000,
            "youtube": 100000
        }
        
        # Should not raise
        result = await budget_agent.validate_allocation_shift(
            current=current,
            proposed=proposed,
            max_shift_pct=0.25
        )
        
        assert result is True

    @pytest.mark.asyncio
    async def test_budget_shift_requires_approval_over_25(self, budget_agent):
        """Verify shifts >25% trigger approval workflow"""
        current = {"meta": 400000, "google": 300000, "tiktok": 200000, "youtube": 100000}
        proposed = {"meta": 600000, "google": 200000, "tiktok": 150000, "youtube": 50000}
        
        with patch.object(budget_agent, 'request_approval', new=AsyncMock()) as mock_approval:
            await budget_agent.apply_allocation_with_approval(
                current=current,
                proposed=proposed,
                max_shift_pct=0.25
            )
            
            # Verify approval requested
            mock_approval.assert_called_once()
            approval_args = mock_approval.call_args[1]
            assert approval_args['reason'] == 'budget_shift_exceeds_limit'
            assert 'meta' in approval_args['details']

    @pytest.mark.asyncio
    async def test_budget_shift_calculation_accuracy(self, budget_agent):
        """Verify shift percentage calculation is accurate"""
        shifts = await budget_agent.calculate_shifts(
            current={"meta": 100000, "google": 50000},
            proposed={"meta": 130000, "google": 45000}
        )
        
        assert shifts["meta"] == pytest.approx(0.30, abs=0.01)  # +30%
        assert shifts["google"] == pytest.approx(-0.10, abs=0.01)  # -10%


class TestBudgetAllocationIdempotency:
    """Q_054: Budget allocation_id dedup idempotent"""

    @pytest.fixture
    def budget_agent(self):
        agent = BudgetAllocationAgent(
            mmm_agent=AsyncMock(),
            mta_agent=AsyncMock(),
            feature_store=Mock(),
            metrics_client=Mock(),
            idempotency_cache=Mock()
        )
        return agent

    @pytest.mark.asyncio
    async def test_budget_allocation_id_dedup(self, budget_agent):
        """
        ACCEPTANCE: allocation_id dedup on retry
        METRIC: duplicate_allocations_prevented (counter)
        """
        allocation = {
            "allocation_id": "alloc_20251019_001",
            "week_start": date(2025, 10, 20),
            "total_budget": 1000000,
            "allocations": {"meta": 400000, "google": 300000, "tiktok": 200000, "youtube": 100000}
        }
        
        # First call succeeds
        result1 = await budget_agent.apply_allocation(allocation)
        assert result1['status'] == 'applied'
        
        # Second call with same allocation_id should be deduplicated
        result2 = await budget_agent.apply_allocation(allocation)
        assert result2['status'] == 'duplicate'
        assert result2['original_allocation_id'] == "alloc_20251019_001"
        
        # Metric validation
        budget_agent.metrics_client.increment.assert_called_with(
            'duplicate_allocations_prevented',
            tags=['allocation_id:alloc_20251019_001']
        )

    @pytest.mark.asyncio
    async def test_allocation_id_uniqueness_constraint(self, budget_agent):
        """Verify database uniqueness constraint on allocation_id"""
        allocation = {
            "allocation_id": "alloc_20251019_002",
            "week_start": date(2025, 10, 20),
            "total_budget": 1000000,
            "allocations": {"meta": 400000, "google": 600000}
        }
        
        # Mock database insertion
        with patch.object(budget_agent.db, 'insert', side_effect=[
            None,  # First insert succeeds
            IdempotencyError("Duplicate allocation_id")  # Second fails
        ]):
            # First call
            await budget_agent.apply_allocation(allocation)
            
            # Second call should catch IdempotencyError
            with pytest.raises(IdempotencyError):
                await budget_agent.apply_allocation(allocation)

    @pytest.mark.asyncio
    async def test_idempotency_key_24h_ttl(self, budget_agent):
        """Verify idempotency keys expire after 24 hours"""
        allocation_id = "alloc_20251019_003"
        
        # Mock Redis cache
        budget_agent.idempotency_cache.get.return_value = None
        
        # Apply allocation
        allocation = {
            "allocation_id": allocation_id,
            "week_start": date(2025, 10, 20),
            "total_budget": 1000000,
            "allocations": {"meta": 500000, "google": 500000}
        }
        await budget_agent.apply_allocation(allocation)
        
        # Verify cache set with 24h TTL
        budget_agent.idempotency_cache.setex.assert_called_once()
        args = budget_agent.idempotency_cache.setex.call_args[0]
        assert args[0] == f"allocation:{allocation_id}"
        assert args[1] == 86400  # 24 hours in seconds

    @pytest.mark.asyncio
    async def test_zero_double_spend_30d_verification(self, budget_agent):
        """
        VERIFICATION: Zero double-spend in 30-day lookback
        Queries audit log for duplicate allocation_id spends
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=30)
        
        # Query audit log for duplicates
        duplicate_count = await budget_agent.query_duplicate_allocations(
            start_date=start_date,
            end_date=end_date
        )
        
        assert duplicate_count == 0, f"Found {duplicate_count} duplicate allocations in last 30 days"

    @pytest.mark.asyncio
    async def test_concurrent_allocation_dedup(self, budget_agent):
        """Verify deduplication works under concurrent requests"""
        allocation = {
            "allocation_id": "alloc_20251019_004",
            "week_start": date(2025, 10, 20),
            "total_budget": 1000000,
            "allocations": {"meta": 400000, "google": 300000, "tiktok": 200000, "youtube": 100000}
        }
        
        # Simulate 3 concurrent requests with same allocation_id
        import asyncio
        results = await asyncio.gather(
            budget_agent.apply_allocation(allocation),
            budget_agent.apply_allocation(allocation),
            budget_agent.apply_allocation(allocation),
            return_exceptions=True
        )
        
        # Only one should succeed, others should be deduplicated
        applied_count = sum(1 for r in results if isinstance(r, dict) and r.get('status') == 'applied')
        duplicate_count = sum(1 for r in results if isinstance(r, dict) and r.get('status') == 'duplicate')
        
        assert applied_count == 1, "Exactly one allocation should succeed"
        assert duplicate_count == 2, "Two allocations should be deduplicated"


class TestBudgetSafetyIntegration:
    """Integration tests for budget safety (shift limits + idempotency)"""

    @pytest.mark.asyncio
    async def test_end_to_end_safe_allocation(self):
        """End-to-end test: shift validation + idempotency"""
        budget_agent = BudgetAllocationAgent(
            mmm_agent=AsyncMock(),
            mta_agent=AsyncMock(),
            feature_store=Mock(),
            metrics_client=Mock(),
            idempotency_cache=Mock()
        )
        
        # Current allocation
        current = {"meta": 400000, "google": 300000, "tiktok": 200000, "youtube": 100000}
        
        # Proposed allocation within ±25%
        proposed = {"meta": 480000, "google": 260000, "tiktok": 180000, "youtube": 80000}
        
        allocation = {
            "allocation_id": "alloc_20251019_e2e_001",
            "week_start": date(2025, 10, 20),
            "total_budget": 1000000,
            "allocations": proposed
        }
        
        # Validate shift
        shift_ok = await budget_agent.validate_allocation_shift(
            current=current,
            proposed=proposed,
            max_shift_pct=0.25
        )
        assert shift_ok is True
        
        # Apply allocation (first time)
        result1 = await budget_agent.apply_allocation(allocation)
        assert result1['status'] == 'applied'
        
        # Retry (should be deduplicated)
        result2 = await budget_agent.apply_allocation(allocation)
        assert result2['status'] == 'duplicate'

    @pytest.mark.asyncio
    async def test_rollback_on_shift_violation(self):
        """Verify rollback when shift limit violated"""
        budget_agent = BudgetAllocationAgent(
            mmm_agent=AsyncMock(),
            mta_agent=AsyncMock(),
            feature_store=Mock(),
            metrics_client=Mock()
        )
        
        current = {"meta": 400000, "google": 300000, "tiktok": 200000, "youtube": 100000}
        proposed = {"meta": 700000, "google": 100000, "tiktok": 150000, "youtube": 50000}  # +75% meta
        
        with pytest.raises(BudgetShiftViolation):
            await budget_agent.validate_allocation_shift(
                current=current,
                proposed=proposed,
                max_shift_pct=0.25
            )
        
        # Verify no mutation occurred
        budget_agent.feature_store.update.assert_not_called()
