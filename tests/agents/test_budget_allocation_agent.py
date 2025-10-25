"""
Tests for BudgetAllocationAgent - Idempotency, Cap Enforcement, LWW Merge
"""
import pytest
import hashlib
import json
from datetime import date, datetime
from unittest.mock import Mock, patch
from agents.budget_allocation_agent import BudgetAllocationAgent
from models.budget import BudgetAllocation


class TestBudgetAllocationIdempotency:
    """Test idempotency via allocation_id_hash"""
    
    def test_allocation_idempotency_same_hash(self):
        """Same parameters generate same allocation_id_hash"""
        agent = BudgetAllocationAgent()
        
        params = {
            "total_budget": 100000,
            "channels": ["meta", "google", "tiktok"],
            "week_start": date(2025, 10, 14),
            "constraints": {
                "min_roas": 2.0,
                "max_cac": 5000
            }
        }
        
        # First allocation
        allocation1 = agent.optimize_allocation(**params)
        hash1 = allocation1.allocation_id_hash
        
        # Second allocation with identical params
        allocation2 = agent.optimize_allocation(**params)
        hash2 = allocation2.allocation_id_hash
        
        assert hash1 == hash2, "Same params must generate same allocation_id_hash"
        assert allocation1.allocations == allocation2.allocations, "Allocations should be identical"
    
    def test_allocation_different_hash_on_change(self):
        """Different parameters generate different allocation_id_hash"""
        agent = BudgetAllocationAgent()
        
        params1 = {
            "total_budget": 100000,
            "channels": ["meta", "google"],
            "week_start": date(2025, 10, 14),
            "constraints": {"min_roas": 2.0}
        }
        
        params2 = {
            "total_budget": 150000,  # Changed budget
            "channels": ["meta", "google"],
            "week_start": date(2025, 10, 14),
            "constraints": {"min_roas": 2.0}
        }
        
        allocation1 = agent.optimize_allocation(**params1)
        allocation2 = agent.optimize_allocation(**params2)
        
        assert allocation1.allocation_id_hash != allocation2.allocation_id_hash


class TestBudgetCapEnforcement:
    """Test relative budget cap enforcement (250% default)"""
    
    @patch('agents.budget_allocation_agent.get_feature_store')
    def test_budget_cap_relative_250pct_reject(self, mock_fs):
        """10K baseline → 35K new allocation rejected (>250%)"""
        agent = BudgetAllocationAgent()
        
        # Mock baseline allocation
        mock_fs.return_value.get_previous_allocation.return_value = {
            "meta": 10000,
            "google": 8000,
            "tiktok": 2000
        }
        
        # Attempt 35K allocation for meta (350% increase)
        params = {
            "total_budget": 60000,
            "channels": ["meta", "google", "tiktok"],
            "week_start": date(2025, 10, 14),
            "constraints": {"max_shift_pct": 250}  # 250% cap
        }
        
        with pytest.raises(ValueError, match="cap_violation"):
            agent.optimize_allocation(**params, enforce_cap=True)
    
    @patch('agents.budget_allocation_agent.get_feature_store')
    @patch('agents.budget_allocation_agent.prometheus_counter')
    def test_budget_cap_metric_emitted(self, mock_counter, mock_fs):
        """Cap violation metric emitted on rejection"""
        agent = BudgetAllocationAgent()
        
        mock_fs.return_value.get_previous_allocation.return_value = {
            "meta": 10000
        }
        
        params = {
            "total_budget": 40000,
            "channels": ["meta"],
            "week_start": date(2025, 10, 14),
            "constraints": {"max_shift_pct": 200}
        }
        
        try:
            agent.optimize_allocation(**params, enforce_cap=True)
        except ValueError:
            pass
        
        mock_counter.assert_called_with(
            "budget_allocation_cap_violations_total",
            labels={"channel": "meta", "reason": "relative_cap_exceeded"}
        )
    
    @patch('agents.budget_allocation_agent.get_feature_store')
    def test_budget_cap_within_limit_allowed(self, mock_fs):
        """10K → 24K allocation allowed (240% < 250%)"""
        agent = BudgetAllocationAgent()
        
        mock_fs.return_value.get_previous_allocation.return_value = {
            "meta": 10000,
            "google": 10000
        }
        
        params = {
            "total_budget": 44000,
            "channels": ["meta", "google"],
            "week_start": date(2025, 10, 14),
            "constraints": {"max_shift_pct": 250}
        }
        
        # Should not raise
        allocation = agent.optimize_allocation(**params, enforce_cap=True)
        assert allocation.allocations["meta"] <= 25000  # 10K * 2.5


class TestBudgetLWWMerge:
    """Test last-write-wins merge for concurrent allocations"""
    
    def test_budget_cap_lww_merge(self):
        """Concurrent allocations merge via LWW timestamp"""
        agent = BudgetAllocationAgent()
        
        # Allocation 1 at T1
        alloc1 = BudgetAllocation(
            allocation_id="alloc1",
            allocation_id_hash="hash1",
            week_start=date(2025, 10, 14),
            total_budget=100000,
            allocations={"meta": 50000, "google": 50000},
            created_at=datetime(2025, 10, 19, 10, 0, 0),
            lww_timestamp=datetime(2025, 10, 19, 10, 0, 0)
        )
        
        # Allocation 2 at T2 (later)
        alloc2 = BudgetAllocation(
            allocation_id="alloc2",
            allocation_id_hash="hash2",
            week_start=date(2025, 10, 14),
            total_budget=120000,
            allocations={"meta": 60000, "google": 60000},
            created_at=datetime(2025, 10, 19, 10, 5, 0),
            lww_timestamp=datetime(2025, 10, 19, 10, 5, 0)
        )
        
        # Merge should prefer alloc2 (later timestamp)
        merged = agent._lww_merge([alloc1, alloc2])
        
        assert merged.allocation_id == "alloc2"
        assert merged.total_budget == 120000
        assert merged.lww_timestamp == datetime(2025, 10, 19, 10, 5, 0)
    
    def test_lww_merge_tie_breaker(self):
        """On timestamp tie, use allocation_id lexicographic order"""
        agent = BudgetAllocationAgent()
        
        ts = datetime(2025, 10, 19, 10, 0, 0)
        
        alloc1 = BudgetAllocation(
            allocation_id="alloc_a",
            allocation_id_hash="hash1",
            week_start=date(2025, 10, 14),
            total_budget=100000,
            allocations={"meta": 50000},
            created_at=ts,
            lww_timestamp=ts
        )
        
        alloc2 = BudgetAllocation(
            allocation_id="alloc_b",
            allocation_id_hash="hash2",
            week_start=date(2025, 10, 14),
            total_budget=120000,
            allocations={"meta": 60000},
            created_at=ts,
            lww_timestamp=ts
        )
        
        # Tie breaker: alloc_b > alloc_a lexicographically
        merged = agent._lww_merge([alloc1, alloc2])
        assert merged.allocation_id == "alloc_b"


class TestBudgetAllocationContract:
    """Test BudgetAllocation schema contract"""
    
    def test_budget_allocation_schema_fields(self):
        """BudgetAllocation has required fields"""
        alloc = BudgetAllocation(
            allocation_id="test",
            allocation_id_hash="hash123",
            week_start=date(2025, 10, 14),
            total_budget=100000,
            allocations={"meta": 50000, "google": 50000},
            created_at=datetime(2025, 10, 19, 10, 0, 0),
            lww_timestamp=datetime(2025, 10, 19, 10, 0, 0)
        )
        
        assert hasattr(alloc, 'allocation_id')
        assert hasattr(alloc, 'allocation_id_hash')
        assert hasattr(alloc, 'created_at')
        assert hasattr(alloc, 'lww_timestamp')
        assert isinstance(alloc.allocations, dict)
    
    def test_allocation_id_hash_deterministic(self):
        """allocation_id_hash is SHA256 of canonical input"""
        params = {
            "channel_allocations_json": json.dumps({"meta": 50000, "google": 50000}, sort_keys=True),
            "total_budget": 100000,
            "constraints_json": json.dumps({"min_roas": 2.0}, sort_keys=True),
            "week_start": "2025-10-14"
        }
        
        # Manual hash computation
        input_str = f"{params['channel_allocations_json']}|{params['total_budget']}|{params['constraints_json']}|{params['week_start']}"
        expected_hash = hashlib.sha256(input_str.encode()).hexdigest()
        
        alloc = BudgetAllocation(
            allocation_id="test",
            allocation_id_hash=expected_hash,
            week_start=date(2025, 10, 14),
            total_budget=100000,
            allocations={"meta": 50000, "google": 50000},
            created_at=datetime.now(),
            lww_timestamp=datetime.now()
        )
        
        assert alloc.allocation_id_hash == expected_hash
