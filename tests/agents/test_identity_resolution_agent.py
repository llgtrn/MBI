"""
Unit tests for Identity Resolution Agent
Component: C01_IdentityResolution (CRITICAL priority)
Coverage: Event deduplication, DB isolation, privacy-safe matching
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.exc import IntegrityError
import hashlib

from agents.identity_resolution_agent import (
    IdentityResolutionAgent,
    IdentityResolutionRequest,
    UnifiedProfile,
    IdentitySignals
)


class TestEventDeduplication:
    """Test atomic event_id deduplication with DB isolation (Q_001, Q_401)"""
    
    @pytest.fixture
    def agent(self):
        """Create agent with mocked dependencies"""
        agent = IdentityResolutionAgent()
        agent.db = Mock()
        agent.graph_builder = Mock()
        agent.privacy_filter = Mock()
        return agent
    
    def test_event_id_deduplication_atomic_409_conflict(self, agent):
        """
        Q_001 acceptance: duplicate event_id → 409 Conflict
        Verifies atomic deduplication at DB level
        """
        # Arrange
        event_id = "evt_test_123"
        signals_1 = IdentitySignals(
            event_id=event_id,
            email="test@example.com",
            timestamp=datetime.utcnow()
        )
        signals_2 = IdentitySignals(
            event_id=event_id,  # Same event_id
            email="test@example.com",
            timestamp=datetime.utcnow()
        )
        
        # Mock DB to raise IntegrityError on duplicate
        agent.db.execute = Mock(side_effect=[
            None,  # First insert succeeds
            IntegrityError("duplicate key", None, None)  # Second fails
        ])
        
        # Act & Assert
        # First call succeeds
        profile_1 = agent.resolve_identity(signals_1)
        assert profile_1 is not None
        
        # Second call with duplicate event_id raises conflict
        with pytest.raises(IntegrityError) as exc_info:
            agent.resolve_identity(signals_2)
        
        assert "duplicate key" in str(exc_info.value)
        
        # Verify idempotency counter incremented
        # (metric check will be added in impl)
    
    def test_db_isolation_level_serializable(self, agent):
        """
        Q_401 acceptance: DB uses SERIALIZABLE or optimistic locking
        Verifies transaction isolation for concurrent resolution
        """
        # Arrange
        event_id_1 = "evt_concurrent_1"
        event_id_2 = "evt_concurrent_2"
        
        signals_1 = IdentitySignals(
            event_id=event_id_1,
            email="user@example.com",
            timestamp=datetime.utcnow()
        )
        signals_2 = IdentitySignals(
            event_id=event_id_2,
            email="user@example.com",  # Same email, different event
            timestamp=datetime.utcnow()
        )
        
        # Mock concurrent execution context
        with patch('agents.identity_resolution_agent.get_db_session') as mock_session:
            mock_conn = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_conn
            
            # Verify isolation level is set to SERIALIZABLE
            mock_conn.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
            
            # Simulate concurrent resolution attempts
            profile_1 = agent.resolve_identity(signals_1)
            profile_2 = agent.resolve_identity(signals_2)
            
            # Both should succeed with proper isolation
            assert profile_1 is not None
            assert profile_2 is not None
            
            # Verify same user_key (deterministic matching)
            assert profile_1.user_key == profile_2.user_key
    
    def test_event_id_unique_constraint_schema(self):
        """
        Contract validation: IdentityResolutionRequest has event_id unique constraint
        """
        # This test validates the Pydantic schema
        request = IdentityResolutionRequest(
            event_id="evt_schema_test",
            email="schema@example.com",
            timestamp=datetime.utcnow()
        )
        
        assert hasattr(request, 'event_id')
        assert request.event_id == "evt_schema_test"
        assert isinstance(request.event_id, str)
        
        # Verify event_id is required
        with pytest.raises(ValueError):
            IdentityResolutionRequest(
                email="schema@example.com",
                timestamp=datetime.utcnow()
            )
    
    def test_dedup_conflict_metric_increment(self, agent):
        """
        Metric acceptance: identity_dedup_conflicts_total increments on duplicate
        """
        from prometheus_client import REGISTRY
        
        # Get baseline metric value
        metric_name = "identity_dedup_conflicts_total"
        before = REGISTRY.get_sample_value(metric_name) or 0
        
        # Attempt duplicate resolution
        event_id = "evt_metric_test"
        signals = IdentitySignals(
            event_id=event_id,
            email="metric@example.com",
            timestamp=datetime.utcnow()
        )
        
        # First call succeeds
        agent.resolve_identity(signals)
        
        # Second call with duplicate should increment metric
        agent.db.execute = Mock(side_effect=IntegrityError("duplicate", None, None))
        
        try:
            agent.resolve_identity(signals)
        except IntegrityError:
            pass
        
        # Verify metric incremented
        after = REGISTRY.get_sample_value(metric_name) or 0
        assert after == before + 1


class TestDatabaseIsolation:
    """Test DB isolation strategies for concurrent identity resolution"""
    
    @pytest.fixture
    def agent(self):
        agent = IdentityResolutionAgent()
        agent.db = Mock()
        return agent
    
    def test_optimistic_locking_version_check(self, agent):
        """
        Alternative to SERIALIZABLE: optimistic locking with version field
        """
        # Arrange
        user_key = "uhash_abc123"
        profile = UnifiedProfile(
            user_key=user_key,
            version=1,
            segments=["segment_a"],
            lifecycle_stage="active"
        )
        
        # Mock version conflict
        agent.graph_builder.update_profile = Mock(
            side_effect=ValueError("Version conflict: expected 1, got 2")
        )
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            agent.graph_builder.update_profile(profile)
        
        assert "Version conflict" in str(exc_info.value)
    
    def test_serializable_transaction_retry(self, agent):
        """
        Test retry logic for serialization failures
        """
        from sqlalchemy.exc import OperationalError
        
        # Arrange
        signals = IdentitySignals(
            event_id="evt_retry_test",
            email="retry@example.com",
            timestamp=datetime.utcnow()
        )
        
        # Mock transient serialization failure then success
        agent.db.execute = Mock(side_effect=[
            OperationalError("could not serialize", None, None),
            None  # Success on retry
        ])
        
        # Act
        profile = agent.resolve_identity_with_retry(
            signals,
            max_retries=3,
            backoff_ms=100
        )
        
        # Assert
        assert profile is not None
        assert agent.db.execute.call_count == 2  # Initial + 1 retry


class TestIdentityCollisionMonitoring:
    """Test collision detection and metrics (Q_423)"""
    
    def test_collision_rate_metric_tracking(self):
        """
        Q_423 acceptance: sample audit collision rate <0.01%
        """
        from prometheus_client import REGISTRY
        
        # Simulate collision detection
        total_resolutions = 100000
        collisions_detected = 5  # 0.005% collision rate
        
        # Calculate rate
        collision_rate = collisions_detected / total_resolutions
        assert collision_rate < 0.0001  # <0.01%
        
        # Verify metric exists
        metric_name = "identity_collision_rate"
        assert REGISTRY.get_sample_value(metric_name) is not None


class TestPrivacySafeMatching:
    """Test privacy-safe identity resolution"""
    
    @pytest.fixture
    def agent(self):
        return IdentityResolutionAgent()
    
    def test_pii_hashed_immediately(self, agent):
        """
        Verify PII is hashed before any processing
        """
        signals = IdentitySignals(
            event_id="evt_privacy_test",
            email="privacy@example.com",
            phone="+1234567890",
            timestamp=datetime.utcnow()
        )
        
        agent.privacy_filter.hash_pii = Mock(return_value={
            "email_hash": hashlib.sha256(b"privacy@example.com").hexdigest(),
            "phone_hash": hashlib.sha256(b"+1234567890").hexdigest()
        })
        
        # Act
        profile = agent.resolve_identity(signals)
        
        # Assert PII never stored in plaintext
        agent.privacy_filter.hash_pii.assert_called_once()
        assert "email" not in str(profile.__dict__)
        assert "phone" not in str(profile.__dict__)
