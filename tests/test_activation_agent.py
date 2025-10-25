"""
Tests for Activation Agent operation_id deduplication
Validates idempotency for Kafka retry scenarios
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
import uuid

from agents.activation_agent import ActivationAgent, ActivationRequest
from core.contracts import ActivationResponse


class TestActivationDeduplication:
    """Test suite for operation_id-based deduplication"""

    @pytest.fixture
    async def agent(self):
        """Create agent with mocked dependencies"""
        with patch('agents.activation_agent.RedisClient') as mock_redis:
            agent = ActivationAgent(
                redis_client=mock_redis.return_value,
                ad_api_client=AsyncMock(),
                enable_dedup=True
            )
            yield agent

    @pytest.mark.asyncio
    async def test_duplicate_operation_skipped(self, agent):
        """
        ACCEPTANCE: unit: test_activation_agent.py::test_duplicate_operation_skipped passes
        
        Verify that sending the same operation_id twice results in only one mutation.
        """
        operation_id = str(uuid.uuid4())
        request = ActivationRequest(
            operation_id=operation_id,
            action="update_bid",
            campaign_id="c123",
            new_bid=5.50,
            reason="pacing_adjustment"
        )

        # Mock Redis to indicate operation_id NOT seen first time
        agent.redis_client.exists.return_value = False
        agent.redis_client.setex.return_value = True

        # First execution - should process
        response1 = await agent.execute_activation(request)
        assert response1.status == "success"
        assert response1.operation_id == operation_id
        assert response1.deduplicated is False

        # Mock Redis to indicate operation_id IS seen second time
        agent.redis_client.exists.return_value = True

        # Second execution with same operation_id - should skip
        response2 = await agent.execute_activation(request)
        assert response2.status == "skipped"
        assert response2.operation_id == operation_id
        assert response2.deduplicated is True
        assert "duplicate operation" in response2.message.lower()

        # Verify ad API called only once
        assert agent.ad_api_client.update_bid.call_count == 1

    @pytest.mark.asyncio
    async def test_operation_id_stored_in_redis(self, agent):
        """
        ACCEPTANCE: unit: test_activation_agent.py::test_operation_id_stored_in_redis passes
        
        Verify operation_id is stored in Redis with TTL on first execution.
        """
        operation_id = str(uuid.uuid4())
        request = ActivationRequest(
            operation_id=operation_id,
            action="update_budget",
            campaign_id="c456",
            new_budget=100000.0,
            reason="weekly_reallocation"
        )

        # Mock Redis
        agent.redis_client.exists.return_value = False
        agent.redis_client.setex.return_value = True

        await agent.execute_activation(request)

        # Verify Redis called with correct key and TTL
        expected_key = f"activation:dedup:{operation_id}"
        expected_ttl = 86400  # 24 hours
        agent.redis_client.setex.assert_called_once()
        call_args = agent.redis_client.setex.call_args
        assert call_args[0][0] == expected_key
        assert call_args[0][1] == expected_ttl
        assert call_args[0][2] == "1"

    @pytest.mark.asyncio
    async def test_deduplication_respects_kill_switch(self, agent):
        """
        RISK GATE: ENABLE_ACTIVATION_DEDUP kill switch
        
        When disabled, operations process without deduplication checks.
        """
        # Create agent with dedup disabled
        with patch('agents.activation_agent.RedisClient') as mock_redis:
            agent_no_dedup = ActivationAgent(
                redis_client=mock_redis.return_value,
                ad_api_client=AsyncMock(),
                enable_dedup=False
            )

            operation_id = str(uuid.uuid4())
            request = ActivationRequest(
                operation_id=operation_id,
                action="update_bid",
                campaign_id="c789",
                new_bid=3.25,
                reason="test"
            )

            # Execute twice - both should process
            response1 = await agent_no_dedup.execute_activation(request)
            response2 = await agent_no_dedup.execute_activation(request)

            assert response1.status == "success"
            assert response2.status == "success"
            assert agent_no_dedup.ad_api_client.update_bid.call_count == 2

            # Redis should NOT be called
            agent_no_dedup.redis_client.exists.assert_not_called()
            agent_no_dedup.redis_client.setex.assert_not_called()

    @pytest.mark.asyncio
    async def test_different_operations_process_independently(self, agent):
        """
        Verify different operation_ids are processed independently.
        """
        agent.redis_client.exists.return_value = False
        agent.redis_client.setex.return_value = True

        request1 = ActivationRequest(
            operation_id=str(uuid.uuid4()),
            action="pause_ad",
            ad_id="ad001",
            reason="creative_fatigue"
        )

        request2 = ActivationRequest(
            operation_id=str(uuid.uuid4()),
            action="pause_ad",
            ad_id="ad002",
            reason="creative_fatigue"
        )

        response1 = await agent.execute_activation(request1)
        response2 = await agent.execute_activation(request2)

        assert response1.status == "success"
        assert response2.status == "success"
        assert response1.operation_id != response2.operation_id
        assert agent.ad_api_client.pause_ad.call_count == 2

    @pytest.mark.asyncio
    async def test_redis_failure_allows_operation(self, agent):
        """
        If Redis fails, operation should proceed (fail-open for availability).
        Log warning but don't block activation.
        """
        operation_id = str(uuid.uuid4())
        request = ActivationRequest(
            operation_id=operation_id,
            action="resume_campaign",
            campaign_id="c999",
            reason="manual_resume"
        )

        # Simulate Redis failure
        agent.redis_client.exists.side_effect = Exception("Redis connection error")

        response = await agent.execute_activation(request)

        # Should proceed despite Redis failure
        assert response.status == "success"
        assert response.redis_failure is True
        agent.ad_api_client.resume_campaign.assert_called_once()

    @pytest.mark.asyncio
    async def test_metrics_emitted_on_duplicate(self, agent):
        """
        ACCEPTANCE: metric: activation_operations_deduplicated counter increments on duplicate
        
        Verify Prometheus counter increments when duplicate detected.
        """
        from core.metrics import activation_operations_deduplicated

        operation_id = str(uuid.uuid4())
        request = ActivationRequest(
            operation_id=operation_id,
            action="update_bid",
            campaign_id="c111",
            new_bid=4.75,
            reason="test_metrics"
        )

        # First call - not duplicate
        agent.redis_client.exists.return_value = False
        agent.redis_client.setex.return_value = True
        await agent.execute_activation(request)

        initial_count = activation_operations_deduplicated._value.get()

        # Second call - duplicate
        agent.redis_client.exists.return_value = True
        await agent.execute_activation(request)

        final_count = activation_operations_deduplicated._value.get()

        # Counter should increment by 1
        assert final_count == initial_count + 1


class TestActivationRequestContract:
    """Test ActivationRequest Pydantic schema"""

    def test_operation_id_required(self):
        """
        ACCEPTANCE: contract: ActivationRequest includes operation_id (UUID)
        """
        # Valid request with operation_id
        request = ActivationRequest(
            operation_id=str(uuid.uuid4()),
            action="update_budget",
            campaign_id="c123",
            new_budget=50000.0,
            reason="test"
        )
        assert request.operation_id is not None
        assert isinstance(request.operation_id, str)

        # Missing operation_id should fail validation
        with pytest.raises(ValueError, match="operation_id"):
            ActivationRequest(
                action="update_budget",
                campaign_id="c123",
                new_budget=50000.0,
                reason="test"
            )

    def test_operation_id_uuid_format(self):
        """Verify operation_id must be valid UUID format"""
        # Valid UUID
        valid_request = ActivationRequest(
            operation_id=str(uuid.uuid4()),
            action="pause_campaign",
            campaign_id="c456",
            reason="test"
        )
        assert valid_request.operation_id is not None

        # Invalid UUID format should fail
        with pytest.raises(ValueError, match="UUID"):
            ActivationRequest(
                operation_id="not-a-uuid",
                action="pause_campaign",
                campaign_id="c456",
                reason="test"
            )

    def test_action_enum_validation(self):
        """Verify action field has allowed values"""
        allowed_actions = [
            "update_bid",
            "update_budget",
            "pause_ad",
            "resume_ad",
            "pause_campaign",
            "resume_campaign",
            "rotate_creative"
        ]

        for action in allowed_actions:
            request = ActivationRequest(
                operation_id=str(uuid.uuid4()),
                action=action,
                campaign_id="c123",
                reason="test"
            )
            assert request.action == action

        # Invalid action should fail
        with pytest.raises(ValueError, match="action"):
            ActivationRequest(
                operation_id=str(uuid.uuid4()),
                action="invalid_action",
                campaign_id="c123",
                reason="test"
            )


class TestActivationResponseContract:
    """Test ActivationResponse Pydantic schema"""

    def test_response_includes_deduplication_flag(self):
        """Verify response includes deduplicated boolean"""
        response = ActivationResponse(
            status="success",
            operation_id=str(uuid.uuid4()),
            deduplicated=False,
            message="Bid updated successfully"
        )
        assert response.deduplicated is False

        dup_response = ActivationResponse(
            status="skipped",
            operation_id=str(uuid.uuid4()),
            deduplicated=True,
            message="Duplicate operation skipped"
        )
        assert dup_response.deduplicated is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
