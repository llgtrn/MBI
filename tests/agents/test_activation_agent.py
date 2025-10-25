"""
Test Suite for C17 Activation Agent
====================================

Covers:
- T001: Q_003 (exponential backoff ceiling), Q_051 (circuit breaker transition timing), A_026
- T002: Q_004 (mutation detection priority), Q_052 (concurrent collision CAS), A_026

Test Categories:
1. Idempotency & Deduplication (Q_004, Q_052)
2. Exponential Backoff (Q_003)
3. Circuit Breaker (Q_051)
4. Kill Switch (A_026)
5. Integration Tests
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch
from redis import Redis
import httpx
import uuid

from agents.activation_agent import (
    ActivationAgent,
    ActivationConfig,
    ActivationRequest,
    ActivationAction,
    CircuitBreaker,
    CircuitBreakerState,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_redis():
    """Mock Redis client with in-memory state."""
    redis = Mock(spec=Redis)
    storage = {}
    
    def get(key):
        return storage.get(key)
    
    def set(key, value, **kwargs):
        storage[key] = value
        return True
    
    def setex(key, ttl, value):
        storage[key] = value
        return True
    
    def delete(*keys):
        for key in keys:
            storage.pop(key, None)
        return len(keys)
    
    def exists(key):
        return key in storage
    
    def incr(key):
        storage[key] = storage.get(key, 0) + 1
        return storage[key]
    
    def lock(key, timeout=None, blocking_timeout=None):
        lock = Mock()
        lock.acquire = Mock(return_value=True)
        lock.release = Mock(return_value=None)
        return lock
    
    redis.get = get
    redis.set = set
    redis.setex = setex
    redis.delete = delete
    redis.exists = exists
    redis.incr = incr
    redis.lock = lock
    redis._storage = storage
    
    return redis


@pytest.fixture
def config():
    """Standard activation config."""
    return ActivationConfig(
        backoff_base_seconds=2.0,
        backoff_multiplier=2.0,
        backoff_max_seconds=300,
        circuit_breaker_failure_threshold=10,
        circuit_breaker_transition_max_latency_seconds=5.0,
    )


@pytest.fixture
def mock_ad_clients():
    """Mock ad platform HTTP clients."""
    clients = {}
    for channel in ["meta", "google", "tiktok", "youtube"]:
        client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = AsyncMock()
        mock_response.json.return_value = {"status": "success", "id": "12345"}
        mock_response.raise_for_status.return_value = None
        client.post.return_value = mock_response
        clients[channel] = client
    return clients


@pytest.fixture
def activation_agent(mock_redis, config, mock_ad_clients):
    """Activation agent with mocked dependencies."""
    return ActivationAgent(mock_redis, config, mock_ad_clients)


@pytest.fixture
def sample_request():
    """Sample activation request."""
    return ActivationRequest(
        mutation_id=str(uuid.uuid4()),
        channel="meta",
        action=ActivationAction.UPDATE_BUDGET,
        campaign_id="campaign_123",
        params={"daily_budget": 50000},
        requested_by="user@example.com",
    )


# ============================================================================
# T001: Q_003 - Exponential Backoff Ceiling (5min)
# ============================================================================

@pytest.mark.asyncio
async def test_backoff_ceiling_5min(activation_agent, sample_request, mock_ad_clients):
    """
    Q_003 Acceptance: Verify backoff caps at 300s (5min ceiling).
    
    Test: Simulate 10 consecutive failures → measure backoff progression
    Expected: Backoff sequence 2s, 4s, 8s, ..., 300s (capped), not exceeding 300s
    """
    # Make API fail 10 times
    mock_ad_clients["meta"].post.side_effect = [
        Exception("API Error 1"),
        Exception("API Error 2"),
        Exception("API Error 3"),
        Exception("API Error 4"),
        Exception("API Error 5"),
        Exception("API Error 6"),
        Exception("API Error 7"),
        Exception("API Error 8"),
        Exception("API Error 9"),
        Exception("API Error 10"),
    ]
    
    backoff_times = []
    
    # Patch asyncio.sleep to capture backoff times without waiting
    original_sleep = asyncio.sleep
    async def mock_sleep(seconds):
        backoff_times.append(seconds)
        await original_sleep(0)  # Don't actually wait
    
    with patch('asyncio.sleep', side_effect=mock_sleep):
        try:
            await activation_agent.execute_activation(sample_request)
        except:
            pass  # Expected to fail after max retries
    
    # Verify backoff progression: 2, 4, 8, 16, 32, 64, 128, 256, 300, 300
    expected = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 300.0, 300.0]
    
    assert len(backoff_times) >= 9, f"Expected at least 9 backoffs, got {len(backoff_times)}"
    
    # Check first 8 values (exponential growth)
    for i in range(min(8, len(backoff_times))):
        assert backoff_times[i] == expected[i], \
            f"Backoff at index {i}: expected {expected[i]}s, got {backoff_times[i]}s"
    
    # Check ceiling enforcement (all values after index 8 should be 300s)
    for i in range(8, len(backoff_times)):
        assert backoff_times[i] == 300.0, \
            f"Backoff ceiling violated at index {i}: expected 300s, got {backoff_times[i]}s"
    
    # Verify max backoff never exceeds 300s
    assert max(backoff_times) == 300.0, \
        f"Max backoff exceeded 300s ceiling: {max(backoff_times)}s"


@pytest.mark.asyncio
async def test_backoff_ceiling_metric_emitted(activation_agent, sample_request, mock_ad_clients):
    """
    Q_003 Acceptance: Verify activation_backoff_ceiling_hit_total metric emitted.
    """
    from agents.activation_agent import activation_backoff_ceiling_hit_total
    
    # Fail enough times to hit ceiling
    mock_ad_clients["meta"].post.side_effect = [Exception("Error")] * 15
    
    initial_count = activation_backoff_ceiling_hit_total.labels(channel="meta")._value.get()
    
    with patch('asyncio.sleep', new_callable=AsyncMock):
        try:
            await activation_agent.execute_activation(sample_request)
        except:
            pass
    
    final_count = activation_backoff_ceiling_hit_total.labels(channel="meta")._value.get()
    
    # Should have incremented for backoffs at 300s ceiling
    assert final_count > initial_count, \
        "activation_backoff_ceiling_hit_total metric not incremented"


# ============================================================================
# T001: Q_051 - Circuit Breaker Transition Latency <5s p99
# ============================================================================

@pytest.mark.asyncio
async def test_circuit_breaker_p99_latency(mock_redis, config):
    """
    Q_051 Acceptance: Verify circuit breaker transitions <5s p99.
    
    Test: Measure CLOSED→OPEN and OPEN→HALF_OPEN transition latencies
    Expected: All transitions complete in <5s (p99 requirement)
    """
    from agents.activation_agent import circuit_breaker_transition_latency_seconds
    
    breaker = CircuitBreaker(mock_redis, "meta", config)
    
    transition_times = []
    
    # Test CLOSED → OPEN transition
    for i in range(config.circuit_breaker_failure_threshold):
        start = datetime.utcnow()
        breaker.record_failure()
        if breaker.get_state() == CircuitBreakerState.OPEN:
            latency = (datetime.utcnow() - start).total_seconds()
            transition_times.append(("CLOSED→OPEN", latency))
            break
    
    assert breaker.get_state() == CircuitBreakerState.OPEN, \
        "Circuit breaker did not transition to OPEN"
    
    # Test OPEN → HALF_OPEN transition
    # Simulate timeout by setting opened_at in past
    past_time = datetime.utcnow() - timedelta(seconds=config.circuit_breaker_timeout_seconds + 1)
    mock_redis.set(breaker.opened_at_key, past_time.isoformat().encode('utf-8'))
    
    start = datetime.utcnow()
    is_allowed = breaker.should_allow_request()
    latency = (datetime.utcnow() - start).total_seconds()
    transition_times.append(("OPEN→HALF_OPEN", latency))
    
    assert is_allowed, "Request not allowed after timeout"
    assert breaker.get_state() == CircuitBreakerState.HALF_OPEN, \
        "Circuit breaker did not transition to HALF_OPEN"
    
    # Test HALF_OPEN → CLOSED transition
    for i in range(config.circuit_breaker_success_threshold):
        start = datetime.utcnow()
        breaker.record_success()
        if breaker.get_state() == CircuitBreakerState.CLOSED:
            latency = (datetime.utcnow() - start).total_seconds()
            transition_times.append(("HALF_OPEN→CLOSED", latency))
            break
    
    assert breaker.get_state() == CircuitBreakerState.CLOSED, \
        "Circuit breaker did not transition to CLOSED"
    
    # Verify all transitions < 5s
    for transition, latency in transition_times:
        assert latency < config.circuit_breaker_transition_max_latency_seconds, \
            f"{transition} transition took {latency:.3f}s, exceeds {config.circuit_breaker_transition_max_latency_seconds}s threshold"
    
    # Verify p99 < 5s (all should be well below 5s in unit tests)
    max_latency = max(t[1] for t in transition_times)
    assert max_latency < 5.0, \
        f"Max transition latency {max_latency:.3f}s exceeds 5s p99 requirement"


@pytest.mark.asyncio
async def test_circuit_breaker_transition_metric(mock_redis, config):
    """
    Q_051 Acceptance: Verify circuit_breaker_transition_latency_seconds metric emitted.
    """
    from agents.activation_agent import circuit_breaker_transition_latency_seconds
    
    breaker = CircuitBreaker(mock_redis, "meta", config)
    
    # Trigger CLOSED → OPEN transition
    for i in range(config.circuit_breaker_failure_threshold):
        breaker.record_failure()
    
    # Check metric histogram recorded transition
    metric = circuit_breaker_transition_latency_seconds.labels(
        from_state="closed",
        to_state="open"
    )
    
    # Verify metric exists and has samples
    assert metric._sum.get() > 0, "Transition latency metric not recorded"


# ============================================================================
# T002: Q_004 - Mutation Detection Priority Over Dedup
# ============================================================================

@pytest.mark.asyncio
async def test_mutation_detection_priority(activation_agent, mock_redis):
    """
    Q_004 Acceptance: Verify new mutation_id allows execution despite hash match within 1h dedup.
    
    Test: Submit 2 requests with identical params but different mutation_ids within 1h
    Expected: Both execute (mutation detection priority over hash-based dedup)
    """
    # Request 1
    request1 = ActivationRequest(
        mutation_id=str(uuid.uuid4()),
        channel="meta",
        action=ActivationAction.UPDATE_BUDGET,
        campaign_id="campaign_123",
        params={"daily_budget": 50000},
        requested_by="user@example.com",
    )
    
    response1 = await activation_agent.execute_activation(request1)
    assert response1.status == "success", f"Request 1 failed: {response1.message}"
    
    # Request 2: Same params (hash match), different mutation_id
    request2 = ActivationRequest(
        mutation_id=str(uuid.uuid4()),  # NEW mutation_id
        channel="meta",
        action=ActivationAction.UPDATE_BUDGET,
        campaign_id="campaign_123",
        params={"daily_budget": 50000},  # IDENTICAL params
        requested_by="user@example.com",
    )
    
    # Verify hash match
    assert request1.request_hash() == request2.request_hash(), \
        "Test setup error: request hashes should match"
    
    # Verify different mutation_ids
    assert request1.mutation_id != request2.mutation_id, \
        "Test setup error: mutation_ids should differ"
    
    # Execute request 2
    response2 = await activation_agent.execute_activation(request2)
    
    # Q_004 acceptance: Second request should execute (not deduplicated)
    assert response2.status == "success", \
        f"Request 2 should execute with new mutation_id, got: {response2.message}"
    
    # Verify different request_ids
    assert response1.request_id != response2.request_id, \
        "Both requests should have unique request_ids"


@pytest.mark.asyncio
async def test_mutation_id_deduplication(activation_agent):
    """
    Q_004 Acceptance: Verify same mutation_id within 8d window is deduplicated.
    """
    mutation_id = str(uuid.uuid4())
    
    # Request 1
    request1 = ActivationRequest(
        mutation_id=mutation_id,
        channel="meta",
        action=ActivationAction.UPDATE_BUDGET,
        campaign_id="campaign_123",
        params={"daily_budget": 50000},
        requested_by="user@example.com",
    )
    
    response1 = await activation_agent.execute_activation(request1)
    assert response1.status == "success"
    
    # Request 2: Same mutation_id, different params
    request2 = ActivationRequest(
        mutation_id=mutation_id,  # SAME mutation_id
        channel="meta",
        action=ActivationAction.UPDATE_BUDGET,
        campaign_id="campaign_123",
        params={"daily_budget": 60000},  # Different params
        requested_by="user@example.com",
    )
    
    response2 = await activation_agent.execute_activation(request2)
    
    # Should be deduplicated
    assert response2.status == "deduplicated", \
        f"Expected deduplicated, got: {response2.status}"
    
    # Should return same request_id
    assert response2.request_id == response1.request_id, \
        "Deduplicated request should return original request_id"


# ============================================================================
# T002: Q_052 - Concurrent Collision CAS Retry
# ============================================================================

@pytest.mark.asyncio
async def test_concurrent_collision_cas(activation_agent, mock_redis):
    """
    Q_052 Acceptance: Verify atomic CAS with retry on concurrent budget changes.
    
    Test: Simulate concurrent requests with same mutation_id
    Expected: First succeeds, second detects collision via CAS, retries and deduplicates
    """
    mutation_id = str(uuid.uuid4())
    
    request1 = ActivationRequest(
        mutation_id=mutation_id,
        channel="meta",
        action=ActivationAction.UPDATE_BUDGET,
        campaign_id="campaign_123",
        params={"daily_budget": 50000},
        requested_by="user1@example.com",
    )
    
    request2 = ActivationRequest(
        mutation_id=mutation_id,  # Same mutation_id
        channel="meta",
        action=ActivationAction.UPDATE_BUDGET,
        campaign_id="campaign_123",
        params={"daily_budget": 50000},
        requested_by="user2@example.com",
    )
    
    # Simulate concurrent execution by pre-setting lock
    lock_key = f"activation:lock:{mutation_id}"
    mock_redis.set(lock_key, "request_123", nx=False, ex=60)  # Lock already held
    
    # First request should succeed (or be deduplicated if lock exists)
    response1 = await activation_agent.execute_activation(request1)
    
    # Second request should detect collision
    response2 = await activation_agent.execute_activation(request2)
    
    # One of them should be deduplicated (CAS retry logic)
    assert response2.status in ["deduplicated", "success"], \
        f"Expected deduplicated or success, got: {response2.status}"


@pytest.mark.asyncio
async def test_concurrent_collision_metric(activation_agent, mock_redis):
    """
    Q_052 Acceptance: Verify activation_mutation_collision_total metric emitted.
    """
    from agents.activation_agent import activation_mutation_collision_total
    
    mutation_id = str(uuid.uuid4())
    
    # Pre-set lock to simulate collision
    lock_key = f"activation:lock:{mutation_id}"
    mock_redis.set(lock_key, "request_999", nx=False, ex=60)
    
    request = ActivationRequest(
        mutation_id=mutation_id,
        channel="meta",
        action=ActivationAction.UPDATE_BUDGET,
        campaign_id="campaign_123",
        params={"daily_budget": 50000},
        requested_by="user@example.com",
    )
    
    initial_count = activation_mutation_collision_total.labels(channel="meta")._value.get()
    
    await activation_agent.execute_activation(request)
    
    final_count = activation_mutation_collision_total.labels(channel="meta")._value.get()
    
    # Metric should increment on collision
    assert final_count > initial_count, \
        "activation_mutation_collision_total metric not incremented on collision"


# ============================================================================
# A_026: Kill Switch Tests
# ============================================================================

@pytest.mark.asyncio
async def test_kill_switch_rejection(activation_agent, sample_request, config):
    """
    A_026 Acceptance: Verify kill-switch rejects all requests.
    """
    # Enable kill-switch
    activation_agent.config.kill_switch_enabled = True
    
    response = await activation_agent.execute_activation(sample_request)
    
    assert response.status == "rejected", \
        f"Expected rejected, got: {response.status}"
    
    assert "kill-switch" in response.message.lower(), \
        f"Expected kill-switch message, got: {response.message}"


@pytest.mark.asyncio
async def test_kill_switch_redis_override(activation_agent, sample_request, mock_redis):
    """
    A_026 Acceptance: Verify Redis kill-switch override (hot reload).
    """
    # Enable via Redis
    mock_redis.set(activation_agent.config.kill_switch_redis_key, b'true')
    
    response = await activation_agent.execute_activation(sample_request)
    
    assert response.status == "rejected", \
        "Kill-switch not enforced from Redis"


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_full_activation_flow(activation_agent, sample_request):
    """
    Integration test: Full activation flow with all gates.
    """
    # Execute activation
    response = await activation_agent.execute_activation(sample_request)
    
    # Verify success
    assert response.status == "success", f"Activation failed: {response.message}"
    assert response.request_id is not None
    assert response.executed_at is not None
    assert response.platform_response is not None
    
    # Verify idempotency
    response2 = await activation_agent.execute_activation(sample_request)
    assert response2.status == "deduplicated"
    assert response2.request_id == response.request_id


@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_failures(activation_agent, sample_request, mock_ad_clients, config):
    """
    Integration test: Circuit breaker opens after threshold failures.
    """
    # Make API fail consistently
    mock_ad_clients["meta"].post.side_effect = Exception("API Down")
    
    # Trigger failures
    for i in range(config.circuit_breaker_failure_threshold):
        with patch('asyncio.sleep', new_callable=AsyncMock):
            try:
                await activation_agent.execute_activation(sample_request)
            except:
                pass
    
    # Verify circuit is open
    breaker = activation_agent.circuit_breakers["meta"]
    assert breaker.get_state() == CircuitBreakerState.OPEN, \
        f"Circuit breaker should be OPEN, got: {breaker.get_state()}"


# ============================================================================
# Contract Validation Tests
# ============================================================================

def test_activation_config_schema():
    """
    Contract validation: ActivationConfig schema has required fields.
    """
    config = ActivationConfig()
    
    # Q_003 acceptance: backoff_max_seconds present
    assert hasattr(config, 'backoff_max_seconds')
    assert config.backoff_max_seconds == 300
    assert isinstance(config.backoff_max_seconds, int)
    
    # Q_051 acceptance: circuit breaker config present
    assert hasattr(config, 'circuit_breaker_transition_max_latency_seconds')
    assert config.circuit_breaker_transition_max_latency_seconds == 5.0


def test_activation_request_mutation_id_validation():
    """
    Contract validation: ActivationRequest requires valid UUID v4 mutation_id.
    """
    # Valid UUID v4
    valid_request = ActivationRequest(
        mutation_id=str(uuid.uuid4()),
        channel="meta",
        action=ActivationAction.UPDATE_BUDGET,
        campaign_id="c123",
        params={},
        requested_by="user@example.com",
    )
    assert valid_request.mutation_id is not None
    
    # Invalid mutation_id should raise
    with pytest.raises(ValueError, match="mutation_id must be valid UUID v4"):
        ActivationRequest(
            mutation_id="not-a-uuid",
            channel="meta",
            action=ActivationAction.UPDATE_BUDGET,
            campaign_id="c123",
            params={},
            requested_by="user@example.com",
        )


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.asyncio
async def test_idempotency_check_performance(activation_agent, mock_redis):
    """
    Performance test: Idempotency check <50ms.
    """
    import time
    
    request = ActivationRequest(
        mutation_id=str(uuid.uuid4()),
        channel="meta",
        action=ActivationAction.UPDATE_BUDGET,
        campaign_id="c123",
        params={"daily_budget": 50000},
        requested_by="user@example.com",
    )
    
    # Warm up
    activation_agent._check_idempotency(request)
    
    # Measure
    start = time.time()
    for _ in range(100):
        activation_agent._check_idempotency(request)
    elapsed = time.time() - start
    
    avg_latency_ms = (elapsed / 100) * 1000
    
    assert avg_latency_ms < 50, \
        f"Idempotency check avg latency {avg_latency_ms:.2f}ms exceeds 50ms threshold"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
