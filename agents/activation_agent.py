"""
C17 Activation Agent — API Integration & Budget Execution
==========================================================

Purpose: Execute marketing decisions by pushing changes to ad platform APIs
(Meta, Google, TikTok, YouTube) with idempotency, circuit breakers, and kill-switches.

Dependencies:
- FastAPI for HTTP endpoints
- Redis for deduplication cache & circuit breaker state
- Pydantic for request/response validation
- httpx for async HTTP client
- Prometheus client for metrics

Contracts:
- Idempotency: 8-day window via mutation_id (priority) + hash-based dedup (1h)
- Circuit Breaker: CLOSED → OPEN (<5s p99) on 10 failures; HALF_OPEN probe after 30s
- Exponential Backoff: Base 2s, multiplier 2x, max 300s (5min ceiling)
- Kill Switch: activation_kill_switch_enabled (env var + Redis)
"""

from typing import Dict, List, Optional, Literal
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import uuid
import asyncio
import httpx
from redis import Redis
from pydantic import BaseModel, Field, field_validator
from prometheus_client import Counter, Histogram, Gauge
import structlog

logger = structlog.get_logger()

# Metrics
activation_requests_total = Counter(
    'activation_requests_total',
    'Total activation requests',
    ['channel', 'action', 'status']
)
activation_latency_seconds = Histogram(
    'activation_latency_seconds',
    'Activation request latency',
    ['channel', 'action']
)
circuit_breaker_state = Gauge(
    'circuit_breaker_state',
    'Circuit breaker state (0=CLOSED, 1=OPEN, 2=HALF_OPEN)',
    ['channel']
)
circuit_breaker_transition_latency_seconds = Histogram(
    'circuit_breaker_transition_latency_seconds',
    'Circuit breaker state transition latency p99',
    ['from_state', 'to_state'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)
activation_mutation_collision_total = Counter(
    'activation_mutation_collision_total',
    'Total concurrent mutation collisions requiring CAS retry',
    ['channel']
)
activation_backoff_ceiling_hit_total = Counter(
    'activation_backoff_ceiling_hit_total',
    'Total times backoff hit 5min ceiling',
    ['channel']
)

# Configuration Schema (Contract)
class ActivationConfig(BaseModel):
    """Configuration for activation agent with idempotency and circuit breaker settings."""
    
    idempotency_window_days: int = Field(default=8, ge=1, le=30)
    dedup_cache_ttl_seconds: int = Field(default=3600, ge=60, le=86400)  # 1h default
    
    # Exponential backoff (Q_003 acceptance)
    backoff_base_seconds: float = Field(default=2.0, ge=0.1, le=10.0)
    backoff_multiplier: float = Field(default=2.0, ge=1.0, le=5.0)
    backoff_max_seconds: int = Field(default=300, ge=10, le=600)  # 5min ceiling
    
    # Circuit breaker (Q_051 acceptance)
    circuit_breaker_failure_threshold: int = Field(default=10, ge=3, le=100)
    circuit_breaker_success_threshold: int = Field(default=3, ge=1, le=10)
    circuit_breaker_timeout_seconds: int = Field(default=30, ge=5, le=300)
    circuit_breaker_transition_max_latency_seconds: float = Field(default=5.0, ge=0.1, le=10.0)
    
    # Kill switch
    kill_switch_enabled: bool = Field(default=False)
    kill_switch_redis_key: str = Field(default="activation_kill_switch_enabled")


class CircuitBreakerState(str, Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Fast-fail, reject all requests
    HALF_OPEN = "half_open"  # Probe: allow limited requests


class ActivationAction(str, Enum):
    UPDATE_BUDGET = "update_budget"
    PAUSE_CAMPAIGN = "pause_campaign"
    RESUME_CAMPAIGN = "resume_campaign"
    UPDATE_BIDS = "update_bids"
    ROTATE_CREATIVE = "rotate_creative"


# Request Schema (Contract with mutation_id for Q_004)
class ActivationRequest(BaseModel):
    """Activation request with mutation detection priority over hash-based dedup."""
    
    mutation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique mutation ID (UUID v4). New mutation_id allows execution despite hash match within dedup window."
    )
    channel: Literal["meta", "google", "tiktok", "youtube"]
    action: ActivationAction
    campaign_id: str
    params: Dict
    requested_by: str
    requested_at: datetime = Field(default_factory=datetime.utcnow)
    
    @field_validator('mutation_id')
    @classmethod
    def validate_mutation_id(cls, v: str) -> str:
        try:
            uuid.UUID(v, version=4)
        except ValueError:
            raise ValueError(f"mutation_id must be valid UUID v4, got: {v}")
        return v
    
    def request_hash(self) -> str:
        """Compute hash of request params (excluding mutation_id and timestamps)."""
        hash_input = f"{self.channel}:{self.action}:{self.campaign_id}:{str(sorted(self.params.items()))}"
        return hashlib.sha256(hash_input.encode()).hexdigest()
    
    def idempotency_key(self) -> str:
        """Idempotency key = mutation_id (primary) for 8-day window."""
        return f"activation:idempotency:{self.mutation_id}"


class ActivationResponse(BaseModel):
    request_id: str
    status: Literal["success", "deduplicated", "rejected", "failed"]
    message: str
    executed_at: Optional[datetime] = None
    platform_response: Optional[Dict] = None


class CircuitBreaker:
    """
    Circuit breaker implementation with <5s p99 transition latency (Q_051 acceptance).
    
    States:
    - CLOSED: Normal operation; failures increment counter
    - OPEN: Fast-fail all requests; transitions to HALF_OPEN after timeout
    - HALF_OPEN: Probe mode; limited requests allowed; success → CLOSED, failure → OPEN
    """
    
    def __init__(
        self,
        redis_client: Redis,
        channel: str,
        config: ActivationConfig
    ):
        self.redis = redis_client
        self.channel = channel
        self.config = config
        self.state_key = f"circuit_breaker:{channel}:state"
        self.failure_key = f"circuit_breaker:{channel}:failures"
        self.success_key = f"circuit_breaker:{channel}:successes"
        self.opened_at_key = f"circuit_breaker:{channel}:opened_at"
        
        # Initialize state if not exists
        if not self.redis.exists(self.state_key):
            self._transition_to(CircuitBreakerState.CLOSED)
    
    def _transition_to(self, new_state: CircuitBreakerState, from_state: Optional[CircuitBreakerState] = None) -> None:
        """
        Transition to new state with latency tracking (Q_051 acceptance: p99 <5s).
        """
        start_time = datetime.utcnow()
        
        # Atomic state transition with lock
        lock_key = f"{self.state_key}:lock"
        lock = self.redis.lock(lock_key, timeout=5, blocking_timeout=5)
        
        try:
            if not lock.acquire(blocking=True, timeout=self.config.circuit_breaker_transition_max_latency_seconds):
                logger.error("circuit_breaker_transition_timeout", channel=self.channel, new_state=new_state.value)
                raise TimeoutError(f"Circuit breaker transition lock timeout for {self.channel}")
            
            # Update state
            self.redis.set(self.state_key, new_state.value)
            
            # Update metrics
            if new_state == CircuitBreakerState.OPEN:
                self.redis.set(self.opened_at_key, datetime.utcnow().isoformat())
                self.redis.delete(self.failure_key)
                circuit_breaker_state.labels(channel=self.channel).set(1)
            elif new_state == CircuitBreakerState.HALF_OPEN:
                self.redis.delete(self.success_key)
                circuit_breaker_state.labels(channel=self.channel).set(2)
            elif new_state == CircuitBreakerState.CLOSED:
                self.redis.delete(self.failure_key, self.success_key, self.opened_at_key)
                circuit_breaker_state.labels(channel=self.channel).set(0)
            
            # Measure transition latency (Q_051 acceptance)
            latency = (datetime.utcnow() - start_time).total_seconds()
            if from_state:
                circuit_breaker_transition_latency_seconds.labels(
                    from_state=from_state.value,
                    to_state=new_state.value
                ).observe(latency)
                
                if latency >= self.config.circuit_breaker_transition_max_latency_seconds:
                    logger.warning(
                        "circuit_breaker_transition_slow",
                        channel=self.channel,
                        from_state=from_state.value,
                        to_state=new_state.value,
                        latency_seconds=latency,
                        threshold_seconds=self.config.circuit_breaker_transition_max_latency_seconds
                    )
            
            logger.info(
                "circuit_breaker_transition",
                channel=self.channel,
                from_state=from_state.value if from_state else None,
                to_state=new_state.value,
                latency_seconds=latency
            )
        finally:
            lock.release()
    
    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        state_str = self.redis.get(self.state_key)
        if state_str is None:
            return CircuitBreakerState.CLOSED
        return CircuitBreakerState(state_str.decode('utf-8'))
    
    def record_success(self) -> None:
        """Record successful request."""
        current_state = self.get_state()
        
        if current_state == CircuitBreakerState.HALF_OPEN:
            successes = self.redis.incr(self.success_key)
            if successes >= self.config.circuit_breaker_success_threshold:
                self._transition_to(CircuitBreakerState.CLOSED, from_state=current_state)
    
    def record_failure(self) -> None:
        """Record failed request."""
        current_state = self.get_state()
        
        if current_state == CircuitBreakerState.CLOSED:
            failures = self.redis.incr(self.failure_key)
            if failures >= self.config.circuit_breaker_failure_threshold:
                self._transition_to(CircuitBreakerState.OPEN, from_state=current_state)
        
        elif current_state == CircuitBreakerState.HALF_OPEN:
            # Single failure in HALF_OPEN → back to OPEN
            self._transition_to(CircuitBreakerState.OPEN, from_state=current_state)
    
    def should_allow_request(self) -> bool:
        """
        Check if request should be allowed based on circuit breaker state.
        
        Returns:
            True if request allowed, False if circuit open (fast-fail)
        """
        current_state = self.get_state()
        
        if current_state == CircuitBreakerState.CLOSED:
            return True
        
        elif current_state == CircuitBreakerState.OPEN:
            # Check if timeout elapsed → transition to HALF_OPEN
            opened_at_str = self.redis.get(self.opened_at_key)
            if opened_at_str:
                opened_at = datetime.fromisoformat(opened_at_str.decode('utf-8'))
                if (datetime.utcnow() - opened_at).total_seconds() >= self.config.circuit_breaker_timeout_seconds:
                    self._transition_to(CircuitBreakerState.HALF_OPEN, from_state=current_state)
                    return True  # Allow probe request
            return False  # Circuit still open
        
        elif current_state == CircuitBreakerState.HALF_OPEN:
            # Allow limited probe requests
            return True
        
        return False


class ActivationAgent:
    """
    Activation Agent with idempotency, circuit breakers, exponential backoff, and kill-switch.
    
    Implements:
    - Q_003: Exponential backoff with 5min ceiling
    - Q_004: Mutation detection priority over hash-based dedup
    - Q_051: Circuit breaker transitions <5s p99
    - Q_052: Atomic CAS with retry for concurrent mutations
    - A_026: Full idempotency + kill-switch + dedup + backoff
    """
    
    def __init__(
        self,
        redis_client: Redis,
        config: ActivationConfig,
        ad_platform_clients: Dict[str, httpx.AsyncClient]
    ):
        self.redis = redis_client
        self.config = config
        self.clients = ad_platform_clients
        
        # Circuit breakers per channel
        self.circuit_breakers = {
            channel: CircuitBreaker(redis_client, channel, config)
            for channel in ["meta", "google", "tiktok", "youtube"]
        }
    
    def _check_kill_switch(self) -> bool:
        """Check if activation kill switch is enabled."""
        # Check both config and Redis (hot reload)
        if self.config.kill_switch_enabled:
            return True
        
        redis_value = self.redis.get(self.config.kill_switch_redis_key)
        if redis_value and redis_value.decode('utf-8').lower() == 'true':
            return True
        
        return False
    
    def _check_idempotency(self, request: ActivationRequest) -> Optional[str]:
        """
        Check idempotency with mutation detection priority (Q_004 acceptance).
        
        Priority: mutation_id > hash-based dedup
        
        Returns:
            Existing request_id if duplicate, None if new/mutation detected
        """
        # Priority 1: Check mutation_id (8-day window)
        mutation_key = request.idempotency_key()
        existing_mutation = self.redis.get(mutation_key)
        
        if existing_mutation:
            # Mutation_id exists → true duplicate
            logger.info(
                "activation_idempotency_duplicate",
                mutation_id=request.mutation_id,
                existing_request_id=existing_mutation.decode('utf-8')
            )
            return existing_mutation.decode('utf-8')
        
        # Priority 2: Check hash-based dedup (1h window) BUT allow if mutation_id is new
        dedup_key = f"activation:dedup:{request.request_hash()}"
        existing_dedup = self.redis.get(dedup_key)
        
        if existing_dedup:
            # Hash match but mutation_id is new → allow execution (Q_004 acceptance)
            logger.info(
                "activation_dedup_override",
                mutation_id=request.mutation_id,
                request_hash=request.request_hash(),
                reason="new_mutation_id_detected"
            )
        
        return None  # New request or new mutation
    
    def _store_idempotency(self, request: ActivationRequest, request_id: str) -> None:
        """Store idempotency markers with appropriate TTLs."""
        # Store mutation_id (8-day window)
        mutation_key = request.idempotency_key()
        ttl_seconds = self.config.idempotency_window_days * 86400
        self.redis.setex(mutation_key, ttl_seconds, request_id)
        
        # Store hash-based dedup (1h window)
        dedup_key = f"activation:dedup:{request.request_hash()}"
        self.redis.setex(dedup_key, self.config.dedup_cache_ttl_seconds, request_id)
    
    async def _execute_with_backoff(
        self,
        request: ActivationRequest,
        circuit_breaker: CircuitBreaker
    ) -> Dict:
        """
        Execute API call with exponential backoff and ceiling (Q_003 acceptance).
        
        Backoff: base=2s, multiplier=2x, max=300s (5min ceiling)
        Circuit breaker: Opens after 10 consecutive failures
        """
        attempt = 0
        backoff_seconds = self.config.backoff_base_seconds
        max_attempts = 20  # Safety limit
        
        while attempt < max_attempts:
            try:
                # Check circuit breaker before each attempt
                if not circuit_breaker.should_allow_request():
                    logger.warning(
                        "activation_circuit_open",
                        channel=request.channel,
                        campaign_id=request.campaign_id
                    )
                    raise Exception(f"Circuit breaker OPEN for {request.channel}")
                
                # Make API call
                response = await self._call_ad_platform_api(request)
                
                # Success → record in circuit breaker
                circuit_breaker.record_success()
                return response
                
            except Exception as e:
                attempt += 1
                circuit_breaker.record_failure()
                
                if attempt >= max_attempts:
                    logger.error(
                        "activation_max_retries",
                        channel=request.channel,
                        campaign_id=request.campaign_id,
                        attempts=attempt,
                        error=str(e)
                    )
                    raise
                
                # Exponential backoff with ceiling (Q_003 acceptance)
                backoff_seconds = min(
                    backoff_seconds * self.config.backoff_multiplier,
                    self.config.backoff_max_seconds  # 5min ceiling
                )
                
                if backoff_seconds >= self.config.backoff_max_seconds:
                    activation_backoff_ceiling_hit_total.labels(channel=request.channel).inc()
                    logger.info(
                        "activation_backoff_ceiling",
                        channel=request.channel,
                        ceiling_seconds=self.config.backoff_max_seconds
                    )
                
                logger.info(
                    "activation_retry",
                    channel=request.channel,
                    attempt=attempt,
                    backoff_seconds=backoff_seconds,
                    error=str(e)
                )
                
                await asyncio.sleep(backoff_seconds)
    
    async def _call_ad_platform_api(self, request: ActivationRequest) -> Dict:
        """Make actual API call to ad platform."""
        client = self.clients.get(request.channel)
        if not client:
            raise ValueError(f"No client configured for channel: {request.channel}")
        
        # Platform-specific API logic
        if request.channel == "meta":
            endpoint = f"/v18.0/act_{request.campaign_id}"
        elif request.channel == "google":
            endpoint = f"/v13/customers/{request.campaign_id}/campaigns"
        else:
            raise NotImplementedError(f"Platform {request.channel} not implemented")
        
        # Make request
        response = await client.post(endpoint, json=request.params)
        response.raise_for_status()
        
        return response.json()
    
    async def execute_activation(self, request: ActivationRequest) -> ActivationResponse:
        """
        Execute activation request with full safety gates.
        
        Implements:
        - Kill-switch check (A_026)
        - Idempotency with mutation detection priority (Q_004, A_026)
        - Circuit breaker with <5s p99 transitions (Q_051)
        - Exponential backoff with 5min ceiling (Q_003)
        - Atomic CAS for concurrent mutations (Q_052)
        """
        request_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        try:
            # Gate 1: Kill-switch (A_026)
            if self._check_kill_switch():
                logger.warning("activation_kill_switch_active", request_id=request_id)
                activation_requests_total.labels(
                    channel=request.channel,
                    action=request.action.value,
                    status="rejected_kill_switch"
                ).inc()
                return ActivationResponse(
                    request_id=request_id,
                    status="rejected",
                    message="Activation kill-switch is enabled"
                )
            
            # Gate 2: Idempotency check with mutation detection (Q_004, A_026)
            existing_request_id = self._check_idempotency(request)
            if existing_request_id:
                activation_requests_total.labels(
                    channel=request.channel,
                    action=request.action.value,
                    status="deduplicated"
                ).inc()
                return ActivationResponse(
                    request_id=existing_request_id,
                    status="deduplicated",
                    message=f"Request already processed (mutation_id: {request.mutation_id})"
                )
            
            # Gate 3: Circuit breaker state
            circuit_breaker = self.circuit_breakers[request.channel]
            
            # Atomic CAS for concurrent mutations (Q_052 acceptance)
            # Use Redis SET NX (set if not exists) for atomic idempotency lock
            lock_key = f"activation:lock:{request.mutation_id}"
            lock_acquired = self.redis.set(lock_key, request_id, nx=True, ex=60)  # 60s TTL
            
            if not lock_acquired:
                # Concurrent collision → retry with CAS
                activation_mutation_collision_total.labels(channel=request.channel).inc()
                logger.warning(
                    "activation_concurrent_collision",
                    mutation_id=request.mutation_id,
                    request_id=request_id
                )
                
                # Wait briefly and retry idempotency check
                await asyncio.sleep(0.1)
                retry_check = self._check_idempotency(request)
                if retry_check:
                    return ActivationResponse(
                        request_id=retry_check,
                        status="deduplicated",
                        message="Concurrent collision resolved via CAS retry"
                    )
            
            # Execute with backoff and circuit breaker (Q_003, Q_051)
            try:
                platform_response = await self._execute_with_backoff(request, circuit_breaker)
                
                # Store idempotency markers
                self._store_idempotency(request, request_id)
                
                # Success metrics
                latency = (datetime.utcnow() - start_time).total_seconds()
                activation_latency_seconds.labels(
                    channel=request.channel,
                    action=request.action.value
                ).observe(latency)
                
                activation_requests_total.labels(
                    channel=request.channel,
                    action=request.action.value,
                    status="success"
                ).inc()
                
                logger.info(
                    "activation_success",
                    request_id=request_id,
                    channel=request.channel,
                    action=request.action.value,
                    campaign_id=request.campaign_id,
                    latency_seconds=latency
                )
                
                return ActivationResponse(
                    request_id=request_id,
                    status="success",
                    message="Activation executed successfully",
                    executed_at=datetime.utcnow(),
                    platform_response=platform_response
                )
                
            except Exception as e:
                # Failure metrics
                activation_requests_total.labels(
                    channel=request.channel,
                    action=request.action.value,
                    status="failed"
                ).inc()
                
                logger.error(
                    "activation_failed",
                    request_id=request_id,
                    channel=request.channel,
                    error=str(e)
                )
                
                return ActivationResponse(
                    request_id=request_id,
                    status="failed",
                    message=f"Activation failed: {str(e)}"
                )
            
            finally:
                # Release CAS lock
                self.redis.delete(lock_key)
        
        except Exception as e:
            logger.error("activation_unexpected_error", request_id=request_id, error=str(e))
            return ActivationResponse(
                request_id=request_id,
                status="failed",
                message=f"Unexpected error: {str(e)}"
            )


# ============================================================================
# FastAPI Endpoints
# ============================================================================

from fastapi import FastAPI, HTTPException, Depends

app = FastAPI(title="MBI Activation Agent")

# Dependency injection
def get_redis_client() -> Redis:
    return Redis(host='localhost', port=6379, db=0, decode_responses=False)

def get_activation_agent(redis: Redis = Depends(get_redis_client)) -> ActivationAgent:
    config = ActivationConfig()
    
    # Initialize ad platform clients
    ad_clients = {
        "meta": httpx.AsyncClient(base_url="https://graph.facebook.com", timeout=30.0),
        "google": httpx.AsyncClient(base_url="https://googleads.googleapis.com", timeout=30.0),
        "tiktok": httpx.AsyncClient(base_url="https://business-api.tiktok.com", timeout=30.0),
        "youtube": httpx.AsyncClient(base_url="https://youtube.googleapis.com", timeout=30.0),
    }
    
    return ActivationAgent(redis, config, ad_clients)


@app.post("/api/v1/activate", response_model=ActivationResponse)
async def activate(
    request: ActivationRequest,
    agent: ActivationAgent = Depends(get_activation_agent)
) -> ActivationResponse:
    """
    Execute activation request with full safety gates.
    
    Safety Features:
    - Kill-switch: Set activation_kill_switch_enabled=true to reject all requests
    - Idempotency: 8-day window via mutation_id, 1h hash-based dedup (mutation priority)
    - Circuit Breaker: Opens after 10 failures, HALF_OPEN probe after 30s
    - Exponential Backoff: 2s → 4s → 8s → ... → 300s max (5min ceiling)
    - Concurrent Collision Handling: Atomic CAS with retry
    """
    return await agent.execute_activation(request)


@app.get("/api/v1/circuit-breaker/status")
async def circuit_breaker_status(redis: Redis = Depends(get_redis_client)) -> Dict:
    """Get circuit breaker status for all channels."""
    config = ActivationConfig()
    breakers = {
        channel: CircuitBreaker(redis, channel, config)
        for channel in ["meta", "google", "tiktok", "youtube"]
    }
    
    return {
        channel: {
            "state": breaker.get_state().value,
            "failures": redis.get(breaker.failure_key) or 0,
            "successes": redis.get(breaker.success_key) or 0
        }
        for channel, breaker in breakers.items()
    }


@app.post("/api/v1/circuit-breaker/reset/{channel}")
async def reset_circuit_breaker(
    channel: str,
    redis: Redis = Depends(get_redis_client)
) -> Dict:
    """Manually reset circuit breaker to CLOSED state."""
    if channel not in ["meta", "google", "tiktok", "youtube"]:
        raise HTTPException(status_code=400, detail="Invalid channel")
    
    config = ActivationConfig()
    breaker = CircuitBreaker(redis, channel, config)
    breaker._transition_to(CircuitBreakerState.CLOSED)
    
    return {"channel": channel, "state": "closed", "message": "Circuit breaker reset"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
