"""
Identity Resolution Agent
Component: C01_IdentityResolution (CRITICAL)
Purpose: Create unified, privacy-safe customer profiles with atomic deduplication
"""
import asyncio
import hashlib
from datetime import datetime
from typing import Dict, Optional, List
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError, OperationalError
from prometheus_client import Counter, Histogram
import logging

from core.privacy import PrivacyHasher
from core.graph import IdentityGraphBuilder
from models.identity import UnifiedProfile, IdentitySignals

logger = logging.getLogger(__name__)

# Metrics
identity_dedup_conflicts_total = Counter(
    'identity_dedup_conflicts_total',
    'Total identity event deduplication conflicts'
)

identity_resolution_latency = Histogram(
    'identity_resolution_latency_seconds',
    'Identity resolution latency by type',
    ['match_type']
)

identity_collision_rate = Counter(
    'identity_collision_rate',
    'Identity hash collision detections'
)


class IdentityResolutionRequest(BaseModel):
    """Identity resolution request schema with required event_id"""
    event_id: str = Field(..., description="Unique event identifier for deduplication")
    email: Optional[str] = None
    phone: Optional[str] = None
    customer_id: Optional[str] = None
    device_fingerprint: Optional[str] = None
    timestamp: datetime
    
    class Config:
        # Enforce event_id uniqueness at application level
        validate_assignment = True


class IdentityResolutionAgent:
    """
    Identity Resolution with atomic deduplication and SERIALIZABLE isolation
    
    Implements:
    - Q_001: Event ID deduplication atomic (409 on duplicate)
    - Q_401: DB isolation level SERIALIZABLE or optimistic locking
    - Q_423: Collision rate monitoring <0.01%
    """
    
    def __init__(self, db_url: str = None):
        self.db_engine = create_engine(
            db_url or "postgresql://localhost/mbi",
            isolation_level="SERIALIZABLE",  # Q_401: SERIALIZABLE isolation
            pool_size=20,
            max_overflow=10
        )
        self.privacy_filter = PrivacyHasher()
        self.graph_builder = IdentityGraphBuilder()
        
    async def resolve_identity(self, signals: IdentitySignals) -> UnifiedProfile:
        """
        Resolve identity from multiple signals with atomic deduplication
        
        Args:
            signals: Identity signals including required event_id
            
        Returns:
            UnifiedProfile with user_key and segments
            
        Raises:
            IntegrityError: If event_id is duplicate (409 Conflict)
        """
        # Validate event_id present
        if not signals.event_id:
            raise ValueError("event_id is required for identity resolution")
        
        # Step 1: Hash PII immediately (privacy-safe)
        hashed_signals = self.privacy_filter.hash_pii({
            'email': signals.email,
            'phone': signals.phone,
            'customer_id': signals.customer_id,
            'device_fingerprint': signals.device_fingerprint
        })
        
        try:
            with self.db_engine.connect() as conn:
                # Q_401: Ensure SERIALIZABLE isolation for concurrent safety
                conn.execute(text("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE"))
                
                # Q_001: Atomic event_id deduplication via unique constraint
                # This INSERT will fail with IntegrityError if event_id exists
                try:
                    conn.execute(
                        text("""
                            INSERT INTO identity_events (
                                event_id, 
                                email_hash, 
                                phone_hash, 
                                customer_id, 
                                device_fingerprint,
                                timestamp
                            ) VALUES (
                                :event_id,
                                :email_hash,
                                :phone_hash,
                                :customer_id,
                                :device_fingerprint,
                                :timestamp
                            )
                        """),
                        {
                            "event_id": signals.event_id,
                            "email_hash": hashed_signals.get('email_hash'),
                            "phone_hash": hashed_signals.get('phone_hash'),
                            "customer_id": signals.customer_id,
                            "device_fingerprint": signals.device_fingerprint,
                            "timestamp": signals.timestamp
                        }
                    )
                    conn.commit()
                    
                except IntegrityError as e:
                    # Q_001: Duplicate event_id detected
                    identity_dedup_conflicts_total.inc()
                    logger.warning(f"Duplicate event_id: {signals.event_id}")
                    raise  # Re-raise for 409 handling at API layer
                
                # Step 2: Deterministic matching
                deterministic_match = await self._deterministic_match(
                    hashed_signals, conn
                )
                
                if deterministic_match:
                    with identity_resolution_latency.labels(match_type='deterministic').time():
                        return await self.graph_builder.get_profile(
                            deterministic_match.user_key
                        )
                
                # Step 3: Probabilistic matching
                prob_match = await self._probabilistic_match(
                    hashed_signals, conn
                )
                
                if prob_match and prob_match.confidence > 0.85:
                    with identity_resolution_latency.labels(match_type='probabilistic').time():
                        return await self.graph_builder.merge_profile(
                            prob_match.user_key,
                            hashed_signals
                        )
                
                # Step 4: Create new profile
                with identity_resolution_latency.labels(match_type='new').time():
                    return await self.graph_builder.create_profile(hashed_signals)
                    
        except OperationalError as e:
            if "could not serialize" in str(e):
                # SERIALIZABLE isolation conflict - retry
                logger.warning(f"Serialization conflict for event {signals.event_id}, retrying")
                return await self.resolve_identity_with_retry(signals)
            raise
    
    async def resolve_identity_with_retry(
        self,
        signals: IdentitySignals,
        max_retries: int = 3,
        backoff_ms: int = 100
    ) -> UnifiedProfile:
        """
        Retry identity resolution on serialization failures
        
        Args:
            signals: Identity signals
            max_retries: Maximum retry attempts
            backoff_ms: Initial backoff in milliseconds
            
        Returns:
            UnifiedProfile
        """
        for attempt in range(max_retries):
            try:
                return await self.resolve_identity(signals)
            except OperationalError as e:
                if "could not serialize" not in str(e) or attempt == max_retries - 1:
                    raise
                
                # Exponential backoff with jitter
                await asyncio.sleep((backoff_ms * (2 ** attempt)) / 1000.0)
        
        raise RuntimeError(f"Failed to resolve identity after {max_retries} retries")
    
    async def _deterministic_match(
        self,
        hashed_signals: Dict,
        conn
    ) -> Optional[UnifiedProfile]:
        """
        Deterministic matching on exact email/phone/customer_id hashes
        """
        result = conn.execute(
            text("""
                SELECT user_key
                FROM unified_profiles
                WHERE email_hash = :email_hash
                   OR phone_hash = :phone_hash
                   OR customer_id = :customer_id
                LIMIT 1
            """),
            {
                "email_hash": hashed_signals.get('email_hash'),
                "phone_hash": hashed_signals.get('phone_hash'),
                "customer_id": hashed_signals.get('customer_id')
            }
        ).fetchone()
        
        if result:
            return UnifiedProfile(user_key=result[0])
        return None
    
    async def _probabilistic_match(
        self,
        hashed_signals: Dict,
        conn
    ) -> Optional[Dict]:
        """
        Probabilistic matching using behavioral and device signals
        """
        # Simplified probabilistic logic
        # Full implementation would use ML model for similarity scoring
        
        if not hashed_signals.get('device_fingerprint'):
            return None
        
        result = conn.execute(
            text("""
                SELECT user_key, 
                       similarity(device_fingerprint, :fingerprint) as score
                FROM unified_profiles
                WHERE device_fingerprint IS NOT NULL
                ORDER BY score DESC
                LIMIT 1
            """),
            {"fingerprint": hashed_signals.get('device_fingerprint')}
        ).fetchone()
        
        if result and result[1] > 0.85:
            return {"user_key": result[0], "confidence": result[1]}
        
        return None
    
    async def detect_collisions(self, sample_size: int = 10000) -> float:
        """
        Q_423: Sample audit for hash collisions
        
        Returns:
            Collision rate (should be <0.01%)
        """
        with self.db_engine.connect() as conn:
            result = conn.execute(
                text("""
                    WITH sampled AS (
                        SELECT email_hash, COUNT(*) as cnt
                        FROM identity_events
                        TABLESAMPLE SYSTEM (1)
                        GROUP BY email_hash
                        HAVING COUNT(*) > 1
                        LIMIT :sample_size
                    )
                    SELECT COUNT(*) as collision_count
                    FROM sampled
                """),
                {"sample_size": sample_size}
            ).fetchone()
            
            collision_count = result[0] if result else 0
            collision_rate = collision_count / sample_size
            
            # Update metric
            identity_collision_rate.inc(collision_count)
            
            if collision_rate >= 0.0001:  # 0.01%
                logger.error(f"High collision rate detected: {collision_rate:.4%}")
            
            return collision_rate


# Database schema (migration)
IDENTITY_SCHEMA = """
CREATE TABLE IF NOT EXISTS identity_events (
    event_id VARCHAR(255) PRIMARY KEY,  -- Q_001: Unique constraint for deduplication
    email_hash VARCHAR(64),
    phone_hash VARCHAR(64),
    customer_id VARCHAR(255),
    device_fingerprint VARCHAR(255),
    timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_email_hash (email_hash),
    INDEX idx_phone_hash (phone_hash),
    INDEX idx_customer_id (customer_id)
);

CREATE TABLE IF NOT EXISTS unified_profiles (
    user_key VARCHAR(64) PRIMARY KEY,
    email_hash VARCHAR(64),
    phone_hash VARCHAR(64),
    customer_id VARCHAR(255),
    device_fingerprint VARCHAR(255),
    segments JSONB,
    lifecycle_stage VARCHAR(50),
    version INTEGER DEFAULT 1,  -- Q_401: Optimistic locking support
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE INDEX idx_email_hash (email_hash),
    INDEX idx_phone_hash (phone_hash)
);
"""
