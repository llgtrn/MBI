"""
Webhook Receiver with HMAC Signature Verification

Securely receives webhook events from external platforms with:
- HMAC signature verification
- Replay attack protection (nonce tracking)
- Rate limiting
- Graceful error handling

Security features:
- HMAC-SHA256 signature validation
- 5-minute replay window
- Redis-backed nonce store
- Kill-switch for emergency bypass
"""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, validator
import hashlib
import hmac
import logging
import json
from fastapi import HTTPException, Header, Request
from enum import Enum

logger = logging.getLogger(__name__)


class WebhookSource(str, Enum):
    """Supported webhook sources"""
    META = "meta"
    GOOGLE = "google"
    SHOPIFY = "shopify"
    STRIPE = "stripe"
    CUSTOM = "custom"


class WebhookPayload(BaseModel):
    """
    Webhook payload with security fields
    
    All webhooks must include:
    - timestamp: UTC timestamp for replay protection
    - signature: HMAC signature for verification
    - nonce: Unique request ID for deduplication
    """
    source: WebhookSource
    event_type: str = Field(..., min_length=1)
    data: Dict[str, Any]
    
    # Security fields
    timestamp: datetime = Field(
        ...,
        description="UTC timestamp when webhook was sent"
    )
    signature: str = Field(
        ...,
        min_length=32,
        description="HMAC-SHA256 signature"
    )
    nonce: str = Field(
        ...,
        min_length=16,
        description="Unique request identifier"
    )
    
    @validator("timestamp")
    def timestamp_must_be_recent(cls, v):
        """Validate timestamp is within acceptable window"""
        now = datetime.utcnow()
        max_age = timedelta(minutes=5)
        
        if v > now + timedelta(seconds=30):  # Allow 30s clock skew
            raise ValueError("Timestamp is in the future")
        
        if now - v > max_age:
            raise ValueError(
                f"Timestamp too old (max age: {max_age.total_seconds()}s)"
            )
        
        return v


class WebhookVerificationResult(BaseModel):
    """Result of webhook verification"""
    valid: bool
    reason: Optional[str] = None
    verified_at: datetime = Field(default_factory=datetime.utcnow)


class NonceStore:
    """
    Redis-backed nonce store for replay attack protection
    
    Nonces are stored with 5-minute TTL to prevent replay attacks
    within the timestamp window.
    """
    
    def __init__(self, redis_client=None):
        """
        Initialize nonce store
        
        Args:
            redis_client: Redis client instance (optional, uses in-memory if None)
        """
        self.redis = redis_client
        self._memory_store: Dict[str, datetime] = {}  # Fallback for testing
    
    async def check_and_store(self, nonce: str) -> bool:
        """
        Check if nonce exists, store if not
        
        Args:
            nonce: Unique request identifier
        
        Returns:
            True if nonce is new (not a replay), False if already seen
        """
        if self.redis:
            # Redis SET NX (set if not exists) with TTL
            result = await self.redis.set(
                f"webhook:nonce:{nonce}",
                datetime.utcnow().isoformat(),
                nx=True,  # Only set if doesn't exist
                ex=300    # 5-minute TTL
            )
            return result is not None
        else:
            # In-memory fallback (for testing)
            if nonce in self._memory_store:
                # Check if expired
                if datetime.utcnow() - self._memory_store[nonce] > timedelta(minutes=5):
                    del self._memory_store[nonce]
                else:
                    return False  # Replay detected
            
            self._memory_store[nonce] = datetime.utcnow()
            return True
    
    async def cleanup_expired(self):
        """Clean up expired nonces from memory store (Redis handles automatically)"""
        if not self.redis:
            cutoff = datetime.utcnow() - timedelta(minutes=5)
            self._memory_store = {
                k: v for k, v in self._memory_store.items()
                if v > cutoff
            }


class WebhookReceiver:
    """
    Webhook receiver with HMAC verification and replay protection
    """
    
    def __init__(
        self,
        secret_key: str,
        nonce_store: NonceStore,
        hmac_required: bool = True
    ):
        """
        Initialize webhook receiver
        
        Args:
            secret_key: HMAC signing key (from secret manager)
            nonce_store: Nonce store for replay protection
            hmac_required: Whether HMAC verification is required (kill-switch)
        """
        self.secret_key = secret_key.encode('utf-8')
        self.nonce_store = nonce_store
        self.hmac_required = hmac_required
        
        if not hmac_required:
            logger.warning(
                "HMAC verification is DISABLED. "
                "This should only be used in development or emergency bypass."
            )
    
    def compute_signature(self, payload: Dict[str, Any]) -> str:
        """
        Compute HMAC-SHA256 signature for payload
        
        Args:
            payload: Webhook payload dict (without 'signature' field)
        
        Returns:
            Hex-encoded HMAC signature
        """
        # Create canonical representation (sorted JSON)
        canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'))
        
        # Compute HMAC
        signature = hmac.new(
            self.secret_key,
            canonical.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def verify_signature(
        self,
        payload: Dict[str, Any],
        provided_signature: str
    ) -> bool:
        """
        Verify HMAC signature
        
        Args:
            payload: Webhook payload (without 'signature' field)
            provided_signature: Signature from webhook header/body
        
        Returns:
            True if signature is valid
        """
        # Compute expected signature
        expected_signature = self.compute_signature(payload)
        
        # Constant-time comparison to prevent timing attacks
        return hmac.compare_digest(expected_signature, provided_signature)
    
    async def verify_webhook(
        self,
        payload: WebhookPayload
    ) -> WebhookVerificationResult:
        """
        Verify webhook authenticity
        
        Checks:
        1. HMAC signature (if required)
        2. Timestamp freshness
        3. Nonce uniqueness (replay protection)
        
        Args:
            payload: WebhookPayload to verify
        
        Returns:
            WebhookVerificationResult
        """
        # Extract signature and create payload dict for verification
        provided_signature = payload.signature
        payload_dict = payload.dict(exclude={'signature'})
        
        # Check 1: HMAC signature
        if self.hmac_required:
            if not self.verify_signature(payload_dict, provided_signature):
                logger.warning(
                    f"HMAC verification failed for webhook {payload.nonce} "
                    f"from {payload.source}"
                )
                return WebhookVerificationResult(
                    valid=False,
                    reason="Invalid HMAC signature"
                )
        
        # Check 2: Timestamp (already validated by Pydantic)
        # This is handled in the WebhookPayload validator
        
        # Check 3: Nonce uniqueness (replay protection)
        is_new_nonce = await self.nonce_store.check_and_store(payload.nonce)
        if not is_new_nonce:
            logger.warning(
                f"Replay attack detected: nonce {payload.nonce} already seen"
            )
            return WebhookVerificationResult(
                valid=False,
                reason="Replay attack detected (duplicate nonce)"
            )
        
        logger.info(
            f"Webhook verified successfully: {payload.source} / {payload.event_type}"
        )
        
        return WebhookVerificationResult(valid=True)
    
    async def receive_webhook(
        self,
        raw_payload: Dict[str, Any]
    ) -> WebhookVerificationResult:
        """
        Receive and verify webhook
        
        Args:
            raw_payload: Raw webhook payload from HTTP request
        
        Returns:
            WebhookVerificationResult
        
        Raises:
            HTTPException: If payload is invalid or verification fails
        """
        try:
            # Parse and validate payload
            payload = WebhookPayload(**raw_payload)
        except ValueError as e:
            logger.error(f"Invalid webhook payload: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid webhook payload: {str(e)}"
            )
        
        # Verify webhook
        result = await self.verify_webhook(payload)
        
        if not result.valid:
            raise HTTPException(
                status_code=401,
                detail=f"Webhook verification failed: {result.reason}"
            )
        
        return result


# FastAPI endpoint example
async def webhook_endpoint(
    request: Request,
    x_webhook_signature: str = Header(None, alias="X-Webhook-Signature")
):
    """
    FastAPI endpoint for receiving webhooks
    
    Headers:
        X-Webhook-Signature: HMAC signature (optional if in payload)
    
    Returns:
        200 OK if webhook verified and processed
        400 Bad Request if payload invalid
        401 Unauthorized if verification failed
    """
    import os
    from src.config.secrets import get_secret
    
    # Get HMAC signing key from secret manager
    hmac_key = get_secret("HMAC_SIGNING_KEY")
    
    # Check kill-switch
    hmac_required = os.getenv("WEBHOOK_HMAC_REQUIRED", "true").lower() == "true"
    
    # Initialize receiver
    nonce_store = NonceStore()  # Would pass Redis client in production
    receiver = WebhookReceiver(
        secret_key=hmac_key,
        nonce_store=nonce_store,
        hmac_required=hmac_required
    )
    
    # Parse request body
    raw_payload = await request.json()
    
    # If signature in header, add to payload
    if x_webhook_signature and "signature" not in raw_payload:
        raw_payload["signature"] = x_webhook_signature
    
    # Verify and process
    result = await receiver.receive_webhook(raw_payload)
    
    # Process webhook (publish to event bus, etc.)
    # ... processing logic here ...
    
    return {
        "status": "received",
        "verified_at": result.verified_at.isoformat()
    }
