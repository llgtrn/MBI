"""Webhook receiver with HMAC-SHA256 signature validation.

Validates webhook authenticity using HMAC signatures to prevent spoofing attacks.

Security features:
- HMAC-SHA256 signature validation
- Replay attack prevention via timestamp window (default 5min)
- Multiple webhook sources with separate secrets
- Prometheus metrics for monitoring
- Configurable header names and algorithms

Architecture:
- HMACConfig: Configuration dataclass
- WebhookReceiver: Validation logic
- WebhookValidationError: Custom exception for 401 responses

Usage:
    config = HMACConfig(
        secrets={"shopify": "secret_key"},
        replay_window_seconds=300
    )
    receiver = WebhookReceiver(hmac_config=config)
    
    # In FastAPI endpoint:
    receiver.validate_hmac(
        source="shopify",
        body=request_body,
        signature=request.headers["X-Signature-SHA256"],
        timestamp=request.headers["X-Timestamp"]
    )

References:
- Q_022: HMAC webhook validation requirement
- Q_010: Spoofing prevention
- Q_098: Security compliance
"""

import hashlib
import hmac
import time
from dataclasses import dataclass
from typing import Dict, Optional

from prometheus_client import Counter

from core.logging import get_logger


logger = get_logger(__name__)


# Prometheus metrics
hmac_validation_success = Counter(
    "hmac_validation_success_total",
    "Total HMAC validation successes",
    ["source"]
)

hmac_validation_failed = Counter(
    "hmac_validation_failed_total",
    "Total HMAC validation failures",
    ["source", "reason"]
)


class WebhookValidationError(Exception):
    """Raised when webhook validation fails (results in 401 response)."""
    
    def __init__(self, message: str, reason: str):
        super().__init__(message)
        self.reason = reason


@dataclass
class HMACConfig:
    """HMAC validation configuration.
    
    Attributes:
        algorithm: Hash algorithm (default: sha256)
        header_name: HTTP header containing HMAC signature
        timestamp_header: HTTP header containing request timestamp
        replay_window_seconds: Maximum age of timestamp before rejection (default: 300s = 5min)
        secrets: Mapping of webhook source names to secret keys
        future_tolerance_seconds: Allow timestamps slightly in future for clock skew (default: 60s)
    """
    
    algorithm: str = "sha256"
    header_name: str = "X-Signature-SHA256"
    timestamp_header: str = "X-Timestamp"
    replay_window_seconds: int = 300  # 5 minutes
    secrets: Dict[str, str] = None
    future_tolerance_seconds: int = 60  # 1 minute clock skew tolerance
    
    def __post_init__(self):
        if self.secrets is None:
            self.secrets = {}
        
        # Validate secrets are not empty
        for source, secret in self.secrets.items():
            if not secret or len(secret) < 16:
                raise ValueError(
                    f"Secret for {source} must be at least 16 characters"
                )


class WebhookReceiver:
    """Webhook receiver with HMAC signature validation.
    
    Validates incoming webhooks using HMAC-SHA256 signatures to prevent:
    - Spoofing attacks (unauthorized webhook submissions)
    - Replay attacks (resubmission of old valid requests)
    - Man-in-the-middle tampering (body modification)
    
    Acceptance criteria (Q_022):
    - Invalid HMAC returns 401
    - Valid HMAC returns 200
    - Replay >5min rejected with 401
    - Metrics counter increments on validation success/failure
    """
    
    def __init__(self, hmac_config: HMACConfig):
        """Initialize webhook receiver.
        
        Args:
            hmac_config: HMAC validation configuration
        """
        self.config = hmac_config
        self._hash_func = self._get_hash_function(hmac_config.algorithm)
        
        logger.info(
            "WebhookReceiver initialized",
            extra={
                "algorithm": hmac_config.algorithm,
                "sources": list(hmac_config.secrets.keys()),
                "replay_window_seconds": hmac_config.replay_window_seconds
            }
        )
    
    def _get_hash_function(self, algorithm: str):
        """Get hash function for algorithm."""
        hash_funcs = {
            "sha256": hashlib.sha256,
            "sha512": hashlib.sha512,
            "sha1": hashlib.sha1  # Legacy support only
        }
        
        if algorithm not in hash_funcs:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        return hash_funcs[algorithm]
    
    def validate_hmac(
        self,
        source: str,
        body: bytes,
        signature: Optional[str],
        timestamp: Optional[str]
    ) -> None:
        """Validate HMAC signature for webhook request.
        
        Args:
            source: Webhook source identifier (e.g., "shopify", "meta", "stripe")
            body: Raw request body bytes
            signature: HMAC signature from request header
            timestamp: Unix timestamp from request header
        
        Raises:
            WebhookValidationError: If validation fails (401 response)
                - Missing signature header
                - Missing timestamp header
                - Unknown source
                - Timestamp too old (replay attack)
                - Timestamp in future (clock skew attack)
                - Invalid HMAC signature (spoofing/tampering)
        
        Side effects:
            - Increments Prometheus metrics
            - Logs validation attempts
        """
        # Check signature header
        if not signature:
            hmac_validation_failed.labels(source=source, reason="missing_signature").inc()
            logger.warning(
                "Webhook validation failed: missing signature",
                extra={"source": source}
            )
            raise WebhookValidationError(
                "Missing signature header",
                reason="missing_signature"
            )
        
        # Check timestamp header
        if not timestamp:
            hmac_validation_failed.labels(source=source, reason="missing_timestamp").inc()
            logger.warning(
                "Webhook validation failed: missing timestamp",
                extra={"source": source}
            )
            raise WebhookValidationError(
                "Missing timestamp header",
                reason="missing_timestamp"
            )
        
        # Check source exists
        if source not in self.config.secrets:
            hmac_validation_failed.labels(source=source, reason="unknown_source").inc()
            logger.warning(
                "Webhook validation failed: unknown source",
                extra={"source": source}
            )
            raise WebhookValidationError(
                f"Unknown webhook source: {source}",
                reason="unknown_source"
            )
        
        # Validate timestamp (replay attack prevention)
        try:
            timestamp_int = int(timestamp)
        except ValueError:
            hmac_validation_failed.labels(source=source, reason="invalid_timestamp").inc()
            raise WebhookValidationError(
                "Invalid timestamp format",
                reason="invalid_timestamp"
            )
        
        current_time = int(time.time())
        time_diff = current_time - timestamp_int
        
        # Check if timestamp is too old (replay attack)
        if time_diff > self.config.replay_window_seconds:
            hmac_validation_failed.labels(source=source, reason="timestamp_too_old").inc()
            logger.warning(
                "Webhook validation failed: timestamp too old",
                extra={
                    "source": source,
                    "timestamp": timestamp,
                    "age_seconds": time_diff,
                    "max_age_seconds": self.config.replay_window_seconds
                }
            )
            raise WebhookValidationError(
                f"Timestamp too old: {time_diff}s > {self.config.replay_window_seconds}s",
                reason="timestamp_too_old"
            )
        
        # Check if timestamp is in future (clock skew attack)
        if time_diff < -self.config.future_tolerance_seconds:
            hmac_validation_failed.labels(source=source, reason="timestamp_in_future").inc()
            logger.warning(
                "Webhook validation failed: timestamp in future",
                extra={
                    "source": source,
                    "timestamp": timestamp,
                    "skew_seconds": abs(time_diff)
                }
            )
            raise WebhookValidationError(
                f"Timestamp in future: {abs(time_diff)}s ahead",
                reason="timestamp_in_future"
            )
        
        # Compute expected HMAC signature
        secret = self.config.secrets[source]
        expected_signature = self._compute_signature(
            secret=secret,
            body=body,
            timestamp=timestamp
        )
        
        # Constant-time comparison to prevent timing attacks
        if not hmac.compare_digest(signature, expected_signature):
            hmac_validation_failed.labels(source=source, reason="signature_mismatch").inc()
            logger.warning(
                "Webhook validation failed: signature mismatch",
                extra={
                    "source": source,
                    "timestamp": timestamp,
                    "body_size": len(body)
                }
            )
            raise WebhookValidationError(
                "HMAC validation failed: signature mismatch",
                reason="signature_mismatch"
            )
        
        # Validation successful
        hmac_validation_success.labels(source=source).inc()
        logger.info(
            "Webhook validation succeeded",
            extra={
                "source": source,
                "timestamp": timestamp,
                "body_size": len(body)
            }
        )
    
    def _compute_signature(
        self,
        secret: str,
        body: bytes,
        timestamp: str
    ) -> str:
        """Compute HMAC signature for request.
        
        Payload format: "{timestamp}.{body_utf8}"
        
        Args:
            secret: Secret key for HMAC
            body: Request body bytes
            timestamp: Unix timestamp string
        
        Returns:
            Hexadecimal HMAC signature
        """
        # Construct payload: "timestamp.body"
        payload = f"{timestamp}.{body.decode('utf-8', errors='replace')}"
        
        # Compute HMAC
        signature = hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            self._hash_func
        ).hexdigest()
        
        return signature
    
    def validate_and_parse(
        self,
        source: str,
        body: bytes,
        signature: Optional[str],
        timestamp: Optional[str]
    ) -> dict:
        """Validate HMAC and parse JSON body.
        
        Convenience method that combines validation + parsing.
        
        Args:
            source: Webhook source
            body: Raw request body
            signature: HMAC signature
            timestamp: Unix timestamp
        
        Returns:
            Parsed JSON payload
        
        Raises:
            WebhookValidationError: If validation fails
            json.JSONDecodeError: If body is not valid JSON
        """
        import json
        
        # Validate first
        self.validate_hmac(
            source=source,
            body=body,
            signature=signature,
            timestamp=timestamp
        )
        
        # Parse JSON
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as e:
            logger.error(
                "Webhook body is not valid JSON",
                extra={"source": source, "error": str(e)}
            )
            raise
        
        return payload


# FastAPI integration example
def create_webhook_endpoint(receiver: WebhookReceiver):
    """Create FastAPI webhook endpoint with HMAC validation.
    
    Example:
        from fastapi import FastAPI, Request, HTTPException
        
        app = FastAPI()
        receiver = WebhookReceiver(hmac_config)
        
        @app.post("/webhooks/{source}")
        async def receive_webhook(source: str, request: Request):
            body = await request.body()
            signature = request.headers.get("X-Signature-SHA256")
            timestamp = request.headers.get("X-Timestamp")
            
            try:
                receiver.validate_hmac(source, body, signature, timestamp)
            except WebhookValidationError as e:
                raise HTTPException(status_code=401, detail=str(e))
            
            # Process webhook payload
            payload = json.loads(body)
            await process_webhook(source, payload)
            
            return {"status": "accepted"}
    """
    from fastapi import FastAPI, Request, HTTPException
    import json
    
    app = FastAPI()
    
    @app.post("/webhooks/{source}")
    async def receive_webhook(source: str, request: Request):
        """Receive and validate webhook."""
        body = await request.body()
        signature = request.headers.get(receiver.config.header_name)
        timestamp = request.headers.get(receiver.config.timestamp_header)
        
        try:
            receiver.validate_hmac(
                source=source,
                body=body,
                signature=signature,
                timestamp=timestamp
            )
        except WebhookValidationError as e:
            logger.error(
                "Webhook validation failed",
                extra={
                    "source": source,
                    "reason": e.reason,
                    "error": str(e)
                }
            )
            raise HTTPException(status_code=401, detail=str(e))
        
        # Parse payload
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as e:
            logger.error(
                "Invalid JSON in webhook body",
                extra={"source": source, "error": str(e)}
            )
            raise HTTPException(status_code=400, detail="Invalid JSON")
        
        # Process webhook (implement based on source)
        logger.info(
            "Webhook received and validated",
            extra={
                "source": source,
                "payload_keys": list(payload.keys())
            }
        )
        
        return {"status": "accepted"}
    
    return app
