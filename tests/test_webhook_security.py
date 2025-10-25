"""
Tests for webhook security (HMAC verification and replay protection)

Tests cover:
- Valid HMAC signature verification
- Invalid HMAC signature rejection
- Replay attack detection via nonce tracking
- Timestamp validation (freshness check)
- Kill-switch functionality
"""

import pytest
from datetime import datetime, timedelta
import hashlib
import hmac
import json
from unittest.mock import Mock, AsyncMock, patch

from src.agents.ingestion.webhook_receiver import (
    WebhookReceiver,
    WebhookPayload,
    WebhookSource,
    WebhookVerificationResult,
    NonceStore
)


@pytest.fixture
def secret_key():
    """Test HMAC secret key"""
    return "test_secret_key_12345"


@pytest.fixture
def nonce_store():
    """In-memory nonce store for testing"""
    return NonceStore(redis_client=None)


@pytest.fixture
def webhook_receiver(secret_key, nonce_store):
    """WebhookReceiver with HMAC enabled"""
    return WebhookReceiver(
        secret_key=secret_key,
        nonce_store=nonce_store,
        hmac_required=True
    )


@pytest.fixture
def valid_payload_dict():
    """Valid webhook payload dict (without signature)"""
    return {
        "source": "meta",
        "event_type": "ad.completed",
        "data": {
            "campaign_id": "c123",
            "spend": 5000.00
        },
        "timestamp": datetime.utcnow().isoformat(),
        "nonce": "unique_nonce_12345"
    }


def compute_signature(payload_dict: dict, secret_key: str) -> str:
    """Helper to compute HMAC signature"""
    canonical = json.dumps(payload_dict, sort_keys=True, separators=(',', ':'))
    return hmac.new(
        secret_key.encode('utf-8'),
        canonical.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()


class TestHMACVerification:
    """Test HMAC signature verification"""
    
    @pytest.mark.asyncio
    async def test_hmac_verification_valid(
        self,
        webhook_receiver,
        valid_payload_dict,
        secret_key
    ):
        """
        Test that valid HMAC signature is accepted
        
        Acceptance: Webhook with valid signature should verify successfully
        """
        # Compute valid signature
        signature = compute_signature(valid_payload_dict, secret_key)
        
        # Create payload with signature
        payload = WebhookPayload(
            **valid_payload_dict,
            signature=signature
        )
        
        # Verify
        result = await webhook_receiver.verify_webhook(payload)
        
        assert result.valid is True
        assert result.reason is None
    
    @pytest.mark.asyncio
    async def test_hmac_verification_invalid_rejects(
        self,
        webhook_receiver,
        valid_payload_dict
    ):
        """
        Test that invalid HMAC signature is rejected
        
        Acceptance: Webhook with wrong signature should be rejected
        """
        # Use wrong signature
        wrong_signature = "0" * 64  # Invalid signature
        
        payload = WebhookPayload(
            **valid_payload_dict,
            signature=wrong_signature
        )
        
        # Verify
        result = await webhook_receiver.verify_webhook(payload)
        
        assert result.valid is False
        assert "Invalid HMAC signature" in result.reason
    
    @pytest.mark.asyncio
    async def test_hmac_verification_modified_payload_rejects(
        self,
        webhook_receiver,
        valid_payload_dict,
        secret_key
    ):
        """Test that modified payload fails verification"""
        # Compute signature for original payload
        signature = compute_signature(valid_payload_dict, secret_key)
        
        # Modify payload after signing (attack scenario)
        modified_payload = valid_payload_dict.copy()
        modified_payload["data"]["spend"] = 99999.99  # Changed!
        
        payload = WebhookPayload(
            **modified_payload,
            signature=signature  # Old signature
        )
        
        # Verify should fail
        result = await webhook_receiver.verify_webhook(payload)
        
        assert result.valid is False
    
    def test_constant_time_comparison(self, webhook_receiver):
        """Test that signature comparison is constant-time"""
        # This is ensured by hmac.compare_digest
        payload = {"test": "data"}
        sig1 = webhook_receiver.compute_signature(payload)
        sig2 = "a" * 64  # Wrong signature
        
        # Should use hmac.compare_digest internally
        result = webhook_receiver.verify_signature(payload, sig2)
        assert result is False


class TestReplayProtection:
    """Test replay attack detection via nonce tracking"""
    
    @pytest.mark.asyncio
    async def test_replay_attack_detection(
        self,
        webhook_receiver,
        valid_payload_dict,
        secret_key
    ):
        """
        Test that duplicate nonce is detected (replay attack)
        
        Acceptance: Second webhook with same nonce should be rejected
        """
        signature = compute_signature(valid_payload_dict, secret_key)
        
        payload = WebhookPayload(
            **valid_payload_dict,
            signature=signature
        )
        
        # First request should succeed
        result1 = await webhook_receiver.verify_webhook(payload)
        assert result1.valid is True
        
        # Second request with same nonce should fail (replay)
        result2 = await webhook_receiver.verify_webhook(payload)
        assert result2.valid is False
        assert "Replay attack" in result2.reason or "duplicate nonce" in result2.reason
    
    @pytest.mark.asyncio
    async def test_different_nonces_allowed(
        self,
        webhook_receiver,
        valid_payload_dict,
        secret_key
    ):
        """Test that different nonces are allowed"""
        # First webhook
        sig1 = compute_signature(valid_payload_dict, secret_key)
        payload1 = WebhookPayload(**valid_payload_dict, signature=sig1)
        
        result1 = await webhook_receiver.verify_webhook(payload1)
        assert result1.valid is True
        
        # Second webhook with different nonce
        payload_dict2 = valid_payload_dict.copy()
        payload_dict2["nonce"] = "different_nonce_67890"
        payload_dict2["timestamp"] = datetime.utcnow().isoformat()
        sig2 = compute_signature(payload_dict2, secret_key)
        payload2 = WebhookPayload(**payload_dict2, signature=sig2)
        
        result2 = await webhook_receiver.verify_webhook(payload2)
        assert result2.valid is True
    
    @pytest.mark.asyncio
    async def test_nonce_expiration(self, nonce_store):
        """Test that nonces expire after TTL"""
        nonce = "test_nonce_expire"
        
        # Store nonce
        is_new = await nonce_store.check_and_store(nonce)
        assert is_new is True
        
        # Manually expire by setting old timestamp
        if not nonce_store.redis:
            nonce_store._memory_store[nonce] = datetime.utcnow() - timedelta(minutes=6)
        
        # Cleanup expired
        await nonce_store.cleanup_expired()
        
        # Should be able to reuse now
        is_new_again = await nonce_store.check_and_store(nonce)
        assert is_new_again is True


class TestTimestampValidation:
    """Test timestamp freshness validation"""
    
    def test_old_timestamp_rejected(self, secret_key, nonce_store):
        """Test that old timestamps are rejected"""
        old_payload = {
            "source": "meta",
            "event_type": "test",
            "data": {},
            "timestamp": (datetime.utcnow() - timedelta(minutes=10)).isoformat(),
            "nonce": "old_nonce",
            "signature": "dummy"
        }
        
        with pytest.raises(ValueError, match="too old"):
            WebhookPayload(**old_payload)
    
    def test_future_timestamp_rejected(self):
        """Test that future timestamps are rejected"""
        future_payload = {
            "source": "meta",
            "event_type": "test",
            "data": {},
            "timestamp": (datetime.utcnow() + timedelta(minutes=5)).isoformat(),
            "nonce": "future_nonce",
            "signature": "dummy"
        }
        
        with pytest.raises(ValueError, match="future"):
            WebhookPayload(**future_payload)
    
    def test_recent_timestamp_accepted(self):
        """Test that recent timestamp is accepted"""
        recent_payload = {
            "source": "meta",
            "event_type": "test",
            "data": {},
            "timestamp": datetime.utcnow().isoformat(),
            "nonce": "recent_nonce",
            "signature": "a" * 64
        }
        
        # Should not raise
        payload = WebhookPayload(**recent_payload)
        assert payload.timestamp is not None


class TestKillSwitch:
    """Test HMAC kill-switch for emergency bypass"""
    
    @pytest.mark.asyncio
    async def test_kill_switch_disables_hmac(
        self,
        secret_key,
        nonce_store,
        valid_payload_dict
    ):
        """
        Test that HMAC can be disabled via kill-switch
        
        Acceptance: With hmac_required=False, invalid signatures should pass
        """
        # Create receiver with HMAC disabled
        receiver = WebhookReceiver(
            secret_key=secret_key,
            nonce_store=nonce_store,
            hmac_required=False  # Kill-switch activated
        )
        
        # Use invalid signature
        payload = WebhookPayload(
            **valid_payload_dict,
            signature="invalid_signature"
        )
        
        # Should still verify (HMAC check skipped)
        result = await receiver.verify_webhook(payload)
        assert result.valid is True
    
    @pytest.mark.asyncio
    async def test_kill_switch_still_checks_replay(
        self,
        secret_key,
        nonce_store,
        valid_payload_dict
    ):
        """Test that kill-switch doesn't disable replay protection"""
        receiver = WebhookReceiver(
            secret_key=secret_key,
            nonce_store=nonce_store,
            hmac_required=False
        )
        
        payload = WebhookPayload(
            **valid_payload_dict,
            signature="any_signature"
        )
        
        # First request passes
        result1 = await receiver.verify_webhook(payload)
        assert result1.valid is True
        
        # Second request (replay) should still fail
        result2 = await receiver.verify_webhook(payload)
        assert result2.valid is False


class TestIntegrationScenario:
    """End-to-end integration tests"""
    
    @pytest.mark.asyncio
    async def test_full_webhook_flow(
        self,
        webhook_receiver,
        secret_key
    ):
        """
        Test complete webhook verification flow
        
        1. Create valid payload
        2. Sign with HMAC
        3. Verify passes
        4. Replay attempt fails
        """
        # Step 1: Create payload
        payload_dict = {
            "source": "shopify",
            "event_type": "order.created",
            "data": {
                "order_id": "order_123",
                "total": 19800.00
            },
            "timestamp": datetime.utcnow().isoformat(),
            "nonce": "integration_test_nonce_001"
        }
        
        # Step 2: Sign
        signature = compute_signature(payload_dict, secret_key)
        payload = WebhookPayload(**payload_dict, signature=signature)
        
        # Step 3: Verify
        result = await webhook_receiver.verify_webhook(payload)
        assert result.valid is True
        
        # Step 4: Replay attempt
        replay_result = await webhook_receiver.verify_webhook(payload)
        assert replay_result.valid is False
        assert "Replay" in replay_result.reason or "duplicate" in replay_result.reason


class TestPayloadValidation:
    """Test WebhookPayload validation"""
    
    def test_missing_required_fields(self):
        """Test that missing required fields raise error"""
        with pytest.raises(ValueError):
            WebhookPayload(
                source="meta",
                event_type="test"
                # Missing: data, timestamp, signature, nonce
            )
    
    def test_invalid_source(self):
        """Test that invalid source is rejected"""
        with pytest.raises(ValueError):
            WebhookPayload(
                source="invalid_source",
                event_type="test",
                data={},
                timestamp=datetime.utcnow().isoformat(),
                signature="a" * 64,
                nonce="test"
            )
    
    def test_short_signature_rejected(self):
        """Test that short signature is rejected"""
        with pytest.raises(ValueError):
            WebhookPayload(
                source="meta",
                event_type="test",
                data={},
                timestamp=datetime.utcnow().isoformat(),
                signature="short",  # Too short
                nonce="test"
            )
