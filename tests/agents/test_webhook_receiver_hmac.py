"""HMAC-SHA256 webhook signature validation tests.

Tests webhook receiver HMAC validation for spoofing prevention.

Coverage:
- Valid HMAC-SHA256 signature acceptance (200)
- Invalid signature rejection (401)
- Missing signature header rejection (401)
- Timestamp replay attack prevention (>5min window rejects)
- Multiple webhook sources (Shopify, Meta, Stripe)
- Signature mismatch on body tampering
- Empty body handling
- Unicode/special characters in payload
- Concurrent webhook validation
- Metrics emission on validation success/failure

Acceptance:
- Invalid HMAC returns 401
- Valid HMAC returns 200
- Replay >5min rejected with 401
- Metrics counter hmac_validation_failed_total increments on failure
"""

import hashlib
import hmac
import json
import time
from datetime import datetime, timedelta
from typing import Dict

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agents.webhook_receiver import WebhookReceiver, HMACConfig, WebhookValidationError


@pytest.fixture
def hmac_config():
    """HMAC configuration for tests."""
    return HMACConfig(
        algorithm="sha256",
        header_name="X-Signature-SHA256",
        timestamp_header="X-Timestamp",
        replay_window_seconds=300,  # 5 minutes
        secrets={
            "shopify": "shopify_test_secret_key_do_not_use_in_prod",
            "meta": "meta_test_secret_key_do_not_use_in_prod",
            "stripe": "stripe_test_secret_key_do_not_use_in_prod"
        }
    )


@pytest.fixture
def webhook_receiver(hmac_config):
    """Webhook receiver instance."""
    return WebhookReceiver(hmac_config=hmac_config)


@pytest.fixture
def app(webhook_receiver):
    """FastAPI app with webhook endpoint."""
    app = FastAPI()
    
    @app.post("/webhooks/{source}")
    async def receive_webhook(source: str, request):
        body = await request.body()
        headers = dict(request.headers)
        
        # Validate HMAC
        webhook_receiver.validate_hmac(
            source=source,
            body=body,
            signature=headers.get(hmac_config.header_name),
            timestamp=headers.get(hmac_config.timestamp_header)
        )
        
        return {"status": "accepted"}
    
    return app


@pytest.fixture
def client(app):
    """Test client."""
    return TestClient(app)


def generate_hmac_signature(secret: str, body: bytes, timestamp: str = None) -> str:
    """Generate HMAC-SHA256 signature."""
    if timestamp:
        payload = f"{timestamp}.{body.decode()}"
    else:
        payload = body.decode()
    
    signature = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()
    
    return signature


def test_valid_hmac_accepts(client, hmac_config):
    """Valid HMAC signature is accepted with 200."""
    source = "shopify"
    body = json.dumps({"order_id": "12345", "total": 99.99}).encode()
    timestamp = str(int(time.time()))
    
    signature = generate_hmac_signature(
        hmac_config.secrets[source],
        body,
        timestamp
    )
    
    response = client.post(
        f"/webhooks/{source}",
        content=body,
        headers={
            hmac_config.header_name: signature,
            hmac_config.timestamp_header: timestamp
        }
    )
    
    assert response.status_code == 200
    assert response.json() == {"status": "accepted"}


def test_invalid_hmac_rejects_401(client, hmac_config):
    """Invalid HMAC signature is rejected with 401."""
    source = "shopify"
    body = json.dumps({"order_id": "12345"}).encode()
    timestamp = str(int(time.time()))
    
    # Use wrong secret to generate invalid signature
    signature = generate_hmac_signature(
        "wrong_secret_key",
        body,
        timestamp
    )
    
    response = client.post(
        f"/webhooks/{source}",
        content=body,
        headers={
            hmac_config.header_name: signature,
            hmac_config.timestamp_header: timestamp
        }
    )
    
    assert response.status_code == 401
    assert "HMAC validation failed" in response.json()["detail"]


def test_missing_signature_header_rejects_401(client, hmac_config):
    """Missing signature header is rejected with 401."""
    source = "shopify"
    body = json.dumps({"order_id": "12345"}).encode()
    timestamp = str(int(time.time()))
    
    response = client.post(
        f"/webhooks/{source}",
        content=body,
        headers={
            hmac_config.timestamp_header: timestamp
            # Missing X-Signature-SHA256
        }
    )
    
    assert response.status_code == 401
    assert "Missing signature header" in response.json()["detail"]


def test_replay_attack_prevention_rejects_old_timestamp(client, hmac_config):
    """Timestamp >5min old is rejected (replay attack prevention)."""
    source = "shopify"
    body = json.dumps({"order_id": "12345"}).encode()
    
    # Timestamp from 6 minutes ago (exceeds 5min window)
    old_timestamp = str(int(time.time()) - 360)
    
    signature = generate_hmac_signature(
        hmac_config.secrets[source],
        body,
        old_timestamp
    )
    
    response = client.post(
        f"/webhooks/{source}",
        content=body,
        headers={
            hmac_config.header_name: signature,
            hmac_config.timestamp_header: old_timestamp
        }
    )
    
    assert response.status_code == 401
    assert "Timestamp too old" in response.json()["detail"]


def test_signature_mismatch_on_body_tampering(client, hmac_config):
    """Body tampering causes signature mismatch and 401."""
    source = "meta"
    original_body = json.dumps({"event": "purchase", "amount": 100}).encode()
    timestamp = str(int(time.time()))
    
    signature = generate_hmac_signature(
        hmac_config.secrets[source],
        original_body,
        timestamp
    )
    
    # Tamper with body after signature generation
    tampered_body = json.dumps({"event": "purchase", "amount": 10000}).encode()
    
    response = client.post(
        f"/webhooks/{source}",
        content=tampered_body,
        headers={
            hmac_config.header_name: signature,
            hmac_config.timestamp_header: timestamp
        }
    )
    
    assert response.status_code == 401
    assert "HMAC validation failed" in response.json()["detail"]


def test_multiple_webhook_sources(client, hmac_config):
    """Multiple webhook sources use different secrets correctly."""
    sources = ["shopify", "meta", "stripe"]
    
    for source in sources:
        body = json.dumps({"source": source, "data": "test"}).encode()
        timestamp = str(int(time.time()))
        
        signature = generate_hmac_signature(
            hmac_config.secrets[source],
            body,
            timestamp
        )
        
        response = client.post(
            f"/webhooks/{source}",
            content=body,
            headers={
                hmac_config.header_name: signature,
                hmac_config.timestamp_header: timestamp
            }
        )
        
        assert response.status_code == 200, f"Source {source} failed"


def test_empty_body_handling(client, hmac_config):
    """Empty body is validated correctly."""
    source = "shopify"
    body = b""
    timestamp = str(int(time.time()))
    
    signature = generate_hmac_signature(
        hmac_config.secrets[source],
        body,
        timestamp
    )
    
    response = client.post(
        f"/webhooks/{source}",
        content=body,
        headers={
            hmac_config.header_name: signature,
            hmac_config.timestamp_header: timestamp
        }
    )
    
    assert response.status_code == 200


def test_unicode_special_characters_in_payload(client, hmac_config):
    """Unicode and special characters in payload validate correctly."""
    source = "shopify"
    body = json.dumps({
        "customer_name": "テスト太郎",
        "notes": "Special chars: äöü ñ € £ ¥ 中文",
        "emoji": "🎉🚀💯"
    }).encode('utf-8')
    timestamp = str(int(time.time()))
    
    signature = generate_hmac_signature(
        hmac_config.secrets[source],
        body,
        timestamp
    )
    
    response = client.post(
        f"/webhooks/{source}",
        content=body,
        headers={
            hmac_config.header_name: signature,
            hmac_config.timestamp_header: timestamp
        }
    )
    
    assert response.status_code == 200


def test_metrics_emission_on_validation(webhook_receiver, hmac_config, monkeypatch):
    """Metrics counters increment on validation success/failure."""
    from prometheus_client import REGISTRY
    
    # Get initial counter values
    initial_success = 0
    initial_failed = 0
    
    for metric in REGISTRY.collect():
        if metric.name == "hmac_validation_success_total":
            initial_success = metric.samples[0].value
        if metric.name == "hmac_validation_failed_total":
            initial_failed = metric.samples[0].value
    
    source = "shopify"
    body = b"test"
    timestamp = str(int(time.time()))
    
    # Valid signature (success)
    valid_signature = generate_hmac_signature(
        hmac_config.secrets[source],
        body,
        timestamp
    )
    
    webhook_receiver.validate_hmac(
        source=source,
        body=body,
        signature=valid_signature,
        timestamp=timestamp
    )
    
    # Invalid signature (failure)
    with pytest.raises(WebhookValidationError):
        webhook_receiver.validate_hmac(
            source=source,
            body=body,
            signature="invalid_signature",
            timestamp=timestamp
        )
    
    # Check counters incremented
    for metric in REGISTRY.collect():
        if metric.name == "hmac_validation_success_total":
            assert metric.samples[0].value == initial_success + 1
        if metric.name == "hmac_validation_failed_total":
            assert metric.samples[0].value == initial_failed + 1


def test_concurrent_webhook_validation(client, hmac_config):
    """Concurrent webhook requests validate correctly."""
    import concurrent.futures
    
    def send_webhook(source: str, idx: int):
        body = json.dumps({"id": idx}).encode()
        timestamp = str(int(time.time()))
        
        signature = generate_hmac_signature(
            hmac_config.secrets[source],
            body,
            timestamp
        )
        
        response = client.post(
            f"/webhooks/{source}",
            content=body,
            headers={
                hmac_config.header_name: signature,
                hmac_config.timestamp_header: timestamp
            }
        )
        
        return response.status_code
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(send_webhook, "shopify", i)
            for i in range(20)
        ]
        
        results = [f.result() for f in futures]
    
    assert all(status == 200 for status in results)


def test_missing_timestamp_header_rejects_401(client, hmac_config):
    """Missing timestamp header is rejected with 401."""
    source = "shopify"
    body = json.dumps({"order_id": "12345"}).encode()
    
    # Generate signature without timestamp for this test
    signature = hmac.new(
        hmac_config.secrets[source].encode(),
        body,
        hashlib.sha256
    ).hexdigest()
    
    response = client.post(
        f"/webhooks/{source}",
        content=body,
        headers={
            hmac_config.header_name: signature
            # Missing X-Timestamp
        }
    )
    
    assert response.status_code == 401
    assert "Missing timestamp header" in response.json()["detail"]


def test_future_timestamp_rejects_401(client, hmac_config):
    """Future timestamp (clock skew attack) is rejected."""
    source = "shopify"
    body = json.dumps({"order_id": "12345"}).encode()
    
    # Timestamp from 10 minutes in the future
    future_timestamp = str(int(time.time()) + 600)
    
    signature = generate_hmac_signature(
        hmac_config.secrets[source],
        body,
        future_timestamp
    )
    
    response = client.post(
        f"/webhooks/{source}",
        content=body,
        headers={
            hmac_config.header_name: signature,
            hmac_config.timestamp_header: future_timestamp
        }
    )
    
    assert response.status_code == 401
    assert "Timestamp in future" in response.json()["detail"]


def test_unknown_source_rejects_401(client, hmac_config):
    """Unknown webhook source is rejected."""
    source = "unknown_platform"
    body = json.dumps({"data": "test"}).encode()
    timestamp = str(int(time.time()))
    
    # Can't generate valid signature for unknown source
    signature = "any_signature"
    
    response = client.post(
        f"/webhooks/{source}",
        content=body,
        headers={
            hmac_config.header_name: signature,
            hmac_config.timestamp_header: timestamp
        }
    )
    
    assert response.status_code == 401
    assert "Unknown webhook source" in response.json()["detail"]
