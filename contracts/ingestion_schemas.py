"""
Ingestion Data Contracts — MBI System

Schemas for ad platform spend, web sessions, and e-commerce orders.
Enforces idempotency, HMAC verification, and data validation.

Schema Version: 1.0.0
Last Updated: 2025-10-19
"""

from pydantic import BaseModel, Field, validator, root_validator
from datetime import date, datetime
from typing import List, Optional, Dict, Any
from enum import Enum
import hashlib
import hmac
import time


class ChannelEnum(str, Enum):
    """Supported ad platforms"""
    META = "meta"
    GOOGLE = "google"
    TIKTOK = "tiktok"
    YOUTUBE = "youtube"
    LINKEDIN = "linkedin"
    TWITTER = "twitter"


class DeviceEnum(str, Enum):
    """Device types"""
    MOBILE = "mobile"
    DESKTOP = "desktop"
    TABLET = "tablet"
    UNKNOWN = "unknown"


class SpendRecord(BaseModel):
    """
    Daily ad spend by campaign/adset with HMAC signature verification.
    
    Security:
    - signature: HMAC-SHA256 of canonical payload
    - timestamp: Unix epoch for replay attack prevention
    - idempotency_key: Prevents duplicate ingestion
    """
    
    # Core fields
    date: date = Field(..., description="Spend date (YYYY-MM-DD)")
    channel: ChannelEnum = Field(..., description="Ad platform")
    campaign_id: str = Field(..., min_length=1, max_length=255)
    adset_id: Optional[str] = Field(None, max_length=255)
    
    # Metrics
    spend: float = Field(..., ge=0, description="Spend amount in currency")
    currency: str = Field(default="JPY", min_length=3, max_length=3)
    impressions: int = Field(..., ge=0)
    clicks: int = Field(..., ge=0)
    
    # Security & Idempotency (Q_002, Q_007)
    idempotency_key: str = Field(
        ..., 
        min_length=1,
        max_length=255,
        description="Unique key for deduplication (e.g., channel:campaign:date:nonce)"
    )
    signature: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="HMAC-SHA256 signature (hex) of canonical payload"
    )
    timestamp: int = Field(
        ...,
        description="Unix timestamp when record was created (for replay attack prevention)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "date": "2025-10-19",
                "channel": "meta",
                "campaign_id": "c123",
                "adset_id": "a456",
                "spend": 120000.0,
                "currency": "JPY",
                "impressions": 45000,
                "clicks": 1200,
                "idempotency_key": "meta:c123:2025-10-19:abc123",
                "signature": "a1b2c3d4e5f6" + "0" * 52,  # 64 hex chars
                "timestamp": 1697740800
            }
        }
    
    @validator('timestamp')
    def validate_timestamp_not_future(cls, v):
        """Prevent future timestamps (clock skew tolerance: 5 minutes)"""
        now = int(time.time())
        if v > now + 300:  # 5 min tolerance
            raise ValueError(f"Timestamp cannot be more than 5 minutes in future: {v} > {now}")
        return v
    
    @validator('timestamp')
    def validate_timestamp_not_too_old(cls, v):
        """Prevent replay attacks (max age: 1 hour)"""
        now = int(time.time())
        if v < now - 3600:  # 1 hour
            raise ValueError(f"Timestamp too old (>1 hour): {v} < {now}")
        return v
    
    def verify_signature(self, secret_key: bytes) -> bool:
        """
        Verify HMAC-SHA256 signature.
        
        Canonical payload format (deterministic ordering):
        "{channel}|{campaign_id}|{date}|{spend}|{currency}|{impressions}|{clicks}|{timestamp}"
        
        Args:
            secret_key: Shared secret for HMAC verification
            
        Returns:
            True if signature valid, False otherwise
        """
        canonical = (
            f"{self.channel}|{self.campaign_id}|{self.date}|"
            f"{self.spend}|{self.currency}|{self.impressions}|{self.clicks}|{self.timestamp}"
        )
        
        expected_sig = hmac.new(
            secret_key,
            canonical.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(expected_sig, self.signature)


class WebSession(BaseModel):
    """
    User session with events (GA4 style).
    
    Privacy:
    - user_key: Hashed user identifier (SHA-256)
    - No PII in session data
    """
    
    session_id: str = Field(..., min_length=1, max_length=255)
    user_key: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 hash of user identifier (no plaintext PII)"
    )
    timestamp: datetime
    
    # Attribution
    source: str = Field(..., description="utm_source or (direct)")
    medium: str = Field(..., description="utm_medium or (none)")
    campaign: Optional[str] = Field(None, max_length=255)
    
    # Context
    landing_page: str = Field(..., max_length=2048)
    device: DeviceEnum
    
    # Events
    events: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of event dicts [{type, page, ...}, ...]"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "s123",
                "user_key": "a" * 64,  # SHA-256 hash
                "timestamp": "2025-10-19T10:30:00Z",
                "source": "google",
                "medium": "cpc",
                "campaign": "holiday_2025",
                "landing_page": "/products/item1",
                "device": "mobile",
                "events": [
                    {"type": "page_view", "page": "/"},
                    {"type": "add_to_cart", "sku": "SKU-1"}
                ]
            }
        }


class Order(BaseModel):
    """
    E-commerce order with idempotency for webhook deduplication.
    
    Idempotency (Q_002):
    - idempotency_key: Prevents duplicate order processing
    - Webhook retries use same key → deduplication
    """
    
    order_id: str = Field(..., min_length=1, max_length=255)
    user_key: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 hash of user identifier"
    )
    order_date: datetime
    
    # Financial
    revenue: float = Field(..., ge=0)
    currency: str = Field(default="JPY", min_length=3, max_length=3)
    
    # Items
    items: List[Dict[str, Any]] = Field(
        ...,
        min_items=1,
        description="List of items [{sku, qty, price}, ...]"
    )
    
    # Attribution
    discount_code: Optional[str] = Field(None, max_length=100)
    utm_source: Optional[str] = Field(None, max_length=255)
    utm_medium: Optional[str] = Field(None, max_length=255)
    
    # Idempotency (Q_002)
    idempotency_key: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Unique key for webhook deduplication (e.g., order_id:timestamp)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "order_id": "o123",
                "user_key": "a" * 64,
                "order_date": "2025-10-19T11:00:00Z",
                "revenue": 19800.0,
                "currency": "JPY",
                "items": [{"sku": "SKU-1", "qty": 1, "price": 19800}],
                "discount_code": "HOLIDAY10",
                "utm_source": "google",
                "utm_medium": "cpc",
                "idempotency_key": "o123:1697740800"
            }
        }
    
    @validator('items')
    def validate_items_structure(cls, v):
        """Ensure each item has required fields"""
        required_fields = {'sku', 'qty', 'price'}
        for item in v:
            missing = required_fields - set(item.keys())
            if missing:
                raise ValueError(f"Item missing required fields: {missing}")
            if item['qty'] <= 0:
                raise ValueError(f"Item qty must be positive: {item}")
            if item['price'] < 0:
                raise ValueError(f"Item price cannot be negative: {item}")
        return v


class IngestResponse(BaseModel):
    """Standard response for ingestion endpoints"""
    
    ok: bool
    records_processed: int = Field(..., ge=0)
    duplicates_skipped: int = Field(default=0, ge=0)
    errors: List[str] = Field(default_factory=list)
    
    class Config:
        schema_extra = {
            "example": {
                "ok": True,
                "records_processed": 42,
                "duplicates_skipped": 3,
                "errors": []
            }
        }
