"""
E-commerce agent for Shopify/WooCommerce order ingestion with timezone normalization and multi-currency support.

Implements:
- Q_028: Shopify timezone normalization (normalized_utc + original_tz preservation)
- A_016: Multi-currency revenue data integrity (FX rates, historical accuracy)

Features:
- Timezone-aware order processing with UTC normalization
- Multi-currency revenue conversion using historical FX rates
- Idempotent order deduplication with timezone-aware content hashing
- Circuit breaker pattern for FX provider failures
- Comprehensive Prometheus metrics

Risk gates:
- timezone_conversion_timeout_5s
- fx_provider_circuit_breaker
- fallback_to_utc_on_invalid_timezone
- kill_switch:ENABLE_TIMEZONE_NORMALIZATION
"""

import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

import pytz
from prometheus_client import Counter, Histogram, Gauge

from core.contracts import Order, OrderTimezoneMetadata
from core.exceptions import (
    TimezoneNormalizationError,
    FXRateUnavailableError,
    CircuitBreakerOpenError
)


# Prometheus metrics
timezone_normalization_duration = Histogram(
    'mbi_timezone_normalization_duration_seconds',
    'Duration of timezone normalization operations',
    ['timezone', 'status']
)

timezone_fallback_counter = Counter(
    'mbi_timezone_fallback_total',
    'Count of timezone normalization fallbacks',
    ['reason']
)

fx_conversion_counter = Counter(
    'mbi_fx_conversion_total',
    'Count of FX conversions',
    ['from_currency', 'to_currency', 'status']
)

fx_fallback_counter = Counter(
    'mbi_fx_fallback_total',
    'Count of FX rate fallbacks',
    ['currency', 'reason']
)

duplicate_order_counter = Counter(
    'mbi_duplicate_order_total',
    'Count of duplicate orders skipped',
    ['source']
)

circuit_breaker_state = Gauge(
    'mbi_circuit_breaker_state',
    'Circuit breaker state (0=closed, 1=open)',
    ['service']
)


@dataclass
class OrderRecord:
    """Normalized order record with timezone metadata."""
    order_id: str
    order_date_utc: datetime
    revenue_original: Decimal
    currency_original: str
    revenue_usd: Optional[Decimal] = None
    fx_rate_applied: Optional[Decimal] = None
    fx_fallback_applied: bool = False
    timezone_metadata: Optional[OrderTimezoneMetadata] = None


class CircuitBreaker:
    """Circuit breaker for external service calls."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        name: str = 'default'
    ):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.name = name
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half_open
    
    def record_success(self):
        """Record successful call."""
        self.failure_count = 0
        self.state = 'closed'
        circuit_breaker_state.labels(service=self.name).set(0)
    
    def record_failure(self):
        """Record failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'open'
            circuit_breaker_state.labels(service=self.name).set(1)
    
    def can_execute(self) -> bool:
        """Check if call can be executed."""
        if self.state == 'closed':
            return True
        
        if self.state == 'open':
            # Check if timeout expired
            if time.time() - self.last_failure_time > self.timeout_seconds:
                self.state = 'half_open'
                return True
            return False
        
        # half_open state - allow single attempt
        return True


class FXRateProvider:
    """FX rate provider with caching and circuit breaker."""
    
    def __init__(
        self,
        provider: str = 'openexchangerates',
        cache_ttl_hours: int = 24,
        api_key: Optional[str] = None
    ):
        self.provider = provider
        self.cache_ttl_hours = cache_ttl_hours
        self.api_key = api_key
        self._cache: Dict[str, tuple[Decimal, float]] = {}  # (rate, timestamp)
        self.circuit_breaker = CircuitBreaker(name='fx_provider')
    
    async def get_rate(
        self,
        from_currency: str,
        to_currency: str,
        date: str
    ) -> Decimal:
        """Get FX rate with caching and circuit breaker."""
        # Check cache
        cache_key = f"{from_currency}_{to_currency}_{date}"
        if cache_key in self._cache:
            rate, cached_at = self._cache[cache_key]
            if time.time() - cached_at < self.cache_ttl_hours * 3600:
                fx_conversion_counter.labels(
                    from_currency=from_currency,
                    to_currency=to_currency,
                    status='cache_hit'
                ).inc()
                return rate
        
        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            raise CircuitBreakerOpenError(
                f"Circuit breaker open for {self.provider}",
                circuit_breaker_open=True
            )
        
        try:
            # Fetch from provider (mock implementation)
            rate = await self._fetch_rate(from_currency, to_currency, date)
            
            # Cache result
            self._cache[cache_key] = (rate, time.time())
            
            self.circuit_breaker.record_success()
            fx_conversion_counter.labels(
                from_currency=from_currency,
                to_currency=to_currency,
                status='success'
            ).inc()
            
            return rate
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            fx_conversion_counter.labels(
                from_currency=from_currency,
                to_currency=to_currency,
                status='error'
            ).inc()
            raise FXRateUnavailableError(
                f"Failed to fetch FX rate: {e}",
                circuit_breaker_open=False
            )
    
    async def _fetch_rate(
        self,
        from_currency: str,
        to_currency: str,
        date: str
    ) -> Decimal:
        """Fetch rate from provider (mock implementation)."""
        # This would call actual API in production
        # Mock rates for testing
        rates = {
            ('JPY', 'USD'): Decimal('0.0067'),
            ('EUR', 'USD'): Decimal('1.12'),
            ('GBP', 'USD'): Decimal('1.31'),
        }
        
        rate = rates.get((from_currency, to_currency))
        if rate is None:
            raise FXRateUnavailableError(f"No rate for {from_currency} to {to_currency}")
        
        return rate


class EcommerceAgent:
    """E-commerce agent with timezone normalization and multi-currency support."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.fx_provider = FXRateProvider(
            provider=config.get('fx_rate_provider', 'openexchangerates'),
            cache_ttl_hours=config.get('fx_cache_ttl_hours', 24)
        )
        self._processed_orders: set = set()  # Order ID deduplication
    
    async def normalize_order(self, shopify_order: Dict[str, Any]) -> OrderRecord:
        """
        Normalize Shopify order with timezone conversion to UTC.
        
        Args:
            shopify_order: Raw Shopify order data
            
        Returns:
            OrderRecord with UTC timestamp and timezone metadata
            
        Raises:
            TimezoneNormalizationError: If timezone conversion fails
        """
        # Kill switch check
        if not self.config.get('enable_timezone_normalization', True):
            return self._preserve_original_timezone(shopify_order)
        
        order_id = shopify_order['order_id']
        created_at_str = shopify_order['created_at']
        timezone_str = shopify_order.get('timezone', 'UTC')
        
        start_time = time.time()
        
        try:
            # Parse timestamp with timezone
            if '+' in created_at_str or created_at_str.endswith('Z'):
                # ISO format with timezone
                created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
            else:
                # No timezone - use provided timezone
                tz = pytz.timezone(timezone_str)
                created_at_naive = datetime.fromisoformat(created_at_str)
                created_at = tz.localize(created_at_naive)
            
            # Convert to UTC
            created_at_utc = created_at.astimezone(timezone.utc)
            
            # Compute UTC offset
            offset_seconds = created_at.utcoffset().total_seconds()
            offset_hours = offset_seconds / 3600
            
            # Build timezone metadata
            tz_metadata = OrderTimezoneMetadata(
                original_timezone=timezone_str,
                original_timestamp=created_at_str,
                utc_offset_hours=offset_hours,
                fallback_applied=False,
                fallback_reason=None,
                normalization_disabled=False
            )
            
            duration = time.time() - start_time
            timezone_normalization_duration.labels(
                timezone=timezone_str,
                status='success'
            ).observe(duration)
            
            # Check timeout gate
            timeout_threshold = self.config.get('timezone_conversion_timeout_s', 5)
            if duration > timeout_threshold:
                raise TimezoneNormalizationError(
                    f"Timezone conversion exceeded {timeout_threshold}s timeout",
                    timeout_exceeded=True,
                    fallback_applied=True
                )
            
            return OrderRecord(
                order_id=order_id,
                order_date_utc=created_at_utc,
                revenue_original=Decimal(shopify_order['total_price']),
                currency_original=shopify_order['currency'],
                timezone_metadata=tz_metadata
            )
            
        except pytz.exceptions.UnknownTimeZoneError:
            # Fallback to UTC
            timezone_fallback_counter.labels(reason='invalid_timezone').inc()
            
            return self._fallback_to_utc(
                shopify_order,
                reason='invalid_timezone',
                original_timezone=timezone_str
            )
        
        except Exception as e:
            timezone_normalization_duration.labels(
                timezone=timezone_str,
                status='error'
            ).observe(time.time() - start_time)
            
            if self.config.get('fallback_to_utc_on_error', True):
                timezone_fallback_counter.labels(reason='conversion_error').inc()
                return self._fallback_to_utc(
                    shopify_order,
                    reason='conversion_error',
                    original_timezone=timezone_str
                )
            else:
                raise TimezoneNormalizationError(f"Failed to normalize timezone: {e}")
    
    def _preserve_original_timezone(self, shopify_order: Dict[str, Any]) -> OrderRecord:
        """Preserve original timezone when normalization disabled."""
        created_at = datetime.fromisoformat(shopify_order['created_at'].replace('Z', '+00:00'))
        
        tz_metadata = OrderTimezoneMetadata(
            original_timezone=shopify_order.get('timezone', 'UTC'),
            original_timestamp=shopify_order['created_at'],
            utc_offset_hours=None,
            fallback_applied=False,
            fallback_reason=None,
            normalization_disabled=True
        )
        
        return OrderRecord(
            order_id=shopify_order['order_id'],
            order_date_utc=created_at,
            revenue_original=Decimal(shopify_order['total_price']),
            currency_original=shopify_order['currency'],
            timezone_metadata=tz_metadata
        )
    
    def _fallback_to_utc(
        self,
        shopify_order: Dict[str, Any],
        reason: str,
        original_timezone: str
    ) -> OrderRecord:
        """Fallback to UTC when timezone conversion fails."""
        # Parse as UTC
        created_at_str = shopify_order['created_at']
        if '+' in created_at_str or created_at_str.endswith('Z'):
            created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
        else:
            created_at = datetime.fromisoformat(created_at_str).replace(tzinfo=timezone.utc)
        
        tz_metadata = OrderTimezoneMetadata(
            original_timezone=original_timezone,
            original_timestamp=created_at_str,
            utc_offset_hours=0.0,
            fallback_applied=True,
            fallback_reason=reason,
            normalization_disabled=False
        )
        
        return OrderRecord(
            order_id=shopify_order['order_id'],
            order_date_utc=created_at,
            revenue_original=Decimal(shopify_order['total_price']),
            currency_original=shopify_order['currency'],
            timezone_metadata=tz_metadata
        )
    
    async def convert_revenue_to_usd(
        self,
        order_data: Dict[str, Any]
    ) -> OrderRecord:
        """
        Convert revenue to USD using historical FX rates.
        
        Args:
            order_data: Order with revenue and currency
            
        Returns:
            OrderRecord with revenue_usd and fx_rate_applied
        """
        revenue = Decimal(str(order_data['revenue']))
        currency = order_data['currency']
        date = order_data['date']
        
        # USD passthrough
        if currency == 'USD':
            return OrderRecord(
                order_id=order_data['order_id'],
                order_date_utc=datetime.fromisoformat(date).replace(tzinfo=timezone.utc),
                revenue_original=revenue,
                currency_original=currency,
                revenue_usd=revenue,
                fx_rate_applied=Decimal('1.0'),
                fx_fallback_applied=False
            )
        
        try:
            # Get FX rate
            fx_rate = await self.fx_provider.get_rate(currency, 'USD', date)
            
            # Convert
            revenue_usd = revenue * fx_rate
            
            return OrderRecord(
                order_id=order_data['order_id'],
                order_date_utc=datetime.fromisoformat(date).replace(tzinfo=timezone.utc),
                revenue_original=revenue,
                currency_original=currency,
                revenue_usd=revenue_usd,
                fx_rate_applied=fx_rate,
                fx_fallback_applied=False
            )
            
        except (FXRateUnavailableError, CircuitBreakerOpenError) as e:
            # Fallback to 1:1
            fx_fallback_counter.labels(
                currency=currency,
                reason='rate_unavailable'
            ).inc()
            
            return OrderRecord(
                order_id=order_data['order_id'],
                order_date_utc=datetime.fromisoformat(date).replace(tzinfo=timezone.utc),
                revenue_original=revenue,
                currency_original=currency,
                revenue_usd=revenue,  # 1:1 fallback
                fx_rate_applied=Decimal('1.0'),
                fx_fallback_applied=True
            )
    
    async def process_order(self, order_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Process order with idempotent deduplication.
        
        Args:
            order_data: Raw order data
            
        Returns:
            Processing result with status
        """
        order_id = order_data['order_id']
        
        # Check for duplicate
        if order_id in self._processed_orders:
            duplicate_order_counter.labels(source='shopify').inc()
            return {'status': 'skipped_duplicate', 'order_id': order_id}
        
        # Process order
        normalized_order = await self.normalize_order(order_data)
        
        # Mark as processed
        self._processed_orders.add(order_id)
        
        return {'status': 'processed', 'order_id': order_id}
    
    def compute_order_hash(self, order_data: Dict[str, Any]) -> str:
        """Compute content hash for order deduplication."""
        # Normalize to UTC for consistent hashing
        order_copy = order_data.copy()
        if 'created_at' in order_copy:
            created_at = datetime.fromisoformat(
                order_copy['created_at'].replace('Z', '+00:00')
            )
            order_copy['created_at_utc'] = created_at.astimezone(timezone.utc).isoformat()
            del order_copy['created_at']
        
        # Sort keys for deterministic hash
        content = json.dumps(order_copy, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def aggregate_revenue_by_utc_date(
        self,
        orders: List[Dict[str, Any]]
    ) -> Dict[str, Decimal]:
        """Aggregate revenue by UTC date to eliminate timezone skew."""
        daily_revenue = {}
        
        for order_data in orders:
            order = await self.normalize_order(order_data)
            date_key = order.order_date_utc.strftime('%Y-%m-%d')
            
            if date_key not in daily_revenue:
                daily_revenue[date_key] = Decimal('0')
            
            daily_revenue[date_key] += order.revenue_original
        
        return daily_revenue
    
    async def fetch_orders_for_mmm(
        self,
        start_date: str,
        end_date: str
    ) -> List[OrderRecord]:
        """Fetch orders for MMM with consistent UTC normalization."""
        # Mock implementation - would query database in production
        orders = []
        # ... fetch logic ...
        return orders
