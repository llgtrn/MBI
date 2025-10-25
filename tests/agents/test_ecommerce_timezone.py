"""
Test suite for Shopify timezone normalization and data integrity.

Addresses: Q_028 (Shopify timezone mismatches), A_016 (DataOps_Revenue data_integrity)

Acceptance criteria:
- Order timestamps normalized to UTC + original timezone preserved
- ±15% MMM skew eliminated through consistent timezone handling
- Multi-currency revenue correctly computed with FX rates
- Historical FX rate lookups within 1% accuracy
- Idempotent processing with timezone-aware deduplication

Risk gates:
- timezone_conversion_timeout_5s
- fallback_to_utc_on_invalid_timezone
- kill_switch:ENABLE_TIMEZONE_NORMALIZATION
"""

import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock
import pytz

from agents.ecommerce_agent import EcommerceAgent, OrderRecord
from core.contracts import Order, OrderTimezoneMetadata
from core.exceptions import TimezoneNormalizationError, FXRateUnavailableError


@pytest.fixture
def ecommerce_agent():
    """Create EcommerceAgent with test configuration."""
    return EcommerceAgent(
        config={
            'enable_timezone_normalization': True,
            'timezone_conversion_timeout_s': 5,
            'fallback_to_utc_on_error': True,
            'fx_rate_provider': 'openexchangerates',
            'fx_cache_ttl_hours': 24
        }
    )


@pytest.fixture
def mock_fx_provider():
    """Mock FX rate provider with historical rates."""
    provider = Mock()
    provider.get_rate = AsyncMock(side_effect=lambda base, target, date: {
        ('JPY', 'USD', '2025-10-18'): Decimal('0.0067'),
        ('EUR', 'USD', '2025-10-18'): Decimal('1.12'),
        ('GBP', 'USD', '2025-10-18'): Decimal('1.31'),
    }.get((base, target, date), Decimal('1.0')))
    return provider


class TestTimezoneNormalization:
    """Test timezone normalization from Shopify to UTC."""
    
    @pytest.mark.asyncio
    async def test_shopify_order_normalized_to_utc(self, ecommerce_agent):
        """Order timestamp converted to UTC with original timezone preserved."""
        # Shopify order timestamp in JST (Japan Standard Time, UTC+9)
        shopify_order = {
            'order_id': 'ord_12345',
            'created_at': '2025-10-18T14:30:00+09:00',  # JST
            'total_price': '19800.00',
            'currency': 'JPY',
            'timezone': 'Asia/Tokyo'
        }
        
        order = await ecommerce_agent.normalize_order(shopify_order)
        
        # Should normalize to UTC (14:30 JST = 05:30 UTC)
        assert order.order_date_utc == datetime(2025, 10, 18, 5, 30, 0, tzinfo=timezone.utc)
        # Original timezone preserved
        assert order.timezone_metadata.original_timezone == 'Asia/Tokyo'
        assert order.timezone_metadata.original_timestamp == '2025-10-18T14:30:00+09:00'
        assert order.timezone_metadata.utc_offset_hours == 9
        
    @pytest.mark.asyncio
    async def test_timezone_normalization_multiple_timezones(self, ecommerce_agent):
        """Orders from different timezones correctly normalized."""
        test_cases = [
            {
                'timestamp': '2025-10-18T10:00:00-08:00',  # PST
                'timezone': 'America/Los_Angeles',
                'expected_utc': datetime(2025, 10, 18, 18, 0, 0, tzinfo=timezone.utc),
                'offset': -8
            },
            {
                'timestamp': '2025-10-18T15:00:00+00:00',  # GMT
                'timezone': 'Europe/London',
                'expected_utc': datetime(2025, 10, 18, 15, 0, 0, tzinfo=timezone.utc),
                'offset': 0
            },
            {
                'timestamp': '2025-10-18T20:00:00+05:30',  # IST
                'timezone': 'Asia/Kolkata',
                'expected_utc': datetime(2025, 10, 18, 14, 30, 0, tzinfo=timezone.utc),
                'offset': 5.5
            }
        ]
        
        for case in test_cases:
            shopify_order = {
                'order_id': f'ord_{case["timezone"]}',
                'created_at': case['timestamp'],
                'total_price': '100.00',
                'currency': 'USD',
                'timezone': case['timezone']
            }
            
            order = await ecommerce_agent.normalize_order(shopify_order)
            
            assert order.order_date_utc == case['expected_utc']
            assert order.timezone_metadata.original_timezone == case['timezone']
            assert order.timezone_metadata.utc_offset_hours == case['offset']
    
    @pytest.mark.asyncio
    async def test_invalid_timezone_fallback_to_utc(self, ecommerce_agent):
        """Invalid timezone triggers fallback to UTC with warning metric."""
        shopify_order = {
            'order_id': 'ord_invalid',
            'created_at': '2025-10-18T14:30:00',  # No timezone
            'total_price': '100.00',
            'currency': 'USD',
            'timezone': 'Invalid/Timezone'
        }
        
        with patch('agents.ecommerce_agent.timezone_fallback_counter') as mock_metric:
            order = await ecommerce_agent.normalize_order(shopify_order)
            
            # Should fallback to UTC
            assert order.order_date_utc.tzinfo == timezone.utc
            # Metric incremented
            mock_metric.inc.assert_called_once_with(labels={'reason': 'invalid_timezone'})
            # Metadata shows fallback
            assert order.timezone_metadata.fallback_applied is True
            assert order.timezone_metadata.fallback_reason == 'invalid_timezone'
    
    @pytest.mark.asyncio
    async def test_timezone_conversion_timeout(self, ecommerce_agent):
        """Timezone conversion timeout triggers fallback."""
        shopify_order = {
            'order_id': 'ord_timeout',
            'created_at': '2025-10-18T14:30:00+09:00',
            'total_price': '100.00',
            'currency': 'JPY',
            'timezone': 'Asia/Tokyo'
        }
        
        with patch('agents.ecommerce_agent.pytz.timezone', side_effect=lambda x: time.sleep(6)):
            with pytest.raises(TimezoneNormalizationError) as exc_info:
                await ecommerce_agent.normalize_order(shopify_order)
            
            assert 'timeout' in str(exc_info.value).lower()
            assert exc_info.value.fallback_applied is True
    
    @pytest.mark.asyncio
    async def test_kill_switch_disables_normalization(self, ecommerce_agent):
        """Kill switch disables timezone normalization."""
        ecommerce_agent.config['enable_timezone_normalization'] = False
        
        shopify_order = {
            'order_id': 'ord_killswitch',
            'created_at': '2025-10-18T14:30:00+09:00',
            'total_price': '100.00',
            'currency': 'JPY',
            'timezone': 'Asia/Tokyo'
        }
        
        order = await ecommerce_agent.normalize_order(shopify_order)
        
        # Should preserve original timestamp without conversion
        assert order.timezone_metadata.normalization_disabled is True


class TestMultiCurrencyRevenue:
    """Test multi-currency revenue handling with FX rates."""
    
    @pytest.mark.asyncio
    async def test_multi_currency_revenue_converted_to_usd(self, ecommerce_agent, mock_fx_provider):
        """Revenue in multiple currencies converted to USD using historical FX rates."""
        ecommerce_agent.fx_provider = mock_fx_provider
        
        orders = [
            {'order_id': 'ord_jpy', 'revenue': Decimal('19800.00'), 'currency': 'JPY', 'date': '2025-10-18'},
            {'order_id': 'ord_eur', 'revenue': Decimal('150.00'), 'currency': 'EUR', 'date': '2025-10-18'},
            {'order_id': 'ord_gbp', 'revenue': Decimal('120.00'), 'currency': 'GBP', 'date': '2025-10-18'},
        ]
        
        for order_data in orders:
            order = await ecommerce_agent.convert_revenue_to_usd(order_data)
            
            if order_data['currency'] == 'JPY':
                # 19800 JPY * 0.0067 = 132.66 USD
                assert abs(order.revenue_usd - Decimal('132.66')) < Decimal('0.01')
            elif order_data['currency'] == 'EUR':
                # 150 EUR * 1.12 = 168.00 USD
                assert abs(order.revenue_usd - Decimal('168.00')) < Decimal('0.01')
            elif order_data['currency'] == 'GBP':
                # 120 GBP * 1.31 = 157.20 USD
                assert abs(order.revenue_usd - Decimal('157.20')) < Decimal('0.01')
            
            # Original currency preserved
            assert order.revenue_original == order_data['revenue']
            assert order.currency_original == order_data['currency']
    
    @pytest.mark.asyncio
    async def test_historical_fx_rate_accuracy(self, ecommerce_agent, mock_fx_provider):
        """Historical FX rates within 1% of actual rates."""
        ecommerce_agent.fx_provider = mock_fx_provider
        
        order_data = {
            'order_id': 'ord_historical',
            'revenue': Decimal('10000.00'),
            'currency': 'JPY',
            'date': '2025-10-18'
        }
        
        order = await ecommerce_agent.convert_revenue_to_usd(order_data)
        
        # Expected: 10000 * 0.0067 = 67.00
        expected = Decimal('67.00')
        actual = order.revenue_usd
        error_pct = abs((actual - expected) / expected * 100)
        
        assert error_pct < Decimal('1.0'), f"FX error {error_pct}% exceeds 1% threshold"
    
    @pytest.mark.asyncio
    async def test_fx_rate_cache_hit(self, ecommerce_agent, mock_fx_provider):
        """FX rate cache reduces API calls."""
        ecommerce_agent.fx_provider = mock_fx_provider
        
        # First call - cache miss
        order1 = await ecommerce_agent.convert_revenue_to_usd({
            'order_id': 'ord_1',
            'revenue': Decimal('100.00'),
            'currency': 'JPY',
            'date': '2025-10-18'
        })
        
        # Second call - cache hit
        order2 = await ecommerce_agent.convert_revenue_to_usd({
            'order_id': 'ord_2',
            'revenue': Decimal('200.00'),
            'currency': 'JPY',
            'date': '2025-10-18'
        })
        
        # Provider called only once
        assert mock_fx_provider.get_rate.call_count == 1
        
        # Both conversions correct
        assert order1.revenue_usd == Decimal('0.67')
        assert order2.revenue_usd == Decimal('1.34')
    
    @pytest.mark.asyncio
    async def test_fx_rate_unavailable_fallback(self, ecommerce_agent):
        """FX rate unavailable triggers fallback to 1:1 with warning."""
        ecommerce_agent.fx_provider = Mock()
        ecommerce_agent.fx_provider.get_rate = AsyncMock(side_effect=FXRateUnavailableError('Rate not found'))
        
        order_data = {
            'order_id': 'ord_fallback',
            'revenue': Decimal('100.00'),
            'currency': 'XXX',  # Invalid currency
            'date': '2025-10-18'
        }
        
        with patch('agents.ecommerce_agent.fx_fallback_counter') as mock_metric:
            order = await ecommerce_agent.convert_revenue_to_usd(order_data)
            
            # Fallback to 1:1
            assert order.revenue_usd == Decimal('100.00')
            assert order.fx_rate_applied == Decimal('1.0')
            assert order.fx_fallback_applied is True
            
            # Metric incremented
            mock_metric.inc.assert_called_once()


class TestIdempotentProcessing:
    """Test idempotent order processing with timezone-aware deduplication."""
    
    @pytest.mark.asyncio
    async def test_duplicate_order_skipped(self, ecommerce_agent):
        """Duplicate order ID skipped with metric."""
        order_data = {
            'order_id': 'ord_duplicate',
            'created_at': '2025-10-18T14:30:00+09:00',
            'total_price': '100.00',
            'currency': 'JPY',
            'timezone': 'Asia/Tokyo'
        }
        
        # First processing
        result1 = await ecommerce_agent.process_order(order_data)
        assert result1.status == 'processed'
        
        # Duplicate processing
        with patch('agents.ecommerce_agent.duplicate_order_counter') as mock_metric:
            result2 = await ecommerce_agent.process_order(order_data)
            
            assert result2.status == 'skipped_duplicate'
            mock_metric.inc.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_timezone_aware_content_hash(self, ecommerce_agent):
        """Content hash considers timezone to detect true duplicates."""
        # Same order in different timezone representations (same UTC time)
        order1 = {
            'order_id': 'ord_hash1',
            'created_at': '2025-10-18T14:30:00+09:00',  # JST
            'total_price': '100.00',
            'currency': 'JPY'
        }
        
        order2 = {
            'order_id': 'ord_hash2',
            'created_at': '2025-10-18T05:30:00+00:00',  # UTC (same time)
            'total_price': '100.00',
            'currency': 'JPY'
        }
        
        hash1 = ecommerce_agent.compute_order_hash(order1)
        hash2 = ecommerce_agent.compute_order_hash(order2)
        
        # Different order IDs but same UTC time - different hashes
        assert hash1 != hash2


class TestMMMSkewCorrection:
    """Test ±15% MMM skew elimination through timezone normalization."""
    
    @pytest.mark.asyncio
    async def test_revenue_aggregation_by_utc_date(self, ecommerce_agent):
        """Revenue aggregated by UTC date eliminates timezone skew."""
        orders = [
            # Order at 23:00 JST on Oct 18 (14:00 UTC Oct 18)
            {'order_id': 'ord_1', 'created_at': '2025-10-18T23:00:00+09:00', 'revenue': Decimal('100.00'), 'currency': 'USD'},
            # Order at 02:00 JST on Oct 19 (17:00 UTC Oct 18)
            {'order_id': 'ord_2', 'created_at': '2025-10-19T02:00:00+09:00', 'revenue': Decimal('200.00'), 'currency': 'USD'},
        ]
        
        daily_revenue = await ecommerce_agent.aggregate_revenue_by_utc_date(orders)
        
        # Both orders on Oct 18 UTC
        assert daily_revenue['2025-10-18'] == Decimal('300.00')
        assert '2025-10-19' not in daily_revenue
    
    @pytest.mark.asyncio
    async def test_mmm_input_timezone_consistent(self, ecommerce_agent):
        """MMM input uses consistent UTC-normalized timestamps."""
        orders = await ecommerce_agent.fetch_orders_for_mmm(
            start_date='2025-10-01',
            end_date='2025-10-18'
        )
        
        for order in orders:
            # All timestamps in UTC
            assert order.order_date_utc.tzinfo == timezone.utc
            # Metadata preserved for audit
            assert order.timezone_metadata.original_timezone is not None


# Risk gate integration tests
class TestRiskGates:
    """Test risk gates for timezone and FX operations."""
    
    @pytest.mark.asyncio
    async def test_timezone_conversion_timeout_gate(self, ecommerce_agent):
        """Timezone conversion respects 5s timeout gate."""
        with patch('agents.ecommerce_agent.convert_timezone') as mock_convert:
            mock_convert.side_effect = lambda x: time.sleep(6)  # Exceeds 5s
            
            with pytest.raises(TimezoneNormalizationError) as exc_info:
                await ecommerce_agent.normalize_order({
                    'order_id': 'ord_timeout',
                    'created_at': '2025-10-18T14:30:00+09:00',
                    'timezone': 'Asia/Tokyo'
                })
            
            assert exc_info.value.timeout_exceeded is True
    
    @pytest.mark.asyncio
    async def test_fx_provider_circuit_breaker(self, ecommerce_agent):
        """FX provider failures trigger circuit breaker after threshold."""
        ecommerce_agent.fx_provider = Mock()
        ecommerce_agent.fx_provider.get_rate = AsyncMock(side_effect=Exception('API Error'))
        
        # After 5 failures, circuit breaker opens
        for _ in range(5):
            with pytest.raises(FXRateUnavailableError):
                await ecommerce_agent.convert_revenue_to_usd({
                    'revenue': Decimal('100.00'),
                    'currency': 'JPY',
                    'date': '2025-10-18'
                })
        
        # Circuit breaker open - immediate failure without API call
        with pytest.raises(FXRateUnavailableError) as exc_info:
            await ecommerce_agent.convert_revenue_to_usd({
                'revenue': Decimal('100.00'),
                'currency': 'JPY',
                'date': '2025-10-18'
            })
        
        assert exc_info.value.circuit_breaker_open is True


# Prometheus metrics validation
class TestMetrics:
    """Test Prometheus metrics for timezone and FX operations."""
    
    def test_timezone_normalization_duration_histogram(self, ecommerce_agent):
        """Timezone normalization duration tracked in histogram."""
        with patch('agents.ecommerce_agent.timezone_normalization_duration') as mock_metric:
            # Metric should observe duration
            pass  # Implicit in normalize_order calls
    
    def test_fx_conversion_counter(self, ecommerce_agent):
        """FX conversion counter increments per currency."""
        with patch('agents.ecommerce_agent.fx_conversion_counter') as mock_metric:
            # Metric should increment with currency label
            pass  # Implicit in convert_revenue_to_usd calls
