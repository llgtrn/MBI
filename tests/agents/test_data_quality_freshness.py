"""
Unit tests for data freshness rejection with >6h SLA
Validates Q_004 + A_012 + Q_027: Freshness enforcement for GA4/MTA integrity
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock


class TestDataQualityFreshness:
    """Test suite for freshness rejection >6h (Q_004 + A_012 + Q_027)"""
    
    @pytest.mark.asyncio
    async def test_reject_6h01m(self):
        """
        Acceptance: Data with export_time >6h01m is rejected
        Contract: Raises DataFreshnessError with metric
        Metric: data_freshness_violations_total increments
        """
        from agents.data_quality import DataQualityAgent, DataFreshnessError
        
        agent = DataQualityAgent(
            freshness_sla_hours=6,
            enable_freshness_check=True
        )
        
        # Mock data with timestamp 6h01m old
        old_timestamp = datetime.utcnow() - timedelta(hours=6, minutes=1)
        stale_data = {
            'export_time': old_timestamp.isoformat(),
            'source': 'ga4',
            'records': [{'event': 'page_view'}]
        }
        
        # Should reject
        with pytest.raises(DataFreshnessError) as exc_info:
            await agent.validate_freshness(stale_data)
        
        # Verify error message
        assert 'export_time' in str(exc_info.value)
        assert '6h' in str(exc_info.value)
        
        # Check metrics
        metrics = agent.get_metrics()
        assert metrics['data_freshness_violations_total'] == 1
        assert metrics['rejected_records_total'] > 0
    
    @pytest.mark.asyncio
    async def test_accept_within_sla(self):
        """
        Data within 6h SLA should be accepted
        """
        from agents.data_quality import DataQualityAgent
        
        agent = DataQualityAgent(
            freshness_sla_hours=6,
            enable_freshness_check=True
        )
        
        # Fresh data (5h59m old)
        fresh_timestamp = datetime.utcnow() - timedelta(hours=5, minutes=59)
        fresh_data = {
            'export_time': fresh_timestamp.isoformat(),
            'source': 'ga4',
            'records': [{'event': 'page_view'}]
        }
        
        # Should accept
        result = await agent.validate_freshness(fresh_data)
        
        assert result is True
        
        # No violations
        metrics = agent.get_metrics()
        assert metrics['data_freshness_violations_total'] == 0
    
    @pytest.mark.asyncio
    async def test_ga4_specific_freshness(self):
        """
        Q_027: GA4 export_time validation for MTA corruption prevention
        """
        from agents.data_quality import DataQualityAgent, DataFreshnessError
        
        agent = DataQualityAgent(freshness_sla_hours=6)
        
        # GA4 BigQuery export with stale export_time
        ga4_stale = {
            'export_time': (datetime.utcnow() - timedelta(hours=7)).isoformat(),
            'source': 'ga4_bigquery_export',
            'table': 'events_20251018',
            'records': [
                {'event_name': 'session_start', 'user_pseudo_id': 'xyz'}
            ]
        }
        
        with pytest.raises(DataFreshnessError) as exc:
            await agent.validate_freshness(ga4_stale)
        
        # Specific error for GA4
        assert 'ga4' in str(exc.value).lower()
        assert 'mta' in str(exc.value).lower() or 'attribution' in str(exc.value).lower()
    
    @pytest.mark.asyncio
    async def test_kill_switch_disables_check(self):
        """
        Risk gate: ENABLE_FRESHNESS_CHECK kill switch
        """
        from agents.data_quality import DataQualityAgent
        
        agent = DataQualityAgent(
            freshness_sla_hours=6,
            enable_freshness_check=False  # Kill switch OFF
        )
        
        # Stale data
        stale_data = {
            'export_time': (datetime.utcnow() - timedelta(hours=24)).isoformat(),
            'source': 'ga4',
            'records': [{'event': 'page_view'}]
        }
        
        # Should accept when kill switch is off
        result = await agent.validate_freshness(stale_data)
        
        assert result is True
        
        # Metrics show checks disabled
        metrics = agent.get_metrics()
        assert metrics['freshness_checks_disabled_total'] > 0
    
    @pytest.mark.asyncio
    async def test_backpressure_monitoring(self):
        """
        Risk gate: Monitor rejection backpressure
        """
        from agents.data_quality import DataQualityAgent, DataFreshnessError
        
        agent = DataQualityAgent(
            freshness_sla_hours=6,
            rejection_backpressure_threshold=100
        )
        
        stale_data = {
            'export_time': (datetime.utcnow() - timedelta(hours=7)).isoformat(),
            'source': 'ga4',
            'records': [{'event': 'page_view'}] * 150  # 150 records
        }
        
        with pytest.raises(DataFreshnessError):
            await agent.validate_freshness(stale_data)
        
        # Check backpressure metrics
        metrics = agent.get_metrics()
        assert metrics['rejected_records_total'] == 150
        
        # Should emit backpressure alert if threshold exceeded
        if metrics['rejected_records_total'] > agent.rejection_backpressure_threshold:
            assert metrics['backpressure_alerts_total'] > 0
    
    @pytest.mark.asyncio
    async def test_prometheus_counter_increments(self):
        """
        Verify prometheus metric data_freshness_violations_total increments
        """
        from agents.data_quality import DataQualityAgent, DataFreshnessError
        
        agent = DataQualityAgent(freshness_sla_hours=6)
        
        # Initial state
        metrics = agent.get_metrics()
        initial_count = metrics['data_freshness_violations_total']
        
        stale_data = {
            'export_time': (datetime.utcnow() - timedelta(hours=7)).isoformat(),
            'source': 'test',
            'records': [{}]
        }
        
        # Trigger violation
        try:
            await agent.validate_freshness(stale_data)
        except DataFreshnessError:
            pass
        
        # Check increment
        metrics = agent.get_metrics()
        assert metrics['data_freshness_violations_total'] == initial_count + 1
    
    @pytest.mark.asyncio
    async def test_freshness_sla_config(self):
        """
        Risk gate: Configurable freshness SLA (6h default)
        """
        from agents.data_quality import DataQualityAgent, DataFreshnessError
        
        # Custom SLA: 12h
        agent = DataQualityAgent(freshness_sla_hours=12)
        
        # 10h old data
        data_10h = {
            'export_time': (datetime.utcnow() - timedelta(hours=10)).isoformat(),
            'source': 'test',
            'records': [{}]
        }
        
        # Should accept (within 12h SLA)
        result = await agent.validate_freshness(data_10h)
        assert result is True
        
        # 13h old data
        data_13h = {
            'export_time': (datetime.utcnow() - timedelta(hours=13)).isoformat(),
            'source': 'test',
            'records': [{}]
        }
        
        # Should reject (exceeds 12h SLA)
        with pytest.raises(DataFreshnessError):
            await agent.validate_freshness(data_13h)
