"""
Tests for ingestion idempotency

Verifies that duplicate SpendRecord insertions are handled correctly
via ON CONFLICT DO UPDATE logic.
"""

import pytest
from datetime import date, datetime
from sqlalchemy import text, create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
import asyncio

from src.agents.ingestion.ad_platform_agent import (
    AdPlatformAgent,
    SpendRecord,
    AdChannel,
    IngestionStats
)


@pytest.fixture
async def async_db_session():
    """Create async test database session"""
    # Use in-memory SQLite for testing (would use PostgreSQL in real tests)
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False
    )
    
    # Create schema
    async with engine.begin() as conn:
        await conn.execute(text("""
            CREATE TABLE fct_ad_metric_daily (
                dt DATE NOT NULL,
                channel VARCHAR(50) NOT NULL,
                campaign_id VARCHAR(255) NOT NULL,
                adset_id VARCHAR(255),
                ad_id VARCHAR(255),
                impressions INTEGER NOT NULL DEFAULT 0,
                clicks INTEGER NOT NULL DEFAULT 0,
                spend DECIMAL(12, 2) NOT NULL DEFAULT 0.00,
                currency VARCHAR(3) NOT NULL DEFAULT 'JPY',
                conversions INTEGER DEFAULT 0,
                conversion_value DECIMAL(12, 2) DEFAULT 0.00,
                source_system VARCHAR(50) NOT NULL,
                ingested_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (dt, channel, campaign_id),
                UNIQUE (channel, campaign_id, dt)
            )
        """))
    
    # Create session
    async_session_maker = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session_maker() as session:
        yield session
    
    await engine.dispose()


@pytest.fixture
def sample_spend_record():
    """Sample SpendRecord for testing"""
    return SpendRecord(
        dt=date(2025, 10, 19),
        channel=AdChannel.META,
        campaign_id="campaign_123",
        adset_id="adset_456",
        ad_id="ad_789",
        impressions=10000,
        clicks=250,
        spend=5000.00,
        currency="JPY",
        conversions=10,
        conversion_value=50000.00,
        source_system="meta_api"
    )


class TestIdempotency:
    """Test idempotent upsert behavior"""
    
    @pytest.mark.asyncio
    async def test_duplicate_spend_record_upserts(
        self,
        async_db_session,
        sample_spend_record
    ):
        """
        Test that inserting the same record twice results in UPDATE
        
        Acceptance: Second insert should update, not error
        """
        agent = AdPlatformAgent(async_db_session)
        
        # First insert
        result1 = await agent._upsert_spend_record(sample_spend_record)
        assert result1 == "inserted"
        
        # Verify record exists
        count_result = await async_db_session.execute(text("""
            SELECT COUNT(*) as cnt FROM fct_ad_metric_daily
            WHERE channel = :channel AND campaign_id = :campaign_id AND dt = :dt
        """), {
            "channel": sample_spend_record.channel.value,
            "campaign_id": sample_spend_record.campaign_id,
            "dt": sample_spend_record.dt
        })
        count = count_result.fetchone().cnt
        assert count == 1
        
        # Second insert (duplicate) with updated metrics
        updated_record = sample_spend_record.copy(update={
            "impressions": 15000,
            "clicks": 350,
            "spend": 7000.00
        })
        
        result2 = await agent._upsert_spend_record(updated_record)
        assert result2 == "updated"
        
        # Verify still only one record
        count_result = await async_db_session.execute(text("""
            SELECT COUNT(*) as cnt FROM fct_ad_metric_daily
            WHERE channel = :channel AND campaign_id = :campaign_id AND dt = :dt
        """), {
            "channel": sample_spend_record.channel.value,
            "campaign_id": sample_spend_record.campaign_id,
            "dt": sample_spend_record.dt
        })
        count = count_result.fetchone().cnt
        assert count == 1
        
        # Verify metrics were updated
        data_result = await async_db_session.execute(text("""
            SELECT impressions, clicks, spend FROM fct_ad_metric_daily
            WHERE channel = :channel AND campaign_id = :campaign_id AND dt = :dt
        """), {
            "channel": sample_spend_record.channel.value,
            "campaign_id": sample_spend_record.campaign_id,
            "dt": sample_spend_record.dt
        })
        row = data_result.fetchone()
        assert row.impressions == 15000
        assert row.clicks == 350
        assert float(row.spend) == 7000.00
    
    @pytest.mark.asyncio
    async def test_different_campaigns_separate_records(
        self,
        async_db_session,
        sample_spend_record
    ):
        """
        Test that different campaigns create separate records
        
        Acceptance: Two records with different campaign_ids should both insert
        """
        agent = AdPlatformAgent(async_db_session)
        
        # Insert first campaign
        result1 = await agent._upsert_spend_record(sample_spend_record)
        assert result1 == "inserted"
        
        # Insert different campaign (same channel and date)
        different_campaign = sample_spend_record.copy(update={
            "campaign_id": "campaign_999"
        })
        
        result2 = await agent._upsert_spend_record(different_campaign)
        assert result2 == "inserted"
        
        # Verify two records exist
        count_result = await async_db_session.execute(text("""
            SELECT COUNT(*) as cnt FROM fct_ad_metric_daily
            WHERE channel = :channel AND dt = :dt
        """), {
            "channel": sample_spend_record.channel.value,
            "dt": sample_spend_record.dt
        })
        count = count_result.fetchone().cnt
        assert count == 2
    
    @pytest.mark.asyncio
    async def test_idempotency_key_generation(self, sample_spend_record):
        """Test idempotency key format"""
        expected_key = f"meta|campaign_123|2025-10-19"
        assert sample_spend_record.idempotency_key == expected_key


class TestValidation:
    """Test SpendRecord validation"""
    
    def test_clicks_cannot_exceed_impressions(self):
        """Test that clicks > impressions raises ValidationError"""
        with pytest.raises(ValueError, match="cannot exceed impressions"):
            SpendRecord(
                dt=date(2025, 10, 19),
                channel=AdChannel.META,
                campaign_id="campaign_123",
                impressions=100,
                clicks=150,  # More than impressions
                spend=1000.00,
                source_system="meta_api"
            )
    
    def test_negative_metrics_rejected(self):
        """Test that negative metrics are rejected"""
        with pytest.raises(ValueError):
            SpendRecord(
                dt=date(2025, 10, 19),
                channel=AdChannel.META,
                campaign_id="campaign_123",
                impressions=-100,  # Negative
                clicks=50,
                spend=1000.00,
                source_system="meta_api"
            )
    
    def test_valid_record_accepted(self, sample_spend_record):
        """Test that valid record is accepted"""
        assert sample_spend_record.impressions == 10000
        assert sample_spend_record.clicks == 250
        assert sample_spend_record.spend == 5000.00


class TestIngestionStats:
    """Test IngestionStats calculations"""
    
    @pytest.mark.asyncio
    async def test_success_rate_calculation(self):
        """Test success rate calculation"""
        stats = IngestionStats(
            records_processed=100,
            records_inserted=70,
            records_updated=20,
            records_skipped=5,
            errors=["error1", "error2", "error3", "error4", "error5"]
        )
        
        assert stats.success_rate == 0.90  # (70 + 20) / 100
    
    @pytest.mark.asyncio
    async def test_zero_processed_success_rate(self):
        """Test success rate when no records processed"""
        stats = IngestionStats(records_processed=0)
        assert stats.success_rate == 0.0


class TestDataQualityValidation:
    """Test data quality validation checks"""
    
    @pytest.mark.asyncio
    async def test_validate_data_quality(
        self,
        async_db_session,
        sample_spend_record
    ):
        """Test data quality validation checks"""
        agent = AdPlatformAgent(async_db_session)
        
        # Insert sample record
        await agent._upsert_spend_record(sample_spend_record)
        await async_db_session.commit()
        
        # Run validation
        checks = await agent.validate_data_quality(
            channel=AdChannel.META,
            dt=date(2025, 10, 19)
        )
        
        assert checks["has_records"] is True
        # Other checks depend on DB computed columns not available in SQLite


# Integration test
class TestIngestionWorkflow:
    """Test complete ingestion workflow"""
    
    @pytest.mark.asyncio
    async def test_full_ingestion_workflow(
        self,
        async_db_session
    ):
        """
        Test complete workflow: ingest, validate, re-ingest
        
        Acceptance: Re-ingestion should update, not create duplicates
        """
        agent = AdPlatformAgent(async_db_session)
        
        # Create test records
        records = [
            SpendRecord(
                dt=date(2025, 10, 19),
                channel=AdChannel.META,
                campaign_id=f"campaign_{i}",
                impressions=1000 * i,
                clicks=50 * i,
                spend=500.00 * i,
                source_system="meta_api"
            )
            for i in range(1, 6)
        ]
        
        # First ingestion
        for record in records:
            result = await agent._upsert_spend_record(record)
            assert result == "inserted"
        
        await async_db_session.commit()
        
        # Verify 5 records
        count_result = await async_db_session.execute(text("""
            SELECT COUNT(*) as cnt FROM fct_ad_metric_daily
        """))
        assert count_result.fetchone().cnt == 5
        
        # Re-ingest with updated metrics
        for i, record in enumerate(records, 1):
            updated = record.copy(update={
                "impressions": 2000 * i,
                "clicks": 100 * i
            })
            result = await agent._upsert_spend_record(updated)
            assert result == "updated"
        
        await async_db_session.commit()
        
        # Still 5 records (no duplicates)
        count_result = await async_db_session.execute(text("""
            SELECT COUNT(*) as cnt FROM fct_ad_metric_daily
        """))
        assert count_result.fetchone().cnt == 5
        
        # Verify updates
        data_result = await async_db_session.execute(text("""
            SELECT impressions FROM fct_ad_metric_daily
            WHERE campaign_id = 'campaign_1'
        """))
        row = data_result.fetchone()
        assert row.impressions == 2000
