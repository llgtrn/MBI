"""
Ad Platform Ingestion Agent

Pulls spend, impressions, clicks from ad platforms (Meta, Google, TikTok)
with idempotent upsert logic to prevent duplicate records.

Features:
- Idempotent ingestion via ON CONFLICT DO UPDATE
- Rate limiting and retry logic
- Backfill support for historical data
- Metric validation and quality checks
"""

from typing import List, Dict, Optional
from datetime import date, datetime, timedelta
from pydantic import BaseModel, Field, validator
import logging
from enum import Enum
import asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class AdChannel(str, Enum):
    """Supported ad platforms"""
    META = "meta"
    GOOGLE = "google"
    TIKTOK = "tiktok"
    YOUTUBE = "youtube"
    LINKEDIN = "linkedin"
    TWITTER = "twitter"
    SNAPCHAT = "snapchat"


class SpendRecord(BaseModel):
    """
    Daily ad spend record with validation
    
    Idempotency: (channel, campaign_id, dt) must be unique
    """
    dt: date = Field(..., description="Date of metrics")
    channel: AdChannel = Field(..., description="Ad platform channel")
    campaign_id: str = Field(..., min_length=1, max_length=255)
    adset_id: Optional[str] = Field(None, max_length=255)
    ad_id: Optional[str] = Field(None, max_length=255)
    
    impressions: int = Field(..., ge=0)
    clicks: int = Field(..., ge=0)
    spend: float = Field(..., ge=0)
    currency: str = Field(default="JPY", max_length=3)
    
    conversions: int = Field(default=0, ge=0)
    conversion_value: float = Field(default=0.0, ge=0)
    
    source_system: str = Field(..., description="Source API name")
    
    @validator("clicks")
    def clicks_cannot_exceed_impressions(cls, v, values):
        """Validate clicks <= impressions"""
        if "impressions" in values and v > values["impressions"]:
            raise ValueError(
                f"Clicks ({v}) cannot exceed impressions ({values['impressions']})"
            )
        return v
    
    @property
    def idempotency_key(self) -> str:
        """Generate idempotency key for deduplication"""
        return f"{self.channel.value}|{self.campaign_id}|{self.dt}"
    
    class Config:
        use_enum_values = True


class IngestionStats(BaseModel):
    """Statistics from ingestion run"""
    records_processed: int = 0
    records_inserted: int = 0
    records_updated: int = 0
    records_skipped: int = 0
    errors: List[str] = []
    duration_seconds: float = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.records_processed == 0:
            return 0.0
        return (self.records_inserted + self.records_updated) / self.records_processed


class AdPlatformAgent:
    """
    Agent for ingesting ad platform data with idempotency
    """
    
    def __init__(self, db_session: AsyncSession):
        """
        Initialize agent
        
        Args:
            db_session: Async database session
        """
        self.db = db_session
        self.rate_limiter = asyncio.Semaphore(10)  # Max 10 concurrent API calls
    
    async def ingest_daily(
        self,
        channel: AdChannel,
        dt: date,
        api_client: Optional[object] = None
    ) -> IngestionStats:
        """
        Ingest daily metrics for a channel
        
        Args:
            channel: Ad platform channel
            dt: Date to ingest
            api_client: Platform-specific API client (injected)
        
        Returns:
            IngestionStats with results
        """
        start_time = datetime.utcnow()
        stats = IngestionStats()
        
        logger.info(f"Starting ingestion for {channel.value} on {dt}")
        
        try:
            # Fetch data from platform API
            records = await self._fetch_from_api(channel, dt, api_client)
            stats.records_processed = len(records)
            
            # Upsert with idempotency
            for record in records:
                try:
                    result = await self._upsert_spend_record(record)
                    if result == "inserted":
                        stats.records_inserted += 1
                    elif result == "updated":
                        stats.records_updated += 1
                    else:
                        stats.records_skipped += 1
                except Exception as e:
                    stats.errors.append(f"{record.idempotency_key}: {str(e)}")
                    logger.error(
                        f"Failed to upsert {record.idempotency_key}: {e}"
                    )
            
            await self.db.commit()
            
        except Exception as e:
            stats.errors.append(f"Ingestion failed: {str(e)}")
            logger.error(f"Ingestion error for {channel.value} on {dt}: {e}")
            await self.db.rollback()
        
        finally:
            stats.duration_seconds = (
                datetime.utcnow() - start_time
            ).total_seconds()
        
        logger.info(
            f"Ingestion complete for {channel.value} on {dt}: "
            f"{stats.records_inserted} inserted, {stats.records_updated} updated, "
            f"{stats.records_skipped} skipped, {len(stats.errors)} errors"
        )
        
        return stats
    
    async def _fetch_from_api(
        self,
        channel: AdChannel,
        dt: date,
        api_client: Optional[object]
    ) -> List[SpendRecord]:
        """
        Fetch data from platform API
        
        Args:
            channel: Ad platform
            dt: Date to fetch
            api_client: Platform-specific API client
        
        Returns:
            List of SpendRecord objects
        """
        # This is a stub - actual implementation would call platform APIs
        # For now, return empty list or mock data
        
        if api_client is None:
            logger.warning(
                f"No API client provided for {channel.value}, skipping fetch"
            )
            return []
        
        # Platform-specific API calls would go here
        # Example for Meta:
        # if channel == AdChannel.META:
        #     return await self._fetch_meta_ads(dt, api_client)
        
        async with self.rate_limiter:
            # Placeholder for actual API call
            logger.info(f"Fetching {channel.value} data for {dt}")
            await asyncio.sleep(0.1)  # Simulate API latency
            
            # Return empty list for now
            return []
    
    async def _upsert_spend_record(self, record: SpendRecord) -> str:
        """
        Insert or update spend record with idempotency
        
        Args:
            record: SpendRecord to upsert
        
        Returns:
            "inserted", "updated", or "skipped"
        """
        # SQL with ON CONFLICT for idempotency
        upsert_sql = text("""
            INSERT INTO fct_ad_metric_daily (
                dt, channel, campaign_id, adset_id, ad_id,
                impressions, clicks, spend, currency,
                conversions, conversion_value, source_system
            ) VALUES (
                :dt, :channel, :campaign_id, :adset_id, :ad_id,
                :impressions, :clicks, :spend, :currency,
                :conversions, :conversion_value, :source_system
            )
            ON CONFLICT (channel, campaign_id, dt) 
            DO UPDATE SET
                adset_id = EXCLUDED.adset_id,
                ad_id = EXCLUDED.ad_id,
                impressions = EXCLUDED.impressions,
                clicks = EXCLUDED.clicks,
                spend = EXCLUDED.spend,
                currency = EXCLUDED.currency,
                conversions = EXCLUDED.conversions,
                conversion_value = EXCLUDED.conversion_value,
                source_system = EXCLUDED.source_system,
                updated_at = CURRENT_TIMESTAMP
            RETURNING (xmax = 0) AS inserted
        """)
        
        result = await self.db.execute(
            upsert_sql,
            {
                "dt": record.dt,
                "channel": record.channel.value,
                "campaign_id": record.campaign_id,
                "adset_id": record.adset_id,
                "ad_id": record.ad_id,
                "impressions": record.impressions,
                "clicks": record.clicks,
                "spend": record.spend,
                "currency": record.currency,
                "conversions": record.conversions,
                "conversion_value": record.conversion_value,
                "source_system": record.source_system
            }
        )
        
        row = result.fetchone()
        if row is None:
            return "skipped"
        
        # xmax = 0 means INSERT, xmax > 0 means UPDATE
        return "inserted" if row.inserted else "updated"
    
    async def backfill(
        self,
        channel: AdChannel,
        start_date: date,
        end_date: date,
        api_client: Optional[object] = None
    ) -> Dict[date, IngestionStats]:
        """
        Backfill historical data for a date range
        
        Args:
            channel: Ad platform
            start_date: Start of backfill range
            end_date: End of backfill range (inclusive)
            api_client: Platform-specific API client
        
        Returns:
            Dict mapping date to IngestionStats
        """
        logger.info(
            f"Starting backfill for {channel.value} "
            f"from {start_date} to {end_date}"
        )
        
        results = {}
        current_date = start_date
        
        while current_date <= end_date:
            stats = await self.ingest_daily(channel, current_date, api_client)
            results[current_date] = stats
            
            # Rate limiting between days
            await asyncio.sleep(0.5)
            
            current_date += timedelta(days=1)
        
        # Summary stats
        total_inserted = sum(s.records_inserted for s in results.values())
        total_updated = sum(s.records_updated for s in results.values())
        total_errors = sum(len(s.errors) for s in results.values())
        
        logger.info(
            f"Backfill complete for {channel.value}: "
            f"{total_inserted} inserted, {total_updated} updated, "
            f"{total_errors} errors across {len(results)} days"
        )
        
        return results
    
    async def validate_data_quality(
        self,
        channel: AdChannel,
        dt: date
    ) -> Dict[str, bool]:
        """
        Validate data quality for ingested records
        
        Args:
            channel: Ad platform
            dt: Date to validate
        
        Returns:
            Dict of validation checks and results
        """
        checks = {}
        
        # Check: Records exist for date
        count_sql = text("""
            SELECT COUNT(*) as cnt
            FROM fct_ad_metric_daily
            WHERE channel = :channel AND dt = :dt
        """)
        result = await self.db.execute(
            count_sql,
            {"channel": channel.value, "dt": dt}
        )
        row = result.fetchone()
        checks["has_records"] = row.cnt > 0 if row else False
        
        # Check: No zero spend with clicks
        invalid_sql = text("""
            SELECT COUNT(*) as cnt
            FROM fct_ad_metric_daily
            WHERE channel = :channel AND dt = :dt
              AND clicks > 0 AND spend = 0
        """)
        result = await self.db.execute(
            invalid_sql,
            {"channel": channel.value, "dt": dt}
        )
        row = result.fetchone()
        checks["no_zero_spend_with_clicks"] = row.cnt == 0 if row else True
        
        # Check: CTR in reasonable range (0-100%)
        ctr_sql = text("""
            SELECT COUNT(*) as cnt
            FROM fct_ad_metric_daily
            WHERE channel = :channel AND dt = :dt
              AND (ctr < 0 OR ctr > 1)
        """)
        result = await self.db.execute(
            ctr_sql,
            {"channel": channel.value, "dt": dt}
        )
        row = result.fetchone()
        checks["ctr_in_range"] = row.cnt == 0 if row else True
        
        return checks


# Example usage (to be integrated with orchestrator)
async def example_usage():
    """Example of how to use AdPlatformAgent"""
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    
    # Create async engine
    engine = create_async_engine(
        "postgresql+asyncpg://user:pass@localhost/mbi",
        echo=True
    )
    
    # Create session
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        agent = AdPlatformAgent(session)
        
        # Ingest today's data
        stats = await agent.ingest_daily(
            channel=AdChannel.META,
            dt=date.today(),
            api_client=None  # Would pass actual API client
        )
        
        print(f"Ingestion stats: {stats}")
        
        # Validate
        quality = await agent.validate_data_quality(
            channel=AdChannel.META,
            dt=date.today()
        )
        
        print(f"Data quality: {quality}")
