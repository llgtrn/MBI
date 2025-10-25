"""
Ad Platform Agent with idempotent spend ingestion
Supports Meta Marketing API with upsert logic for duplicate prevention

Features:
- Idempotent spend record insertion via ON CONFLICT DO UPDATE
- Schema validation with version fallback
- Automatic deduplication
- Data normalization
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, date
from decimal import Decimal
import logging

from agents.schema_validator import SchemaValidator
from core.contracts import AdMetric, SchemaChangeEvent
from contracts.ingestion_schemas import SpendRecord

logger = logging.getLogger(__name__)


class MetaAdPlatformAgent:
    """
    Meta Marketing API integration with idempotent ingestion
    
    Features:
    - Multi-version support (v18/v19)
    - Backward compatibility checking
    - Automatic fallback on breaking changes
    - **Idempotent spend record insertion**
    - Schema drift alerts
    """
    
    def __init__(
        self,
        api_version: str = "v19.0",
        fallback_version: str = "v18.0",
        access_token: Optional[str] = None,
        account_id: Optional[str] = None,
        strict_validation: bool = True,
        db_connection = None  # Database connection for upserts
    ):
        self.requested_api_version = api_version
        self.fallback_version = fallback_version
        self.current_api_version = api_version
        self.access_token = access_token
        self.account_id = account_id
        self.strict_validation = strict_validation
        self.validation_passed = False
        self.fallback_triggered = False
        self.db_conn = db_connection
        
        # Initialize schema validator
        self.schema_validator = SchemaValidator(
            enable_strict_validation=strict_validation
        )
        
        # Register known schemas
        self._register_known_schemas()
    
    def _register_known_schemas(self) -> None:
        """Register known Meta API schemas"""
        # v18.0 schema
        v18_schema = {
            "version": "v18.0",
            "fields": {
                "id": {"type": "string", "required": True},
                "campaign_id": {"type": "string", "required": True},
                "adset_id": {"type": "string", "required": False},
                "spend": {"type": "number", "required": True},
                "impressions": {"type": "integer", "required": True},
                "clicks": {"type": "integer", "required": True},
                "cpm": {"type": "number", "required": False},
                "ctr": {"type": "number", "required": False},
                "date_start": {"type": "string", "format": "date", "required": True},
                "date_stop": {"type": "string", "format": "date", "required": True},
                "account_currency": {"type": "string", "required": True},
                "buying_type": {"type": "string", "enum": ["AUCTION", "RESERVED"], "required": False}
            }
        }
        
        # v19.0 compatible schema (adds optional fields)
        v19_schema = {
            "version": "v19.0",
            "fields": {
                "id": {"type": "string", "required": True},
                "campaign_id": {"type": "string", "required": True},
                "adset_id": {"type": "string", "required": False},
                "spend": {"type": "number", "required": True},
                "impressions": {"type": "integer", "required": True},
                "clicks": {"type": "integer", "required": True},
                "cpm": {"type": "number", "required": False},
                "ctr": {"type": "number", "required": False},
                "date_start": {"type": "string", "format": "date", "required": True},
                "date_stop": {"type": "string", "format": "date", "required": True},
                "account_currency": {"type": "string", "required": True},
                "buying_type": {"type": "string", "enum": ["AUCTION", "RESERVED"], "required": False},
                "optimization_goal": {"type": "string", "required": False},
                "bid_strategy": {"type": "string", "required": False}
            }
        }
        
        self.schema_validator.register_schema("meta", "v18.0", v18_schema)
        self.schema_validator.register_schema("meta", "v19.0", v19_schema)
    
    def validate_api_version(self) -> None:
        """
        Validate API version compatibility
        
        Raises:
            ValueError: If requested version breaks backward compatibility with fallback
        """
        if self.requested_api_version == self.fallback_version:
            self.validation_passed = True
            return
        
        # Check backward compatibility
        current_schema = self.schema_validator.get_schema("meta", self.requested_api_version)
        if not current_schema:
            raise ValueError(f"Schema not found for {self.requested_api_version}")
        
        result = self.schema_validator.validate_backward_compatibility(
            provider="meta",
            old_version=self.fallback_version,
            new_version=self.requested_api_version,
            new_schema=current_schema.schema
        )
        
        if not result.is_compatible:
            # Breaking changes detected - fallback to v18
            logger.warning(
                f"Meta API {self.requested_api_version} breaks backward compatibility. "
                f"Falling back to {self.fallback_version}"
            )
            logger.warning(f"Breaking changes: {[c.description for c in result.breaking_changes]}")
            
            self.current_api_version = self.fallback_version
            self.fallback_triggered = True
            self.validation_passed = True
            
            raise ValueError(
                f"Meta API {self.requested_api_version} breaks backward compatibility. "
                f"Fallback to {self.fallback_version} activated."
            )
        else:
            self.validation_passed = True
            logger.info(f"Meta API {self.requested_api_version} is backward compatible")
    
    def ensure_compatible_version(self) -> None:
        """Ensure using compatible API version (with fallback)"""
        try:
            self.validate_api_version()
        except ValueError as e:
            if "backward compatibility" in str(e):
                # Fallback already triggered
                logger.info(f"Using fallback version {self.current_api_version}")
            else:
                raise
    
    def parse_response(self, response: Dict[str, Any]) -> List[SpendRecord]:
        """
        Parse Meta API response into SpendRecord objects
        
        Args:
            response: Raw API response
        
        Returns:
            List of SpendRecord objects
        """
        data = response.get("data", [])
        records = []
        
        for item in data:
            # Normalize to match our internal schema
            record = SpendRecord(
                date=date.fromisoformat(item["date_start"]),
                channel="meta",
                campaign_id=item["campaign_id"],
                adset_id=item.get("adset_id"),
                spend=float(item["spend"]),
                currency=item.get("account_currency", "USD"),
                impressions=int(item["impressions"]) if isinstance(item["impressions"], str) else item["impressions"],
                clicks=int(item["clicks"]) if isinstance(item["clicks"], str) else item["clicks"]
            )
            records.append(record)
        
        return records
    
    def ingest_spend_records(
        self,
        records: List[SpendRecord],
        dry_run: bool = False
    ) -> Dict[str, int]:
        """
        Ingest spend records with idempotent upsert logic.
        
        Uses PostgreSQL ON CONFLICT DO UPDATE to prevent duplicates
        based on unique constraint (channel, campaign_id, dt).
        
        Args:
            records: List of SpendRecord objects to ingest
            dry_run: If True, validate but don't insert
        
        Returns:
            Dict with counts: {inserted: int, updated: int, skipped: int}
        
        Raises:
            ValueError: If DB connection not configured
        """
        if not self.db_conn:
            raise ValueError("Database connection required for ingestion")
        
        stats = {"inserted": 0, "updated": 0, "skipped": 0}
        
        for record in records:
            # Generate idempotency key
            idempotency_key = f"{record.channel}:{record.campaign_id}:{record.date.isoformat()}"
            
            if dry_run:
                logger.info(f"[DRY RUN] Would upsert: {idempotency_key}")
                stats["skipped"] += 1
                continue
            
            # Upsert with ON CONFLICT DO UPDATE
            # This prevents duplicate (channel, campaign_id, dt) tuples
            query = """
                INSERT INTO fct_ad_metric_daily (
                    dt, ad_id, channel, campaign_id, adset_id,
                    impressions, clicks, spend, currency
                )
                VALUES (
                    %(dt)s, %(ad_id)s, %(channel)s, %(campaign_id)s, %(adset_id)s,
                    %(impressions)s, %(clicks)s, %(spend)s, %(currency)s
                )
                ON CONFLICT (channel, campaign_id, dt)
                DO UPDATE SET
                    adset_id = EXCLUDED.adset_id,
                    impressions = EXCLUDED.impressions,
                    clicks = EXCLUDED.clicks,
                    spend = EXCLUDED.spend,
                    currency = EXCLUDED.currency,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING (xmax = 0) AS is_insert
            """
            
            params = {
                "dt": record.date,
                "ad_id": f"{record.channel}_{record.campaign_id}_{record.date.isoformat()}",
                "channel": record.channel,
                "campaign_id": record.campaign_id,
                "adset_id": record.adset_id,
                "impressions": record.impressions,
                "clicks": record.clicks,
                "spend": record.spend,
                "currency": record.currency
            }
            
            try:
                cursor = self.db_conn.cursor()
                cursor.execute(query, params)
                result = cursor.fetchone()
                
                # result[0] (is_insert) is True if row was inserted, False if updated
                if result and result[0]:
                    stats["inserted"] += 1
                    logger.debug(f"Inserted new record: {idempotency_key}")
                else:
                    stats["updated"] += 1
                    logger.info(f"Updated existing record: {idempotency_key}")
                
                self.db_conn.commit()
                cursor.close()
            
            except Exception as e:
                self.db_conn.rollback()
                logger.error(f"Failed to upsert {idempotency_key}: {e}")
                stats["skipped"] += 1
                # Continue processing other records
        
        logger.info(
            f"Ingestion complete: inserted={stats['inserted']}, "
            f"updated={stats['updated']}, skipped={stats['skipped']}"
        )
        
        return stats
    
    def fetch_and_ingest_spend(
        self,
        start_date: date,
        end_date: date,
        campaign_ids: Optional[List[str]] = None,
        dry_run: bool = False
    ) -> Dict[str, int]:
        """
        Fetch spend data from Meta API and ingest with idempotency.
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            campaign_ids: Optional list of campaign IDs to filter
            dry_run: If True, validate but don't insert
        
        Returns:
            Dict with ingestion stats
        """
        # Ensure compatible version
        self.ensure_compatible_version()
        
        # Build API request (stub - would call actual Meta API)
        params = {
            "fields": [
                "id",
                "campaign_id",
                "adset_id",
                "spend",
                "impressions",
                "clicks",
                "cpm",
                "ctr",
                "account_currency",
                "buying_type"
            ],
            "time_range": {
                "since": start_date.isoformat(),
                "until": end_date.isoformat()
            }
        }
        
        if campaign_ids:
            params["filtering"] = [
                {
                    "field": "campaign.id",
                    "operator": "IN",
                    "value": campaign_ids
                }
            ]
        
        # Mock response for now
        response = {
            "data": []
        }
        
        # Parse and ingest
        records = self.parse_response(response)
        return self.ingest_spend_records(records, dry_run=dry_run)
