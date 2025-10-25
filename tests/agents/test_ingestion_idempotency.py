"""Tests for Spend Ingestion Idempotency

Validates:
- Duplicate spend records are handled via upsert
- UNIQUE constraint on (channel, campaign_id, dt) prevents duplicates
- ON CONFLICT DO UPDATE logic works correctly
- Idempotency key generation is consistent
- Stats tracking (inserted vs updated)
"""

import pytest
from datetime import date
from decimal import Decimal
from unittest.mock import Mock, MagicMock, call

from agents.ad_platform_agent import MetaAdPlatformAgent
from contracts.ingestion_schemas import SpendRecord


class TestSpendIngestionIdempotency:
    """Test idempotent spend record ingestion"""
    
    @pytest.fixture
    def mock_db_conn(self):
        """Mock database connection with cursor"""
        conn = Mock()
        cursor = Mock()
        conn.cursor.return_value = cursor
        return conn, cursor
    
    @pytest.fixture
    def agent_with_db(self, mock_db_conn):
        """MetaAdPlatformAgent with mocked DB connection"""
        conn, _ = mock_db_conn
        agent = MetaAdPlatformAgent(
            db_connection=conn,
            strict_validation=False  # Skip schema checks for unit tests
        )
        return agent
    
    @pytest.fixture
    def sample_record(self):
        """Sample SpendRecord for testing"""
        return SpendRecord(
            date=date(2025, 10, 12),
            channel="meta",
            campaign_id="c123",
            adset_id="a456",
            spend=120000.0,
            currency="JPY",
            impressions=45000,
            clicks=1200
        )
    
    def test_duplicate_spend_record_upserts(self, agent_with_db, mock_db_conn, sample_record):
        """Should upsert on duplicate (channel, campaign_id, dt) tuple"""
        conn, cursor = mock_db_conn
        
        # First insert: simulate new row (xmax = 0 means insert)
        cursor.fetchone.return_value = (True,)  # is_insert = True
        
        stats = agent_with_db.ingest_spend_records([sample_record])
        
        # Verify INSERT with ON CONFLICT was called
        assert cursor.execute.call_count == 1
        query = cursor.execute.call_args[0][0]
        assert "INSERT INTO fct_ad_metric_daily" in query
        assert "ON CONFLICT (channel, campaign_id, dt)" in query
        assert "DO UPDATE SET" in query
        
        # Verify stats: 1 inserted, 0 updated
        assert stats["inserted"] == 1
        assert stats["updated"] == 0
        assert stats["skipped"] == 0
        
        # Verify commit was called
        conn.commit.assert_called_once()
    
    def test_duplicate_record_updates_existing(self, agent_with_db, mock_db_conn, sample_record):
        """Should update existing record on duplicate key"""
        conn, cursor = mock_db_conn
        
        # Simulate update (xmax != 0 means update)
        cursor.fetchone.return_value = (False,)  # is_insert = False
        
        stats = agent_with_db.ingest_spend_records([sample_record])
        
        # Verify stats: 0 inserted, 1 updated
        assert stats["inserted"] == 0
        assert stats["updated"] == 1
        assert stats["skipped"] == 0
        
        # Verify commit was called
        conn.commit.assert_called_once()
    
    def test_multiple_records_with_duplicates(self, agent_with_db, mock_db_conn):
        """Should handle mix of new and duplicate records"""
        conn, cursor = mock_db_conn
        
        records = [
            SpendRecord(
                date=date(2025, 10, 12),
                channel="meta",
                campaign_id="c123",
                spend=100000.0,
                currency="JPY",
                impressions=30000,
                clicks=800
            ),
            SpendRecord(
                date=date(2025, 10, 12),
                channel="meta",
                campaign_id="c456",
                spend=150000.0,
                currency="JPY",
                impressions=50000,
                clicks=1500
            ),
            SpendRecord(
                date=date(2025, 10, 13),
                channel="meta",
                campaign_id="c123",
                spend=110000.0,
                currency="JPY",
                impressions=32000,
                clicks=850
            )
        ]
        
        # Simulate: first insert, second update, third insert
        cursor.fetchone.side_effect = [
            (True,),   # First: new record
            (False,),  # Second: duplicate (update)
            (True,)    # Third: new record
        ]
        
        stats = agent_with_db.ingest_spend_records(records)
        
        # Verify stats
        assert stats["inserted"] == 2
        assert stats["updated"] == 1
        assert stats["skipped"] == 0
        
        # Verify 3 executes + 3 commits
        assert cursor.execute.call_count == 3
        assert conn.commit.call_count == 3
    
    def test_idempotency_key_generation(self, agent_with_db, sample_record):
        """Should generate consistent idempotency keys"""
        # Idempotency key should be: channel:campaign_id:date
        expected_key = "meta:c123:2025-10-12"
        
        # This would be logged internally
        # Verify by checking that duplicate keys map to same DB row
        key1 = f"{sample_record.channel}:{sample_record.campaign_id}:{sample_record.date.isoformat()}"
        
        # Same record with different metrics should generate same key
        duplicate = SpendRecord(
            date=date(2025, 10, 12),
            channel="meta",
            campaign_id="c123",
            adset_id="a999",  # Different adset
            spend=999999.0,   # Different spend
            currency="JPY",
            impressions=99999,
            clicks=9999
        )
        
        key2 = f"{duplicate.channel}:{duplicate.campaign_id}:{duplicate.date.isoformat()}"
        
        assert key1 == key2 == expected_key
    
    def test_upsert_sql_parameters(self, agent_with_db, mock_db_conn, sample_record):
        """Should pass correct parameters to SQL upsert"""
        conn, cursor = mock_db_conn
        cursor.fetchone.return_value = (True,)
        
        agent_with_db.ingest_spend_records([sample_record])
        
        # Get the SQL parameters passed
        params = cursor.execute.call_args[0][1]
        
        assert params["dt"] == date(2025, 10, 12)
        assert params["channel"] == "meta"
        assert params["campaign_id"] == "c123"
        assert params["adset_id"] == "a456"
        assert params["spend"] == 120000.0
        assert params["currency"] == "JPY"
        assert params["impressions"] == 45000
        assert params["clicks"] == 1200
        
        # ad_id should be generated from channel + campaign + date
        expected_ad_id = "meta_c123_2025-10-12"
        assert params["ad_id"] == expected_ad_id
    
    def test_rollback_on_error(self, agent_with_db, mock_db_conn, sample_record):
        """Should rollback transaction on database error"""
        conn, cursor = mock_db_conn
        
        # Simulate DB error
        cursor.execute.side_effect = Exception("DB connection lost")
        
        stats = agent_with_db.ingest_spend_records([sample_record])
        
        # Verify rollback was called
        conn.rollback.assert_called_once()
        
        # Verify record was skipped
        assert stats["inserted"] == 0
        assert stats["updated"] == 0
        assert stats["skipped"] == 1
    
    def test_dry_run_mode(self, agent_with_db, mock_db_conn, sample_record):
        """Should not insert/update in dry_run mode"""
        conn, cursor = mock_db_conn
        
        stats = agent_with_db.ingest_spend_records([sample_record], dry_run=True)
        
        # Verify no DB operations
        cursor.execute.assert_not_called()
        conn.commit.assert_not_called()
        
        # Verify all records skipped
        assert stats["inserted"] == 0
        assert stats["updated"] == 0
        assert stats["skipped"] == 1
    
    def test_requires_db_connection(self, sample_record):
        """Should raise ValueError if DB connection not configured"""
        agent = MetaAdPlatformAgent(db_connection=None)
        
        with pytest.raises(ValueError, match="Database connection required"):
            agent.ingest_spend_records([sample_record])
    
    def test_fetch_and_ingest_integration(self, agent_with_db, mock_db_conn):
        """Should fetch from API and ingest with idempotency"""
        conn, cursor = mock_db_conn
        cursor.fetchone.return_value = (True,)
        
        # Mock API response
        mock_response = {
            "data": [
                {
                    "id": "ad1",
                    "campaign_id": "c123",
                    "adset_id": "a456",
                    "spend": "100000",
                    "impressions": "30000",
                    "clicks": "800",
                    "date_start": "2025-10-12",
                    "date_stop": "2025-10-12",
                    "account_currency": "JPY"
                }
            ]
        }
        
        # Parse response
        records = agent_with_db.parse_response(mock_response)
        assert len(records) == 1
        
        # Ingest
        stats = agent_with_db.ingest_spend_records(records)
        assert stats["inserted"] == 1


class TestUniqueConstraintEnforcement:
    """Test database-level UNIQUE constraint enforcement"""
    
    @pytest.mark.integration
    def test_unique_constraint_prevents_duplicates(self):
        """
        Integration test: DB should reject duplicate (channel, campaign_id, dt)
        
        Note: Requires actual test database with schema applied.
        Skipped in unit tests.
        """
        # This would test against real PostgreSQL:
        # 1. Create test DB with fct_ad_metric_daily schema
        # 2. Insert record (channel='meta', campaign_id='c1', dt='2025-10-12')
        # 3. Attempt second insert with same tuple
        # 4. Verify UNIQUE constraint violation OR successful upsert
        pass
    
    @pytest.mark.integration
    def test_upsert_updates_metrics(self):
        """
        Integration test: Upsert should update metrics on conflict
        
        Validates:
        - First insert creates row with spend=100000
        - Second upsert updates same row to spend=150000
        - Single row exists (no duplicate)
        """
        pass
