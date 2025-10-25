-- fct_ad_metric_daily: Daily ad metrics with idempotency guarantees
-- 
-- Purpose: Store daily advertising metrics from multiple platforms
-- with guaranteed uniqueness and upsert behavior
--
-- Idempotency: UNIQUE constraint on (channel, campaign_id, dt)
-- ensures duplicate insertions are handled gracefully via ON CONFLICT

CREATE TABLE IF NOT EXISTS fct_ad_metric_daily (
    -- Composite primary key
    dt DATE NOT NULL,
    channel VARCHAR(50) NOT NULL,
    campaign_id VARCHAR(255) NOT NULL,
    
    -- Optional hierarchical identifiers
    adset_id VARCHAR(255),
    ad_id VARCHAR(255),
    
    -- Core metrics
    impressions BIGINT NOT NULL DEFAULT 0,
    clicks BIGINT NOT NULL DEFAULT 0,
    spend DECIMAL(12, 2) NOT NULL DEFAULT 0.00,
    currency VARCHAR(3) NOT NULL DEFAULT 'JPY',
    
    -- Conversion metrics (from platform API)
    conversions INT DEFAULT 0,
    conversion_value DECIMAL(12, 2) DEFAULT 0.00,
    
    -- Derived metrics (computed)
    ctr DECIMAL(6, 4) GENERATED ALWAYS AS (
        CASE 
            WHEN impressions > 0 THEN clicks::DECIMAL / impressions
            ELSE 0
        END
    ) STORED,
    cpc DECIMAL(10, 2) GENERATED ALWAYS AS (
        CASE 
            WHEN clicks > 0 THEN spend / clicks
            ELSE 0
        END
    ) STORED,
    cpa DECIMAL(10, 2) GENERATED ALWAYS AS (
        CASE 
            WHEN conversions > 0 THEN spend / conversions
            ELSE 0
        END
    ) STORED,
    roas DECIMAL(10, 4) GENERATED ALWAYS AS (
        CASE 
            WHEN spend > 0 THEN conversion_value / spend
            ELSE 0
        END
    ) STORED,
    
    -- Metadata
    ingested_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    source_system VARCHAR(50) NOT NULL,  -- meta_api, google_ads_api, etc.
    idempotency_key VARCHAR(255) GENERATED ALWAYS AS (
        channel || '|' || campaign_id || '|' || dt
    ) STORED,
    
    -- Constraints
    CONSTRAINT pk_ad_metric_daily PRIMARY KEY (dt, channel, campaign_id),
    CONSTRAINT uq_ad_metric_idempotency UNIQUE (channel, campaign_id, dt),
    CONSTRAINT chk_positive_metrics CHECK (
        impressions >= 0 AND 
        clicks >= 0 AND 
        spend >= 0 AND
        conversions >= 0 AND
        conversion_value >= 0
    ),
    CONSTRAINT chk_valid_channel CHECK (
        channel IN ('meta', 'google', 'tiktok', 'youtube', 'linkedin', 'twitter', 'snapchat')
    ),
    CONSTRAINT chk_clicks_le_impressions CHECK (clicks <= impressions)
) PARTITION BY RANGE (dt);

-- Partitioning for performance (monthly partitions)
CREATE TABLE fct_ad_metric_daily_y2025m01 PARTITION OF fct_ad_metric_daily
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

CREATE TABLE fct_ad_metric_daily_y2025m02 PARTITION OF fct_ad_metric_daily
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');

CREATE TABLE fct_ad_metric_daily_y2025m03 PARTITION OF fct_ad_metric_daily
    FOR VALUES FROM ('2025-03-01') TO ('2025-04-01');

CREATE TABLE fct_ad_metric_daily_y2025m04 PARTITION OF fct_ad_metric_daily
    FOR VALUES FROM ('2025-04-01') TO ('2025-05-01');

CREATE TABLE fct_ad_metric_daily_y2025m05 PARTITION OF fct_ad_metric_daily
    FOR VALUES FROM ('2025-05-01') TO ('2025-06-01');

CREATE TABLE fct_ad_metric_daily_y2025m06 PARTITION OF fct_ad_metric_daily
    FOR VALUES FROM ('2025-06-01') TO ('2025-07-01');

CREATE TABLE fct_ad_metric_daily_y2025m07 PARTITION OF fct_ad_metric_daily
    FOR VALUES FROM ('2025-07-01') TO ('2025-08-01');

CREATE TABLE fct_ad_metric_daily_y2025m08 PARTITION OF fct_ad_metric_daily
    FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');

CREATE TABLE fct_ad_metric_daily_y2025m09 PARTITION OF fct_ad_metric_daily
    FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');

CREATE TABLE fct_ad_metric_daily_y2025m10 PARTITION OF fct_ad_metric_daily
    FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');

CREATE TABLE fct_ad_metric_daily_y2025m11 PARTITION OF fct_ad_metric_daily
    FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');

CREATE TABLE fct_ad_metric_daily_y2025m12 PARTITION OF fct_ad_metric_daily
    FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');

-- Indexes for query performance
CREATE INDEX idx_ad_metric_channel_dt ON fct_ad_metric_daily (channel, dt DESC);
CREATE INDEX idx_ad_metric_campaign_dt ON fct_ad_metric_daily (campaign_id, dt DESC);
CREATE INDEX idx_ad_metric_ingested_at ON fct_ad_metric_daily (ingested_at DESC);
CREATE INDEX idx_ad_metric_spend ON fct_ad_metric_daily (spend DESC) WHERE spend > 0;

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_ad_metric_updated_at
    BEFORE UPDATE ON fct_ad_metric_daily
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- View for latest metrics (materialized for performance)
CREATE MATERIALIZED VIEW mv_ad_metric_latest_7d AS
SELECT 
    channel,
    campaign_id,
    SUM(impressions) as total_impressions,
    SUM(clicks) as total_clicks,
    SUM(spend) as total_spend,
    SUM(conversions) as total_conversions,
    SUM(conversion_value) as total_conversion_value,
    AVG(ctr) as avg_ctr,
    AVG(cpc) as avg_cpc,
    AVG(cpa) as avg_cpa,
    CASE 
        WHEN SUM(spend) > 0 THEN SUM(conversion_value) / SUM(spend)
        ELSE 0 
    END as roas,
    MIN(dt) as period_start,
    MAX(dt) as period_end
FROM fct_ad_metric_daily
WHERE dt >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY channel, campaign_id;

CREATE UNIQUE INDEX idx_mv_ad_metric_latest_7d ON mv_ad_metric_latest_7d (channel, campaign_id);

-- Refresh schedule for materialized view (via cron or scheduler)
-- Recommended: REFRESH MATERIALIZED VIEW CONCURRENTLY mv_ad_metric_latest_7d;

-- Comments for documentation
COMMENT ON TABLE fct_ad_metric_daily IS 'Daily advertising metrics with idempotency guarantees via UNIQUE constraint';
COMMENT ON COLUMN fct_ad_metric_daily.idempotency_key IS 'Generated key for deduplication: channel|campaign_id|dt';
COMMENT ON CONSTRAINT uq_ad_metric_idempotency ON fct_ad_metric_daily IS 'Ensures no duplicate records for same (channel, campaign_id, date) tuple';
COMMENT ON COLUMN fct_ad_metric_daily.ctr IS 'Click-through rate: clicks / impressions';
COMMENT ON COLUMN fct_ad_metric_daily.cpc IS 'Cost per click: spend / clicks';
COMMENT ON COLUMN fct_ad_metric_daily.cpa IS 'Cost per acquisition: spend / conversions';
COMMENT ON COLUMN fct_ad_metric_daily.roas IS 'Return on ad spend: conversion_value / spend';
