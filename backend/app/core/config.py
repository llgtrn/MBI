"""
Application Configuration
Loads settings from environment variables
"""

from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # ========================================================================
    # APPLICATION
    # ========================================================================
    ENV: str = Field(default="development", description="Environment name")
    DEBUG: bool = Field(default=True, description="Debug mode")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    SECRET_KEY: str = Field(
        default="change-this-in-production",
        description="Secret key for JWT and encryption"
    )
    API_VERSION: str = Field(default="v2", description="API version")
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:3001"],
        description="Allowed CORS origins"
    )
    
    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    # ========================================================================
    # DATABASE
    # ========================================================================
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://mbi_user:mbi_password@localhost:5432/mbi_db",
        description="PostgreSQL connection URL"
    )
    DATABASE_POOL_SIZE: int = Field(default=20, description="DB pool size")
    DATABASE_MAX_OVERFLOW: int = Field(default=40, description="DB max overflow")
    
    # BigQuery (optional for MVP)
    BIGQUERY_PROJECT_ID: Optional[str] = Field(default=None)
    BIGQUERY_DATASET_RAW: str = Field(default="MBI_raw")
    BIGQUERY_DATASET_CORE: str = Field(default="MBI_core")
    BIGQUERY_DATASET_MARTS: str = Field(default="MBI_marts")
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = Field(default=None)
    
    # ========================================================================
    # REDIS
    # ========================================================================
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    REDIS_MAX_CONNECTIONS: int = Field(default=50)
    REDIS_SOCKET_TIMEOUT: int = Field(default=5)
    REDIS_SOCKET_CONNECT_TIMEOUT: int = Field(default=5)
    REDIS_FEATURE_STORE_URL: str = Field(
        default="redis://localhost:6379/1",
        description="Redis feature store DB"
    )
    
    # ========================================================================
    # KAFKA
    # ========================================================================
    KAFKA_BOOTSTRAP_SERVERS: str = Field(
        default="localhost:9092",
        description="Kafka bootstrap servers"
    )
    KAFKA_CONSUMER_GROUP: str = Field(default="mbi-consumers")
    KAFKA_AUTO_OFFSET_RESET: str = Field(default="earliest")
    
    # Topics
    KAFKA_TOPIC_SPEND_INGESTED: str = Field(default="spend-ingested")
    KAFKA_TOPIC_ORDER_COMPLETED: str = Field(default="order-completed")
    KAFKA_TOPIC_SESSION_TRACKED: str = Field(default="session-tracked")
    KAFKA_TOPIC_CRISIS_DETECTED: str = Field(default="crisis-detected")
    KAFKA_TOPIC_LEAD_SCORED: str = Field(default="lead-scored")
    
    # ========================================================================
    # LLM PROVIDERS
    # ========================================================================
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None)
    ANTHROPIC_DEFAULT_MODEL: str = Field(default="claude-sonnet-4-20250514")
    ANTHROPIC_MAX_TOKENS: int = Field(default=2000)
    ANTHROPIC_TEMPERATURE: float = Field(default=0.2)
    
    OPENAI_API_KEY: Optional[str] = Field(default=None)
    OPENAI_DEFAULT_MODEL: str = Field(default="gpt-4-turbo-preview")
    OPENAI_ORGANIZATION_ID: Optional[str] = Field(default=None)
    
    LLM_CACHE_TTL_HOURS: int = Field(default=24)
    LLM_MAX_RETRIES: int = Field(default=3)
    
    # ========================================================================
    # VECTOR DATABASE
    # ========================================================================
    CHROMA_HOST: str = Field(default="localhost")
    CHROMA_PORT: int = Field(default=8001)
    CHROMA_COLLECTION_SSOT: str = Field(default="mbi_ssot")
    EMBEDDING_MODEL: str = Field(default="BAAI/bge-m3")
    
    # ========================================================================
    # AD PLATFORMS
    # ========================================================================
    # Meta
    META_APP_ID: Optional[str] = Field(default=None)
    META_APP_SECRET: Optional[str] = Field(default=None)
    META_ACCESS_TOKEN: Optional[str] = Field(default=None)
    META_AD_ACCOUNT_ID: Optional[str] = Field(default=None)
    
    # Google Ads
    GOOGLE_ADS_DEVELOPER_TOKEN: Optional[str] = Field(default=None)
    GOOGLE_ADS_CLIENT_ID: Optional[str] = Field(default=None)
    GOOGLE_ADS_CLIENT_SECRET: Optional[str] = Field(default=None)
    GOOGLE_ADS_REFRESH_TOKEN: Optional[str] = Field(default=None)
    GOOGLE_ADS_CUSTOMER_ID: Optional[str] = Field(default=None)
    
    # TikTok
    TIKTOK_ACCESS_TOKEN: Optional[str] = Field(default=None)
    TIKTOK_ADVERTISER_ID: Optional[str] = Field(default=None)
    
    # ========================================================================
    # ANALYTICS
    # ========================================================================
    GA4_PROPERTY_ID: Optional[str] = Field(default=None)
    GA4_CREDENTIALS_PATH: Optional[str] = Field(default=None)
    
    # ========================================================================
    # E-COMMERCE
    # ========================================================================
    SHOPIFY_SHOP_NAME: Optional[str] = Field(default=None)
    SHOPIFY_ACCESS_TOKEN: Optional[str] = Field(default=None)
    SHOPIFY_API_VERSION: str = Field(default="2024-01")
    SHOPIFY_WEBHOOK_SECRET: Optional[str] = Field(default=None)
    
    # ========================================================================
    # AUTHENTICATION
    # ========================================================================
    JWT_SECRET_KEY: str = Field(
        default="change-this-in-production",
        description="JWT secret key"
    )
    JWT_ALGORITHM: str = Field(default="HS256")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=60)
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=30)
    
    # ========================================================================
    # EMAIL
    # ========================================================================
    SMTP_HOST: str = Field(default="smtp.gmail.com")
    SMTP_PORT: int = Field(default=587)
    SMTP_USER: Optional[str] = Field(default=None)
    SMTP_PASSWORD: Optional[str] = Field(default=None)
    SMTP_FROM_EMAIL: str = Field(default="noreply@company.com")
    SMTP_FROM_NAME: str = Field(default="MBI System")
    
    # ========================================================================
    # NOTIFICATIONS
    # ========================================================================
    SLACK_WEBHOOK_URL: Optional[str] = Field(default=None)
    SLACK_ALERT_CHANNEL: str = Field(default="#mbi-alerts")
    SLACK_BOT_TOKEN: Optional[str] = Field(default=None)
    
    PAGERDUTY_API_KEY: Optional[str] = Field(default=None)
    PAGERDUTY_SERVICE_ID: Optional[str] = Field(default=None)
    
    # ========================================================================
    # FEATURE FLAGS
    # ========================================================================
    FEATURE_MMM_ENABLED: bool = Field(default=True)
    FEATURE_MTA_ENABLED: bool = Field(default=True)
    FEATURE_LLM_ENABLED: bool = Field(default=True)
    FEATURE_CRISIS_DETECTION_ENABLED: bool = Field(default=True)
    FEATURE_CREATIVE_GENERATION_ENABLED: bool = Field(default=True)
    FEATURE_AUTO_BUDGET_ALLOCATION: bool = Field(default=False)
    FEATURE_AUTO_CREATIVE_ROTATION: bool = Field(default=True)
    
    # ========================================================================
    # RATE LIMITING
    # ========================================================================
    RATE_LIMIT_ENABLED: bool = Field(default=True)
    RATE_LIMIT_PER_MINUTE: int = Field(default=100)
    RATE_LIMIT_PER_HOUR: int = Field(default=1000)
    RATE_LIMIT_PER_DAY: int = Field(default=10000)
    
    # ========================================================================
    # DATA RETENTION
    # ========================================================================
    PII_HASH_SALT: str = Field(
        default="change-this-in-production",
        description="Salt for hashing PII"
    )
    DATA_RETENTION_DAYS: int = Field(default=90)
    AUTO_DELETE_ENABLED: bool = Field(default=True)
    COOKIE_CONSENT_REQUIRED: bool = Field(default=True)
    ANALYTICS_OPT_OUT_ENABLED: bool = Field(default=True)
    
    # ========================================================================
    # MODEL SETTINGS
    # ========================================================================
    # MMM
    MMM_LOOKBACK_WEEKS: int = Field(default=52)
    MMM_RETRAIN_SCHEDULE: str = Field(default="weekly")
    MMM_MIN_DATA_POINTS: int = Field(default=13)
    
    # MTA
    MTA_LOOKBACK_DAYS: int = Field(default=30)
    MTA_MIN_TOUCHPOINTS: int = Field(default=1)
    MTA_MAX_TOUCHPOINTS: int = Field(default=10)
    MTA_METHOD: str = Field(default="markov")
    
    # Lead Scoring
    LEAD_SCORING_MODEL_PATH: str = Field(default="./models/lead_scoring_v1.pkl")
    LEAD_SCORING_THRESHOLD: float = Field(default=0.7)
    LEAD_SCORING_RETRAIN_DAYS: int = Field(default=7)
    
    # Creative Intelligence
    CREATIVE_FATIGUE_THRESHOLD: float = Field(default=0.7)
    CREATIVE_ANALYSIS_BATCH_SIZE: int = Field(default=50)
    
    # ========================================================================
    # BUDGET & CONSTRAINTS
    # ========================================================================
    DEFAULT_WEEKLY_BUDGET: float = Field(default=500000)
    MIN_CHANNEL_BUDGET_PCT: float = Field(default=0.05)
    MAX_CHANNEL_BUDGET_PCT: float = Field(default=0.60)
    MAX_BUDGET_SHIFT_PER_WEEK: float = Field(default=0.25)
    
    TARGET_ROAS: float = Field(default=2.5)
    TARGET_CAC: float = Field(default=5000)
    TARGET_CAC_CURRENCY: str = Field(default="JPY")
    
    # ========================================================================
    # MONITORING
    # ========================================================================
    SENTRY_DSN: Optional[str] = Field(default=None)
    SENTRY_ENVIRONMENT: str = Field(default="development")
    SENTRY_TRACES_SAMPLE_RATE: float = Field(default=0.1)
    
    OTEL_ENABLED: bool = Field(default=True)
    OTEL_EXPORTER_OTLP_ENDPOINT: str = Field(default="http://localhost:4317")
    OTEL_SERVICE_NAME: str = Field(default="mbi-backend")
    
    # ========================================================================
    # COMPUTED PROPERTIES
    # ========================================================================
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.ENV.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.ENV.lower() == "development"
    
    @property
    def kafka_bootstrap_servers_list(self) -> List[str]:
        """Get Kafka bootstrap servers as list"""
        return [s.strip() for s in self.KAFKA_BOOTSTRAP_SERVERS.split(",")]


# Create global settings instance
settings = Settings()
