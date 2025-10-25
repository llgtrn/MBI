"""
Pacing Agent - Asset-level Pause Enforcement & Idempotency
"""
from datetime import date, datetime
from typing import List, Dict, Optional
from pydantic import BaseModel, Field, validator
import hashlib
from prometheus_client import Counter
import redis


# Prometheus metrics
pacing_asset_paused_counter = Counter(
    'pacing_asset_paused_total',
    'Total assets paused due to overpacing',
    ['campaign_id', 'asset_id', 'reason']
)

pacing_asset_resumed_counter = Counter(
    'pacing_asset_resumed_total',
    'Total assets resumed after pacing normalized',
    ['campaign_id', 'asset_id']
)


class PacingStatus(BaseModel):
    """Asset pacing status with action"""
    asset_id: str
    campaign_id: str
    date: date
    pacing_percent: float = Field(..., description="Current pacing percentage (spend/budget * 100)")
    action: str = Field(..., description="Action taken: pause|resume|none")
    timestamp: datetime
    spend_today: Optional[float] = None
    budget_daily: Optional[float] = None
    
    @validator('action')
    def validate_action(cls, v):
        """Validate action is enum"""
        if v not in ['pause', 'resume', 'none']:
            raise ValueError(f"action must be pause|resume|none, got: {v}")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "asset_id": "asset_123",
                "campaign_id": "camp_456",
                "date": "2025-10-19",
                "pacing_percent": 120.0,
                "action": "pause",
                "timestamp": "2025-10-19T15:30:00Z",
                "spend_today": 12000,
                "budget_daily": 10000
            }
        }


class PacingAgent:
    """
    Pacing agent with:
    - Asset-level granular pause/resume
    - Idempotency via Redis deduplication
    - Configurable thresholds
    """
    
    def __init__(
        self,
        feature_store=None,
        ad_platform_api=None,
        redis_client=None,
        pause_threshold: float = 110.0,
        resume_threshold: float = 95.0
    ):
        self.feature_store = feature_store or get_feature_store()
        self.ad_platform_api = ad_platform_api or get_ad_platform_api()
        self.redis_client = redis_client or get_redis_client()
        self.pause_threshold = pause_threshold
        self.resume_threshold = resume_threshold
    
    def check_asset_pacing(
        self,
        campaign_id: str,
        date: date,
        pause_threshold: Optional[float] = None,
        resume_threshold: Optional[float] = None
    ) -> List[PacingStatus]:
        """
        Check asset-level pacing and take selective pause/resume actions
        
        Args:
            campaign_id: Campaign to check
            date: Date to check (default today)
            pause_threshold: Pacing % threshold for pause (default 110%)
            resume_threshold: Pacing % threshold for resume (default 95%)
        
        Returns:
            List of PacingStatus for each asset
        """
        pause_threshold = pause_threshold or self.pause_threshold
        resume_threshold = resume_threshold or self.resume_threshold
        
        # Get asset pacing data
        asset_pacing = self.feature_store.get_asset_pacing(campaign_id, date)
        asset_statuses = self.feature_store.get_asset_status(campaign_id)
        
        results = []
        
        for asset_id, pacing_data in asset_pacing.items():
            pacing_percent = pacing_data['pacing_percent']
            spend_today = pacing_data['spend_today']
            budget_daily = pacing_data['budget_daily']
            current_status = asset_statuses.get(asset_id, 'active')
            
            action = 'none'
            
            # Determine action
            if pacing_percent >= pause_threshold and current_status == 'active':
                action = 'pause'
            elif pacing_percent <= resume_threshold and current_status == 'paused':
                action = 'resume'
            
            # Execute action with idempotency
            if action in ['pause', 'resume']:
                executed = self._execute_action_idempotent(
                    asset_id=asset_id,
                    action=action,
                    date=date,
                    campaign_id=campaign_id
                )
                
                if not executed:
                    # Already processed, change action to 'none'
                    action = 'none'
            
            # Create status
            status = PacingStatus(
                asset_id=asset_id,
                campaign_id=campaign_id,
                date=date,
                pacing_percent=pacing_percent,
                action=action,
                timestamp=datetime.utcnow(),
                spend_today=spend_today,
                budget_daily=budget_daily
            )
            
            results.append(status)
        
        return results
    
    def _execute_action_idempotent(
        self,
        asset_id: str,
        action: str,
        date: date,
        campaign_id: str
    ) -> bool:
        """
        Execute pause/resume action with idempotency via Redis
        
        Returns:
            True if action executed, False if already processed
        """
        # Compute action ID for deduplication
        action_id = self._compute_action_id(asset_id, action, date)
        
        # Check Redis cache (24h TTL)
        cache_key = f"pacing_action:{action_id}"
        cached = self.redis_client.get(cache_key)
        
        if cached:
            # Already processed
            return False
        
        # Execute action
        if action == 'pause':
            self.ad_platform_api.pause_asset(asset_id)
            
            # Emit metric
            pacing_asset_paused_counter.labels(
                campaign_id=campaign_id,
                asset_id=asset_id,
                reason='overpacing'
            ).inc()
        
        elif action == 'resume':
            self.ad_platform_api.resume_asset(asset_id)
            
            # Emit metric
            pacing_asset_resumed_counter.labels(
                campaign_id=campaign_id,
                asset_id=asset_id
            ).inc()
        
        # Store in Redis with 24h TTL
        self.redis_client.setex(
            cache_key,
            86400,  # 24 hours
            'processed'
        )
        
        return True
    
    def _compute_action_id(self, asset_id: str, action: str, date: date) -> str:
        """
        Compute deterministic action ID for idempotency
        
        Formula: SHA256(asset_id + action + date)
        """
        input_str = f"{asset_id}|{action}|{date.isoformat()}"
        return hashlib.sha256(input_str.encode()).hexdigest()


def get_feature_store():
    """Get feature store instance (placeholder)"""
    return MockFeatureStore()


def get_ad_platform_api():
    """Get ad platform API instance (placeholder)"""
    return MockAdPlatformAPI()


def get_redis_client():
    """Get Redis client instance (placeholder)"""
    return MockRedisClient()


class MockFeatureStore:
    """Mock feature store for testing"""
    
    def get_asset_pacing(self, campaign_id: str, date: date) -> Dict:
        """Get asset pacing data"""
        return {}
    
    def get_asset_status(self, campaign_id: str) -> Dict:
        """Get current asset statuses"""
        return {}


class MockAdPlatformAPI:
    """Mock ad platform API for testing"""
    
    def pause_asset(self, asset_id: str):
        """Pause asset"""
        pass
    
    def resume_asset(self, asset_id: str):
        """Resume asset"""
        pass


class MockRedisClient:
    """Mock Redis client for testing"""
    
    def __init__(self):
        self._cache = {}
    
    def get(self, key: str):
        """Get from cache"""
        return self._cache.get(key)
    
    def setex(self, key: str, ttl: int, value: str):
        """Set with expiry"""
        self._cache[key] = value.encode() if isinstance(value, str) else value


# Redis client singleton (production would use redis.Redis)
redis_client = MockRedisClient()
