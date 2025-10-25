"""
Crisis Detection Agent - Tier-1 Source Validation and Velocity Baseline
Component: C11_CrisisDetection
Features: Tier-1 domain risk capping (Q_018), 24h velocity baseline (Q_019, A_022)
Risk Gates: ≥2 tier-1 sources → risk_score ≤ 0.6
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np

from prometheus_client import Gauge, Counter
from pydantic import BaseModel, Field

from core.config import Config


# Prometheus metrics
crisis_velocity_baseline_mean = Gauge(
    'crisis_velocity_baseline_mean',
    'Rolling 24h velocity baseline mean',
    ['topic_id']
)

crisis_velocity_baseline_std = Gauge(
    'crisis_velocity_baseline_std',
    'Rolling 24h velocity baseline std deviation',
    ['topic_id']
)

crisis_tier1_sources_total = Counter(
    'crisis_tier1_sources_total',
    'Total tier-1 sources detected',
    ['topic_id']
)


class TierLevel(str, Enum):
    """Source tier classification (A_022)"""
    TIER1 = "tier1"  # Reuters, AP, Bloomberg, etc.
    TIER2 = "tier2"  # Major news outlets
    TIER3 = "tier3"  # Social media, forums


class CrisisConfig(BaseModel):
    """Crisis detection configuration (Q_018, Q_019, A_022)"""
    tier1_domains: List[str] = Field(
        default_factory=lambda: [
            "reuters.com",
            "apnews.com",
            "bloomberg.com",
            "wsj.com",
            "ft.com"
        ],
        description="Tier-1 trusted news domains"
    )
    tier1_risk_cap: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Risk score cap when ≥2 tier-1 sources present (Q_018)"
    )
    baseline_window_hours: int = Field(
        default=24,
        description="Velocity baseline window in hours (Q_019)"
    )
    min_baseline_data_points: int = Field(
        default=12,
        description="Minimum data points for reliable baseline"
    )


@dataclass
class CrisisSource:
    """Crisis source with tier classification"""
    id: str
    url: str
    title: str = ""
    text: str = ""
    tier: Optional[TierLevel] = None
    domain: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class VelocityBaseline:
    """Velocity baseline statistics (Q_019)"""
    topic_id: str
    mean: float
    std: float
    window_hours: int
    data_points: int
    timestamp: datetime
    insufficient_data: bool = False
    
    def z_score(self, velocity: float) -> float:
        """Compute z-score for given velocity"""
        if self.std == 0:
            return 0.0
        return (velocity - self.mean) / self.std


@dataclass
class NormalizedVelocity:
    """Velocity normalized by baseline"""
    raw_velocity: float
    z_score: float
    sigma_above_mean: float
    baseline_mean: float
    baseline_std: float


@dataclass
class CrisisBrief:
    """Crisis detection result"""
    topic_id: str
    risk_score: float
    tier1_source_count: int
    risk_cap_applied: bool
    velocity_normalized: Optional[NormalizedVelocity]
    sources: List[CrisisSource]
    metadata: Dict
    timestamp: datetime = field(default_factory=datetime.utcnow)


class CrisisDetectionAgent:
    """
    Crisis Detection Agent with Tier-1 Validation and Baseline Normalization
    
    Features (Q_018, Q_019, A_022):
    - Tier-1 domain classification from official_domains config
    - Risk score cap at 0.6 when ≥2 tier-1 sources present
    - 24h rolling velocity baseline (mean/std)
    - Z-score normalization for velocity anomalies
    
    Risk Gates (A_022):
    - ≥2 tier-1 sources → risk_score ≤ 0.6
    - Velocity normalized by 24h baseline
    - Quarterly tier-1 domain audit required
    """
    
    def __init__(self, config: Optional[CrisisConfig] = None):
        self.config = config or CrisisConfig()
        
        # Baseline storage (in production: Redis/Memcache)
        self.baselines: Dict[str, VelocityBaseline] = {}
    
    def classify_source_tiers(
        self,
        sources: List[CrisisSource]
    ) -> List[CrisisSource]:
        """
        Classify sources by tier level (A_022)
        
        Tier-1: Official trusted news domains (config.tier1_domains)
        Tier-2: Other news outlets
        Tier-3: Social media, forums, blogs
        """
        for source in sources:
            if not source.domain:
                source.domain = self._extract_domain(source.url)
            
            # Classify tier
            if source.domain in self.config.tier1_domains:
                source.tier = TierLevel.TIER1
            elif self._is_news_outlet(source.domain):
                source.tier = TierLevel.TIER2
            else:
                source.tier = TierLevel.TIER3
        
        return sources
    
    async def verify_crisis(
        self,
        topic_id: str,
        sources: List[CrisisSource],
        velocity: float,
        brand: Optional[str] = None
    ) -> CrisisBrief:
        """
        Verify crisis with tier-1 validation and baseline normalization
        
        Steps:
        1. Classify sources by tier
        2. Count tier-1 sources
        3. Compute risk score
        4. Apply tier-1 cap if ≥2 tier-1 sources (Q_018)
        5. Normalize velocity by baseline (Q_019)
        
        Args:
            topic_id: Topic identifier
            sources: List of crisis sources
            velocity: Current velocity (mentions/hour or similar)
            brand: Brand name (optional)
            
        Returns:
            CrisisBrief with risk_score, tier-1 count, baseline normalization
        """
        # 1. Classify sources
        classified_sources = self.classify_source_tiers(sources)
        
        # 2. Count tier-1 sources
        tier1_count = sum(1 for s in classified_sources if s.tier == TierLevel.TIER1)
        
        # Emit metric
        if tier1_count > 0:
            crisis_tier1_sources_total.labels(topic_id=topic_id).inc(tier1_count)
        
        # 3. Compute base risk score (from velocity, source diversity, etc.)
        base_risk = self._compute_base_risk(
            velocity=velocity,
            sources=classified_sources
        )
        
        # 4. Apply tier-1 risk cap (Q_018, A_022)
        risk_cap_applied = False
        final_risk = base_risk
        
        if tier1_count >= 2:
            # Cap at configured limit (default 0.6)
            if base_risk > self.config.tier1_risk_cap:
                final_risk = self.config.tier1_risk_cap
                risk_cap_applied = True
        
        # 5. Normalize velocity by baseline (Q_019)
        velocity_normalized = None
        if topic_id in self.baselines:
            baseline = self.baselines[topic_id]
            velocity_normalized = self.normalize_velocity(velocity, baseline)
        
        # Build crisis brief
        return CrisisBrief(
            topic_id=topic_id,
            risk_score=final_risk,
            tier1_source_count=tier1_count,
            risk_cap_applied=risk_cap_applied,
            velocity_normalized=velocity_normalized,
            sources=classified_sources,
            metadata={
                "base_risk": base_risk,
                "tier1_sources": [s.url for s in classified_sources if s.tier == TierLevel.TIER1],
                "velocity": velocity
            }
        )
    
    async def compute_velocity_baseline(
        self,
        topic_id: str,
        velocity_data: List[Dict]
    ) -> VelocityBaseline:
        """
        Compute 24h rolling velocity baseline (Q_019)
        
        Args:
            topic_id: Topic identifier
            velocity_data: List of dicts with 'timestamp' and 'velocity' keys
            
        Returns:
            VelocityBaseline with mean/std over 24h window
        """
        # Filter to last 24 hours
        cutoff = datetime.utcnow() - timedelta(hours=self.config.baseline_window_hours)
        recent_data = [
            d for d in velocity_data
            if d.get('timestamp', datetime.min) > cutoff
        ]
        
        if not recent_data:
            # No data: return zero baseline
            return VelocityBaseline(
                topic_id=topic_id,
                mean=0.0,
                std=0.0,
                window_hours=self.config.baseline_window_hours,
                data_points=0,
                timestamp=datetime.utcnow(),
                insufficient_data=True
            )
        
        # Compute statistics
        velocities = [d['velocity'] for d in recent_data]
        mean = np.mean(velocities)
        std = np.std(velocities)
        
        baseline = VelocityBaseline(
            topic_id=topic_id,
            mean=float(mean),
            std=float(std),
            window_hours=self.config.baseline_window_hours,
            data_points=len(recent_data),
            timestamp=datetime.utcnow(),
            insufficient_data=len(recent_data) < self.config.min_baseline_data_points
        )
        
        # Emit metrics (Q_019)
        crisis_velocity_baseline_mean.labels(topic_id=topic_id).set(baseline.mean)
        crisis_velocity_baseline_std.labels(topic_id=topic_id).set(baseline.std)
        
        return baseline
    
    def set_baseline(self, topic_id: str, baseline: VelocityBaseline):
        """Set baseline for topic (for testing/manual override)"""
        self.baselines[topic_id] = baseline
    
    def normalize_velocity(
        self,
        velocity: float,
        baseline: VelocityBaseline
    ) -> NormalizedVelocity:
        """
        Normalize velocity by baseline for anomaly detection
        
        Returns z-score and sigma above mean
        """
        z_score = baseline.z_score(velocity)
        sigma_above = z_score  # Same as z-score
        
        return NormalizedVelocity(
            raw_velocity=velocity,
            z_score=z_score,
            sigma_above_mean=sigma_above,
            baseline_mean=baseline.mean,
            baseline_std=baseline.std
        )
    
    def _compute_base_risk(
        self,
        velocity: float,
        sources: List[CrisisSource]
    ) -> float:
        """
        Compute base risk score before tier-1 capping
        
        Factors:
        - Velocity magnitude
        - Source count
        - Source diversity
        - Tier distribution
        """
        # Simple heuristic (production: ML model)
        
        # Velocity factor (higher = more risk)
        velocity_score = min(velocity / 10.0, 1.0)  # Normalize to 0-1
        
        # Source count factor
        source_count = len(sources)
        source_score = min(source_count / 10.0, 1.0)
        
        # Tier diversity factor
        tier_levels = set(s.tier for s in sources if s.tier)
        diversity_score = len(tier_levels) / 3.0  # Max 3 tiers
        
        # Weighted combination
        base_risk = (
            0.5 * velocity_score +
            0.3 * source_score +
            0.2 * diversity_score
        )
        
        return min(base_risk, 1.0)
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc.replace('www.', '')
    
    def _is_news_outlet(self, domain: str) -> bool:
        """Simple heuristic for news outlet detection"""
        news_keywords = ['news', 'times', 'post', 'journal', 'tribune', 'herald']
        return any(keyword in domain.lower() for keyword in news_keywords)
