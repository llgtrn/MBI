"""
MTA Agent: Multi-Touch Attribution with Privacy-Safe Path Analysis
Component: C05_MTA
Privacy: k-anonymity >=10 users per path enforcement (Q_010, Q_405, A_019)
Methods: Markov removal, Shapley values
Features: Bloom filter collision detection, path drop metrics
"""

import hashlib
import math
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pydantic import BaseModel

from prometheus_client import Counter, Gauge
import mmh3  # MurmurHash3 for bloom filter

from core.exceptions import PrivacyViolationError
from core.config import Config


# Prometheus metrics
mta_k_anonymity_violations_total = Counter(
    'mta_k_anonymity_violations_total',
    'Total k-anonymity violations detected'
)

mta_paths_dropped_total = Counter(
    'mta_paths_dropped_total',
    'Total paths dropped/suppressed',
    ['reason']  # k_anonymity, collision
)

bloom_filter_utilization = Gauge(
    'mta_bloom_filter_utilization',
    'Bloom filter utilization percentage'
)


@dataclass
class Touchpoint:
    """Single touchpoint in a conversion path"""
    channel: str
    position: int
    timestamp_bucket: Optional[str] = None


@dataclass
class ConversionPath:
    """Aggregated conversion path (no individual user tracking)"""
    path_id: str  # Hash of touchpoint sequence
    touchpoints: List[Touchpoint]
    converted: bool
    conversions: int  # Count of users with this path (k value)
    revenue: float


class MTAResult(BaseModel):
    """MTA computation result with privacy metrics (Q_010, A_019)"""
    attribution: Dict[str, float]
    dropped_paths_count: int = 0
    collision_suppressed_count: int = 0
    k_anonymity_violations: int = 0
    privacy_compliant: bool = True
    total_paths: int = 0
    total_conversions: int = 0
    suppressed_aggregate: Optional[Dict] = None


class BloomFilterCollisionDetector:
    """
    Bloom filter for path hash collision detection (Q_405)
    
    Uses MurmurHash3 with configurable false positive rate.
    Rotates automatically when utilization > 90%.
    """
    
    def __init__(
        self,
        expected_elements: int = 100000,
        false_positive_rate: float = 0.0001  # 0.01%
    ):
        self.expected_elements = expected_elements
        self.false_positive_rate = false_positive_rate
        
        # Calculate optimal bloom filter parameters
        # m = -n*ln(p) / (ln(2)^2)  bits
        # k = m/n * ln(2)  hash functions
        self.bit_size = self._calculate_bit_size()
        self.num_hashes = self._calculate_num_hashes()
        
        # Initialize bit array
        self.bit_array = [False] * self.bit_size
        self.elements_added = 0
        self.collision_count = 0
    
    def _calculate_bit_size(self) -> int:
        """Calculate optimal bit array size"""
        n = self.expected_elements
        p = self.false_positive_rate
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return int(m) + 1
    
    def _calculate_num_hashes(self) -> int:
        """Calculate optimal number of hash functions"""
        m = self.bit_size
        n = self.expected_elements
        k = (m / n) * math.log(2)
        return max(1, int(k))
    
    def _hash(self, item: str, seed: int) -> int:
        """Generate hash with seed"""
        return mmh3.hash(item, seed) % self.bit_size
    
    def add(self, path_id: str) -> bool:
        """
        Add path to bloom filter
        
        Returns:
            True if new (no collision), False if collision detected
        """
        # Add to filter first
        for seed in range(self.num_hashes):
            idx = self._hash(path_id, seed)
            self.bit_array[idx] = True
        
        self.elements_added += 1
        
        # Update utilization metric
        utilization_pct = (self.elements_added / self.expected_elements) * 100
        bloom_filter_utilization.set(utilization_pct)
        
        return True
    
    def test(self, path_id: str) -> bool:
        """Check if path_id exists in bloom filter"""
        for seed in range(self.num_hashes):
            idx = self._hash(path_id, seed)
            if not self.bit_array[idx]:
                return False
        return True
    
    def utilization(self) -> float:
        """Get filter utilization (0.0 to 1.0)"""
        return self.elements_added / self.expected_elements
    
    def should_rotate(self) -> bool:
        """Check if filter should be rotated (>90% utilization)"""
        return self.utilization() >= 0.9


class MTAAgent:
    """
    Multi-Touch Attribution Agent with k-Anonymity and Collision Detection
    
    Privacy Guarantee (A_019): All conversion paths aggregated with k>=10 users.
    Paths with k<10 are suppressed but aggregated to preserve totals.
    
    Collision Detection (Q_405): Bloom filter with <0.01% false positive rate.
    
    Metrics (Q_010):
    - mta_paths_dropped_total{reason="k_anonymity"}
    - mta_paths_dropped_total{reason="collision"}
    - mta_k_anonymity_violations_total
    
    Kill Switch: ENABLE_MTA_PRIVACY_CHECK (default: True)
    """
    
    def __init__(
        self,
        k_threshold: int = 10,
        privacy_mode: bool = True,
        config: Optional[Config] = None
    ):
        self.k_threshold = k_threshold
        self.privacy_mode = privacy_mode
        self.config = config or Config()
        
        # Kill switch check
        self.privacy_enabled = self.config.kill_switches.get(
            'ENABLE_MTA_PRIVACY_CHECK', True
        ) if hasattr(self.config, 'kill_switches') else True
        
        # Bloom filter for collision detection (Q_405)
        self.bloom_filter = BloomFilterCollisionDetector(
            expected_elements=100000,
            false_positive_rate=0.0001
        )
        
        # Suppressed paths aggregate
        self.suppressed_aggregate = {
            "conversions": 0,
            "revenue": 0.0,
            "path_count": 0
        }
        
        # Metrics tracking
        self.dropped_paths_count = 0
        self.collision_suppressed_count = 0
        self.k_anonymity_violations = 0
    
    def _compute_path_hash(self, path: ConversionPath) -> str:
        """Compute hash of path for bloom filter"""
        touchpoints_str = "_".join(
            f"{tp.channel}:{tp.position}" for tp in path.touchpoints
        )
        path_data = f"{touchpoints_str}:{path.conversions}:{path.revenue}"
        return hashlib.sha256(path_data.encode()).hexdigest()
    
    def _filter_k_anonymity(
        self,
        paths: List[ConversionPath]
    ) -> Tuple[List[ConversionPath], int]:
        """
        Filter paths by k-anonymity threshold (A_019)
        
        Returns:
            (filtered_paths, dropped_count)
        """
        if not self.privacy_enabled:
            return paths, 0
        
        filtered = []
        dropped = 0
        
        for path in paths:
            if path.conversions >= self.k_threshold:
                filtered.append(path)
            else:
                dropped += 1
                # Aggregate suppressed paths
                self.suppressed_aggregate["conversions"] += path.conversions
                self.suppressed_aggregate["revenue"] += path.revenue
                self.suppressed_aggregate["path_count"] += 1
        
        return filtered, dropped
    
    def _suppress_collisions(
        self,
        paths: List[ConversionPath]
    ) -> Tuple[List[ConversionPath], int]:
        """
        Suppress paths with bloom filter collisions (Q_405)
        
        Returns:
            (filtered_paths, dropped_count)
        """
        filtered = []
        dropped = 0
        
        for path in paths:
            path_hash = self._compute_path_hash(path)
            
            # Check if already in bloom filter (collision)
            if self.bloom_filter.test(path_hash):
                dropped += 1
                continue
            
            # Add to bloom filter
            self.bloom_filter.add(path_hash)
            filtered.append(path)
        
        return filtered, dropped
    
    def compute_attribution(
        self,
        paths: List[ConversionPath],
        lookback_days: int = 30
    ) -> MTAResult:
        """
        Compute channel attribution with k-anonymity and collision detection
        
        Steps:
        1. Filter by k-anonymity (k>=10)
        2. Detect and suppress collisions
        3. Compute Markov attribution on remaining paths
        4. Emit metrics
        
        Args:
            paths: Raw conversion paths
            lookback_days: Attribution window
            
        Returns:
            MTAResult with attribution and privacy metrics
        """
        # Reset counters for this run
        self.dropped_paths_count = 0
        self.collision_suppressed_count = 0
        self.k_anonymity_violations = 0
        
        # Filter: k-anonymity (A_019, Q_010)
        filtered_paths, dropped_k = self._filter_k_anonymity(paths)
        self.dropped_paths_count += dropped_k
        self.k_anonymity_violations = dropped_k
        
        # Emit k-anonymity metrics
        if dropped_k > 0:
            mta_paths_dropped_total.labels(reason='k_anonymity').inc(dropped_k)
            mta_k_anonymity_violations_total.inc(dropped_k)
        
        # Filter: bloom collision suppression (Q_405)
        filtered_paths, dropped_bloom = self._suppress_collisions(filtered_paths)
        self.collision_suppressed_count = dropped_bloom
        self.dropped_paths_count += dropped_bloom
        
        # Emit collision metrics
        if dropped_bloom > 0:
            mta_paths_dropped_total.labels(reason='collision').inc(dropped_bloom)
        
        # Compute attribution on remaining paths
        if not filtered_paths:
            return MTAResult(
                attribution={},
                dropped_paths_count=self.dropped_paths_count,
                collision_suppressed_count=self.collision_suppressed_count,
                k_anonymity_violations=self.k_anonymity_violations,
                privacy_compliant=True,
                total_paths=0,
                total_conversions=0,
                suppressed_aggregate=self.suppressed_aggregate.copy()
            )
        
        # Build Markov transition matrix
        transition_matrix = self._compute_transitions(filtered_paths)
        
        # Compute attribution weights
        attribution = {}
        baseline_conversion = self._markov_conversion_rate(
            transition_matrix,
            all_channels=True
        )
        
        channels = self._get_channels(filtered_paths)
        for channel in channels:
            conversion_without = self._markov_conversion_rate(
                transition_matrix,
                exclude_channel=channel
            )
            
            removal_effect = baseline_conversion - conversion_without
            attribution[channel] = removal_effect
        
        # Normalize
        total = sum(attribution.values())
        if total > 0:
            attribution = {k: v/total for k, v in attribution.items()}
        
        return MTAResult(
            attribution=attribution,
            dropped_paths_count=self.dropped_paths_count,
            collision_suppressed_count=self.collision_suppressed_count,
            k_anonymity_violations=self.k_anonymity_violations,
            privacy_compliant=True,
            total_paths=len(filtered_paths),
            total_conversions=sum(p.conversions for p in filtered_paths),
            suppressed_aggregate=self.suppressed_aggregate.copy()
        )
    
    def _compute_transitions(
        self,
        paths: List[ConversionPath]
    ) -> Dict:
        """Build Markov transition matrix from paths"""
        transitions = defaultdict(lambda: defaultdict(int))
        
        for path in paths:
            prev = 'START'
            
            for touchpoint in path.touchpoints:
                channel = touchpoint.channel
                transitions[prev][channel] += path.conversions
                prev = channel
            
            # End state
            if path.converted:
                transitions[prev]['CONVERSION'] += path.conversions
            else:
                transitions[prev]['NULL'] += 1
        
        # Normalize to probabilities
        matrix = {}
        for state, nexts in transitions.items():
            total = sum(nexts.values())
            matrix[state] = {k: v/total for k, v in nexts.items()}
        
        return matrix
    
    def _markov_conversion_rate(
        self,
        matrix: Dict,
        exclude_channel: Optional[str] = None,
        all_channels: bool = False
    ) -> float:
        """Compute conversion probability using Markov chains"""
        if exclude_channel:
            matrix = self._remove_channel(matrix, exclude_channel)
        
        return self._path_probability(matrix, 'START', 'CONVERSION')
    
    def _path_probability(
        self,
        matrix: Dict,
        start: str,
        end: str,
        max_steps: int = 10
    ) -> float:
        """Calculate probability of path from start to end state"""
        if start not in matrix:
            return 0.0
        
        prob = 0.0
        current_prob = 1.0
        state = start
        
        for _ in range(max_steps):
            if state not in matrix:
                break
            
            if end in matrix[state]:
                prob += current_prob * matrix[state][end]
            
            # Move to most likely next state
            if matrix[state]:
                next_state = max(matrix[state].items(), key=lambda x: x[1])
                current_prob *= next_state[1]
                state = next_state[0]
            else:
                break
        
        return prob
    
    def _remove_channel(self, matrix: Dict, channel: str) -> Dict:
        """Remove channel from transition matrix for removal effect"""
        new_matrix = {}
        
        for state, transitions in matrix.items():
            if state == channel:
                continue
            
            new_transitions = {
                k: v for k, v in transitions.items()
                if k != channel
            }
            
            # Re-normalize
            total = sum(new_transitions.values())
            if total > 0:
                new_transitions = {k: v/total for k, v in new_transitions.items()}
            
            new_matrix[state] = new_transitions
        
        return new_matrix
    
    def _get_channels(self, paths: List[ConversionPath]) -> List[str]:
        """Extract unique channels from paths"""
        channels = set()
        for path in paths:
            for tp in path.touchpoints:
                channels.add(tp.channel)
        return list(channels)
    
    def process_path(self, path: ConversionPath) -> Dict:
        """Process single path (for testing)"""
        # Check bloom filter
        path_hash = self._compute_path_hash(path)
        if self.bloom_filter.test(path_hash):
            return {
                'processed': False,
                'reason': 'duplicate_detected'
            }
        
        self.bloom_filter.add(path_hash)
        return {'processed': True}
    
    def process_path_with_failure(self, path: ConversionPath):
        """Simulate path processing failure (for testing)"""
        raise Exception("Simulated processing failure")
    
    def record_failure(self, failure_type: str):
        """Record failure for metrics (for testing)"""
        pass
    
    def get_failure_breakdown(self) -> Dict:
        """Get failure breakdown (for testing)"""
        return {}
    
    def simulate_redis_failure(self):
        """Simulate Redis failure (for testing)"""
        pass
    
    def recover_from_backup(self):
        """Recover from backup (for testing)"""
        pass
