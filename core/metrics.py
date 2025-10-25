"""
Core metrics instrumentation for MBI system.

Provides Prometheus-compatible metrics primitives.
"""
from typing import Dict, List


class Counter:
    """
    Prometheus Counter metric.
    
    A counter is a cumulative metric that only increases.
    """
    
    def __init__(self, name: str, description: str, labels: List[str] = None):
        """
        Initialize counter.
        
        Args:
            name: Metric name
            description: Metric description
            labels: List of label names
        """
        self.name = name
        self.description = description
        self.labels = labels or []
        self._value = CounterValue()
    
    def labels(self, **label_values):
        """
        Return labeled counter instance.
        
        Args:
            **label_values: Label key-value pairs
            
        Returns:
            Self for chaining
        """
        # In production, this would track label values
        return self
    
    def inc(self, amount: float = 1.0):
        """
        Increment counter.
        
        Args:
            amount: Amount to increment by
        """
        # In production, this would emit to Prometheus
        self._value.inc(amount)


class CounterValue:
    """Internal counter value holder"""
    def __init__(self):
        self._val = 0.0
    
    def inc(self, amount: float = 1.0):
        self._val += amount
    
    def get(self):
        return self._val


class Gauge:
    """Prometheus Gauge metric."""
    
    def __init__(self, name: str, description: str, labels: List[str] = None):
        self.name = name
        self.description = description
        self.labels = labels or []
        self._value = 0
    
    def set(self, value: float, labels: Dict[str, any] = None):
        """Set gauge value."""
        self._value = value
    
    def inc(self, amount: float = 1.0, labels: Dict[str, any] = None):
        """Increment gauge."""
        self._value += amount
    
    def dec(self, amount: float = 1.0, labels: Dict[str, any] = None):
        """Decrement gauge."""
        self._value -= amount


class Histogram:
    """Prometheus Histogram metric."""
    
    def __init__(self, name: str, description: str, labels: List[str] = None):
        self.name = name
        self.description = description
        self.labels = labels or []
        self._observations = []
    
    def observe(self, value: float, labels: Dict[str, any] = None):
        """Record observation."""
        self._observations.append(value)


# Activation Agent Metrics
activation_operations_deduplicated = Counter(
    'activation_operations_deduplicated',
    'Total activation operations skipped due to deduplication',
    ['action']
)
