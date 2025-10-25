"""
OTEL Distributed Tracing Implementation
Implements E2E trace propagation across API → Agent → Database
"""
import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class OTELConfig:
    """OTEL configuration from environment"""
    
    OTEL_ENABLED: bool = os.getenv('OTEL_ENABLED', 'true').lower() == 'true'
    JAEGER_ENDPOINT: str = os.getenv('JAEGER_ENDPOINT', 'http://localhost:14268/api/traces')
    SERVICE_NAME: str = os.getenv('SERVICE_NAME', 'mbi-api')
    TRACE_SAMPLE_RATE: float = float(os.getenv('TRACE_SAMPLE_RATE', '1.0'))


def init_otel_tracing() -> Optional[trace.Tracer]:
    """
    Initialize OTEL distributed tracing
    
    RISK GATES:
    - Kill switch: OTEL_ENABLED env flag
    - Idempotency: trace_id is UUID v4 (128-bit random)
    
    Returns:
        Tracer instance if enabled, None otherwise
    """
    if not OTELConfig.OTEL_ENABLED:
        logger.info("OTEL tracing disabled via OTEL_ENABLED=false")
        return None
    
    try:
        # Setup tracer provider
        provider = TracerProvider()
        
        # Jaeger exporter for spans
        jaeger_exporter = JaegerExporter(
            collector_endpoint=OTELConfig.JAEGER_ENDPOINT,
        )
        
        # Batch processor for performance
        processor = BatchSpanProcessor(jaeger_exporter)
        provider.add_span_processor(processor)
        
        # Set global tracer provider
        trace.set_tracer_provider(provider)
        
        # Instrument frameworks
        FastAPIInstrumentor().instrument()
        RequestsInstrumentor().instrument()
        SQLAlchemyInstrumentor().instrument()
        
        logger.info(f"OTEL tracing initialized; service={OTELConfig.SERVICE_NAME}")
        
        return trace.get_tracer(__name__)
    
    except Exception as e:
        logger.error(f"Failed to initialize OTEL: {e}")
        # Graceful degradation: continue without tracing
        return None


def propagate_trace_context(carrier: Dict[str, str]) -> None:
    """
    Inject current trace context into carrier (for async tasks)
    
    Used for Prefect/Celery task context propagation (Q_105)
    
    Args:
        carrier: Dictionary to inject trace headers into
    """
    if not OTELConfig.OTEL_ENABLED:
        return
    
    propagator = TraceContextTextMapPropagator()
    propagator.inject(carrier)


def extract_trace_context(carrier: Dict[str, str]) -> Any:
    """
    Extract trace context from carrier (async task receives parent context)
    
    Args:
        carrier: Dictionary containing trace headers
        
    Returns:
        Context object for span creation
    """
    if not OTELConfig.OTEL_ENABLED:
        return None
    
    propagator = TraceContextTextMapPropagator()
    return propagator.extract(carrier)


def create_span_with_context(
    tracer: trace.Tracer,
    span_name: str,
    context: Optional[Any] = None,
    attributes: Optional[Dict[str, Any]] = None
) -> trace.Span:
    """
    Create span with optional parent context
    
    Args:
        tracer: OTEL tracer instance
        span_name: Name of the span
        context: Optional parent context (for async tasks)
        attributes: Optional span attributes
        
    Returns:
        Span context manager
    """
    if context:
        span = tracer.start_as_current_span(span_name, context=context)
    else:
        span = tracer.start_as_current_span(span_name)
    
    # Add custom attributes
    if attributes:
        for key, value in attributes.items():
            span.set_attribute(key, value)
    
    return span


def db_execute(query: str, params: Optional[Dict] = None) -> Dict:
    """
    Mock database execution with tracing
    Used in tests; real implementation would use SQLAlchemy instrumentation
    
    Args:
        query: SQL query string
        params: Query parameters
        
    Returns:
        Mock result
    """
    # In real implementation, SQLAlchemyInstrumentor automatically creates spans
    # This is a mock for testing
    tracer = trace.get_tracer(__name__)
    
    with tracer.start_as_current_span("db_execute") as span:
        span.set_attribute("db.statement", query)
        if params:
            span.set_attribute("db.params", str(params))
        
        # Mock result
        return {"rows": 1}


# Global tracer instance
_tracer: Optional[trace.Tracer] = None


def get_tracer() -> Optional[trace.Tracer]:
    """Get global tracer instance (lazy init)"""
    global _tracer
    if _tracer is None and OTELConfig.OTEL_ENABLED:
        _tracer = init_otel_tracing()
    return _tracer
