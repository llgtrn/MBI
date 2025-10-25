"""
OpenTelemetry distributed tracing implementation with E2E latency tracking.

Provides trace propagation across service boundaries, span management, and
latency metrics for gateway->DB flows (<5s SLA enforcement).

CRITICAL REQUIREMENTS (Q_018):
- E2E trace from API gateway through agents to database
- Jaeger exporter with batch processing
- Latency metrics: gateway->DB <5s (99th percentile)
- Automatic trace context propagation
- Span attributes: service.name, component, operation

ACCEPTANCE CRITERIA:
- test_e2e_trace_gateway_to_db passes (simulated 4.2s flow)
- test_latency_sla_violation_metric passes (>5s triggers alert)
- test_trace_context_propagation passes (cross-service)
- Prometheus metric: trace_latency_seconds_bucket exported
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from prometheus_client import Histogram
import time
from typing import Optional, Dict, Any
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

# Prometheus metric for E2E latency
trace_latency = Histogram(
    'trace_latency_seconds',
    'End-to-end trace latency from gateway to database',
    ['service', 'operation'],
    buckets=[0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0]
)

# SLA threshold
E2E_LATENCY_SLA_SECONDS = 5.0


class TracingConfig:
    """Configuration for distributed tracing."""
    
    def __init__(
        self,
        service_name: str = "mbi-system",
        jaeger_agent_host: str = "localhost",
        jaeger_agent_port: int = 6831,
        enable_tracing: bool = True,
        sample_rate: float = 1.0,
        max_queue_size: int = 2048,
        max_export_batch_size: int = 512
    ):
        self.service_name = service_name
        self.jaeger_agent_host = jaeger_agent_host
        self.jaeger_agent_port = jaeger_agent_port
        self.enable_tracing = enable_tracing
        self.sample_rate = sample_rate
        self.max_queue_size = max_queue_size
        self.max_export_batch_size = max_export_batch_size


class DistributedTracer:
    """
    Distributed tracing manager with OpenTelemetry and Jaeger backend.
    
    Features:
    - Automatic trace context propagation
    - E2E latency tracking with SLA enforcement
    - Batch span export to Jaeger
    - Prometheus metrics integration
    - Service mesh support
    """
    
    def __init__(self, config: TracingConfig):
        self.config = config
        self.tracer_provider: Optional[TracerProvider] = None
        self.tracer: Optional[trace.Tracer] = None
        self.propagator = TraceContextTextMapPropagator()
        
        if config.enable_tracing:
            self._initialize_tracer()
    
    def _initialize_tracer(self):
        """Initialize OpenTelemetry tracer with Jaeger exporter."""
        # Create resource with service metadata
        resource = Resource.create({
            "service.name": self.config.service_name,
            "service.version": "2.0.0",
            "deployment.environment": "production"
        })
        
        # Create tracer provider
        self.tracer_provider = TracerProvider(resource=resource)
        
        # Configure Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name=self.config.jaeger_agent_host,
            agent_port=self.config.jaeger_agent_port,
        )
        
        # Add batch span processor
        span_processor = BatchSpanProcessor(
            jaeger_exporter,
            max_queue_size=self.config.max_queue_size,
            max_export_batch_size=self.config.max_export_batch_size
        )
        self.tracer_provider.add_span_processor(span_processor)
        
        # Set global tracer provider
        trace.set_tracer_provider(self.tracer_provider)
        
        # Get tracer instance
        self.tracer = trace.get_tracer(__name__)
        
        # Instrument common libraries
        RequestsInstrumentor().instrument()
        SQLAlchemyInstrumentor().instrument()
        
        logger.info(
            f"Distributed tracing initialized: service={self.config.service_name}, "
            f"jaeger={self.config.jaeger_agent_host}:{self.config.jaeger_agent_port}"
        )
    
    @contextmanager
    def trace_operation(
        self,
        operation_name: str,
        attributes: Optional[Dict[str, Any]] = None,
        record_latency: bool = False,
        sla_seconds: Optional[float] = None
    ):
        """
        Context manager for tracing an operation.
        
        Args:
            operation_name: Name of the operation being traced
            attributes: Additional span attributes
            record_latency: If True, record latency in Prometheus
            sla_seconds: If set, check against SLA threshold
        
        Yields:
            Span object for additional instrumentation
        
        Example:
            with tracer.trace_operation("process_order", {"order_id": "123"}):
                # ... operation code ...
                pass
        """
        if not self.config.enable_tracing or not self.tracer:
            # Tracing disabled, no-op
            yield None
            return
        
        start_time = time.time()
        
        with self.tracer.start_as_current_span(operation_name) as span:
            # Add attributes
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            
            span.set_attribute("component", self.config.service_name)
            
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
            finally:
                # Record latency
                latency = time.time() - start_time
                
                if record_latency:
                    trace_latency.labels(
                        service=self.config.service_name,
                        operation=operation_name
                    ).observe(latency)
                
                # Check SLA
                if sla_seconds is not None and latency > sla_seconds:
                    logger.warning(
                        f"SLA violation: {operation_name} took {latency:.2f}s "
                        f"(threshold: {sla_seconds}s)"
                    )
                    span.set_attribute("sla.violated", True)
                    span.set_attribute("sla.threshold_seconds", sla_seconds)
                    span.set_attribute("sla.actual_seconds", latency)
    
    def inject_trace_context(self, carrier: Dict[str, str]):
        """
        Inject current trace context into carrier (e.g., HTTP headers).
        
        Args:
            carrier: Dictionary to inject context into (modified in-place)
        
        Example:
            headers = {}
            tracer.inject_trace_context(headers)
            requests.get(url, headers=headers)
        """
        self.propagator.inject(carrier)
    
    def extract_trace_context(self, carrier: Dict[str, str]):
        """
        Extract trace context from carrier (e.g., HTTP headers).
        
        Args:
            carrier: Dictionary containing trace context
        
        Returns:
            Trace context object
        
        Example:
            context = tracer.extract_trace_context(request.headers)
        """
        return self.propagator.extract(carrier)
    
    def shutdown(self, timeout_seconds: int = 30):
        """
        Shutdown tracer and flush pending spans.
        
        Args:
            timeout_seconds: Maximum time to wait for span export
        """
        if self.tracer_provider:
            self.tracer_provider.shutdown()
            logger.info("Distributed tracing shutdown completed")


# Global tracer instance
_global_tracer: Optional[DistributedTracer] = None


def initialize_tracing(config: TracingConfig):
    """
    Initialize global distributed tracer.
    
    Args:
        config: Tracing configuration
    
    Example:
        config = TracingConfig(
            service_name="mbi-api",
            jaeger_agent_host="jaeger",
            jaeger_agent_port=6831
        )
        initialize_tracing(config)
    """
    global _global_tracer
    _global_tracer = DistributedTracer(config)


def get_tracer() -> DistributedTracer:
    """
    Get global tracer instance.
    
    Returns:
        Global DistributedTracer instance
    
    Raises:
        RuntimeError: If tracing not initialized
    """
    if _global_tracer is None:
        raise RuntimeError(
            "Tracing not initialized. Call initialize_tracing() first."
        )
    return _global_tracer


@contextmanager
def trace_e2e_flow(
    flow_name: str,
    attributes: Optional[Dict[str, Any]] = None
):
    """
    Trace end-to-end flow from gateway to database with SLA enforcement.
    
    Args:
        flow_name: Name of the E2E flow
        attributes: Additional span attributes
    
    Yields:
        Span object
    
    Example:
        with trace_e2e_flow("order_processing", {"user_id": "123"}):
            # Gateway receives request
            # ... process through agents ...
            # Database query executes
            pass
    
    CRITICAL: Enforces <5s SLA for gateway->DB flows (Q_018)
    """
    tracer = get_tracer()
    
    with tracer.trace_operation(
        operation_name=f"e2e.{flow_name}",
        attributes=attributes,
        record_latency=True,
        sla_seconds=E2E_LATENCY_SLA_SECONDS
    ) as span:
        yield span
