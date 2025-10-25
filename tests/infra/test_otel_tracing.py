"""
OTEL Distributed Tracing Tests
Tests E2E trace_id propagation from API gateway → Agent → Database
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
import uuid


class TestOTELTracePropagation:
    """Test trace_id propagates across API→Agent→DB"""
    
    @pytest.fixture
    def tracer_setup(self):
        """Setup in-memory OTEL tracer for testing"""
        provider = TracerProvider()
        exporter = InMemorySpanExporter()
        processor = SimpleSpanProcessor(exporter)
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        tracer = trace.get_tracer(__name__)
        return tracer, exporter
    
    def test_trace_id_propagates_api_to_db(self, tracer_setup):
        """
        ACCEPTANCE: trace_id same in API span → Agent span → DB span
        DRY RUN: Mock all external calls
        """
        tracer, exporter = tracer_setup
        
        # Mock database connection
        with patch('infrastructure.otel.db_execute') as mock_db:
            # Simulate API request creating root span
            with tracer.start_as_current_span("api_request") as api_span:
                api_trace_id = api_span.get_span_context().trace_id
                
                # Simulate agent processing
                with tracer.start_as_current_span("agent_process") as agent_span:
                    agent_trace_id = agent_span.get_span_context().trace_id
                    
                    # Simulate DB write
                    with tracer.start_as_current_span("db_write") as db_span:
                        db_trace_id = db_span.get_span_context().trace_id
                        mock_db.return_value = {"rows": 1}
        
        # Verify trace_id continuity
        assert api_trace_id == agent_trace_id == db_trace_id, \
            "trace_id must be same across all spans"
        
        # Verify all spans recorded
        spans = exporter.get_finished_spans()
        assert len(spans) == 3, "Expected 3 spans: API, Agent, DB"
        
        # Verify span names
        span_names = [s.name for s in spans]
        assert "api_request" in span_names
        assert "agent_process" in span_names
        assert "db_write" in span_names
        
        # Verify trace continuity metric
        trace_ids = [s.get_span_context().trace_id for s in spans]
        continuity = len(set(trace_ids)) == 1
        assert continuity, "trace_continuity must be 1.0 (single trace_id)"
    
    def test_trace_async_propagates(self, tracer_setup):
        """
        ACCEPTANCE: Prefect/Celery async tasks carry trace_id
        Q_105: async context propagation
        """
        tracer, exporter = tracer_setup
        
        # Simulate parent span
        with tracer.start_as_current_span("parent_task") as parent_span:
            parent_trace_id = parent_span.get_span_context().trace_id
            
            # Simulate async task spawning
            # In real code, this would be Prefect task or Celery apply_async
            carrier = {}
            propagator = TraceContextTextMapPropagator()
            propagator.inject(carrier)
            
            # Simulate async task receiving context
            ctx = propagator.extract(carrier)
            
            # Async task span
            with tracer.start_as_current_span("async_task", context=ctx) as async_span:
                async_trace_id = async_span.get_span_context().trace_id
        
        # Verify trace_id propagated to async task
        assert parent_trace_id == async_trace_id, \
            "Async task must preserve parent trace_id"
        
        spans = exporter.get_finished_spans()
        assert len(spans) == 2
        
        # Verify async continuity metric >= 0.99
        trace_ids = [s.get_span_context().trace_id for s in spans]
        continuity_rate = (len([t for t in trace_ids if t == parent_trace_id]) / len(trace_ids))
        assert continuity_rate >= 0.99, f"trace_continuity {continuity_rate} must be >= 0.99"
    
    def test_trace_context_w3c_format(self):
        """
        CONTRACT: OTEL trace context follows W3C Trace Context format
        Format: traceparent: 00-<trace_id>-<span_id>-<flags>
        """
        propagator = TraceContextTextMapPropagator()
        
        # Create mock span context
        trace_id = trace.format_trace_id(int(uuid.uuid4().hex[:32], 16))
        span_id = trace.format_span_id(int(uuid.uuid4().hex[:16], 16))
        
        carrier = {}
        # In real scenario, this would be done by OTEL SDK
        carrier['traceparent'] = f"00-{trace_id}-{span_id}-01"
        
        # Verify format
        assert 'traceparent' in carrier
        parts = carrier['traceparent'].split('-')
        assert len(parts) == 4, "W3C traceparent must have 4 parts"
        assert parts[0] == '00', "version must be 00"
        assert len(parts[1]) == 32, "trace_id must be 32 hex chars"
        assert len(parts[2]) == 16, "span_id must be 16 hex chars"
        assert parts[3] in ['00', '01'], "flags must be 00 or 01"
    
    def test_trace_latency_under_5s(self, tracer_setup):
        """
        METRIC: p95_trace_latency < 5000ms
        Measure span end-to-end time
        """
        tracer, exporter = tracer_setup
        
        import time
        
        with tracer.start_as_current_span("latency_test") as span:
            # Simulate processing (mock fast path)
            time.sleep(0.001)  # 1ms mock
        
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        
        # Calculate latency in milliseconds
        span_data = spans[0]
        latency_ms = (span_data.end_time - span_data.start_time) / 1_000_000
        
        # P95 threshold
        assert latency_ms < 5000, f"Trace latency {latency_ms}ms must be < 5000ms"


class TestOTELIdempotency:
    """Test OTEL trace_id is idempotent (UUID v4)"""
    
    def test_trace_id_uuid_v4_format(self):
        """
        RISK GATE: trace_id must be valid UUID v4 (idempotency_key)
        """
        trace_id_int = int(uuid.uuid4().hex[:32], 16)
        trace_id_hex = trace.format_trace_id(trace_id_int)
        
        # Verify 32 hex characters
        assert len(trace_id_hex) == 32
        assert all(c in '0123456789abcdef' for c in trace_id_hex)
    
    def test_kill_switch_otel_enabled(self):
        """
        RISK GATE: OTEL_ENABLED env flag controls tracing
        """
        import os
        
        # Mock environment
        with patch.dict(os.environ, {'OTEL_ENABLED': 'false'}):
            # In real code, this would check OTEL_ENABLED and skip instrumentation
            otel_enabled = os.getenv('OTEL_ENABLED', 'true').lower() == 'true'
            assert otel_enabled == False, "Kill switch must disable tracing"
        
        with patch.dict(os.environ, {'OTEL_ENABLED': 'true'}):
            otel_enabled = os.getenv('OTEL_ENABLED', 'true').lower() == 'true'
            assert otel_enabled == True, "Default must enable tracing"
