"""
Test suite for LLM schemas
Validates contracts for RAG-only LLM operations with guardrails
"""

import pytest
from datetime import datetime
from pydantic import ValidationError
from contracts.llm_schemas import (
    SourceReference,
    LLMRequest,
    LLMResponse,
    ValidationResult,
    GuardrailsConfig
)


class TestSourceReference:
    """Test SourceReference schema"""
    
    def test_valid_ssot_source(self):
        """Valid SSOT source should pass validation"""
        source = SourceReference(
            id="ssot_doc_123",
            type="ssot_doc",
            quote="Revenue increased by 15%"
        )
        assert source.id == "ssot_doc_123"
        assert source.type == "ssot_doc"
        assert source.url is None
    
    def test_valid_web_source(self):
        """Valid web source with URL should pass validation"""
        source = SourceReference(
            id="web_1",
            url="https://example.com/article",
            type="web_search",
            quote="Market analysis shows..."
        )
        assert source.url == "https://example.com/article"
        assert source.type == "web_search"
    
    def test_missing_required_fields(self):
        """Missing required fields should fail validation"""
        with pytest.raises(ValidationError):
            SourceReference(id="test")  # Missing type


class TestLLMRequest:
    """Test LLMRequest schema"""
    
    def test_valid_request(self):
        """Valid LLM request should pass validation"""
        request = LLMRequest(
            task="crisis_brief",
            prompt="SYSTEM: Analyze...",
            context_sources=[
                SourceReference(id="src_1", type="ssot_doc"),
                SourceReference(id="src_2", type="web_search", url="https://example.com")
            ],
            model="claude-sonnet-4.5",
            temperature=0.2,
            max_tokens=800,
            response_format="json"
        )
        assert request.task == "crisis_brief"
        assert request.temperature == 0.2
        assert len(request.context_sources) == 2
    
    def test_temperature_validation(self):
        """Temperature >0.2 should fail validation"""
        with pytest.raises(ValidationError) as exc_info:
            LLMRequest(
                task="test",
                prompt="test",
                context_sources=[
                    SourceReference(id="src_1", type="ssot_doc"),
                    SourceReference(id="src_2", type="ssot_doc")
                ],
                model="test",
                temperature=0.5  # Too high
            )
        assert "Temperature must be ≤0.2" in str(exc_info.value)
    
    def test_min_sources_validation(self):
        """Less than 2 sources should fail validation"""
        with pytest.raises(ValidationError) as exc_info:
            LLMRequest(
                task="test",
                prompt="test",
                context_sources=[
                    SourceReference(id="src_1", type="ssot_doc")  # Only 1 source
                ],
                model="test",
                temperature=0.2
            )
        assert "≥2 sources" in str(exc_info.value)
    
    def test_temperature_boundary(self):
        """Temperature exactly 0.2 should pass"""
        request = LLMRequest(
            task="test",
            prompt="test",
            context_sources=[
                SourceReference(id="src_1", type="ssot_doc"),
                SourceReference(id="src_2", type="ssot_doc")
            ],
            model="test",
            temperature=0.2
        )
        assert request.temperature == 0.2


class TestLLMResponse:
    """Test LLMResponse schema"""
    
    def test_valid_response(self):
        """Valid LLM response should pass validation"""
        response = LLMResponse(
            task="crisis_brief",
            content='{"stance": "against", "risk_score": 0.82}',
            source_ids=["src_1", "src_2", "src_3"],
            model="claude-sonnet-4.5",
            input_tokens=1500,
            output_tokens=350,
            latency_ms=2300,
            cost_usd=0.015,
            verified=True
        )
        assert response.task == "crisis_brief"
        assert len(response.source_ids) == 3
        assert response.verified is True
    
    def test_source_ids_validation(self):
        """Less than 2 source_ids should fail validation"""
        with pytest.raises(ValidationError) as exc_info:
            LLMResponse(
                task="test",
                content="test",
                source_ids=["src_1"],  # Only 1 source
                model="test",
                input_tokens=100,
                output_tokens=50,
                latency_ms=1000,
                cost_usd=0.01
            )
        assert "≥2 sources" in str(exc_info.value)
    
    def test_timestamp_auto_generation(self):
        """Timestamp should auto-generate if not provided"""
        response = LLMResponse(
            task="test",
            content="test",
            source_ids=["src_1", "src_2"],
            model="test",
            input_tokens=100,
            output_tokens=50,
            latency_ms=1000,
            cost_usd=0.01
        )
        assert isinstance(response.timestamp, datetime)
    
    def test_verified_default_false(self):
        """Verified should default to False"""
        response = LLMResponse(
            task="test",
            content="test",
            source_ids=["src_1", "src_2"],
            model="test",
            input_tokens=100,
            output_tokens=50,
            latency_ms=1000,
            cost_usd=0.01
        )
        assert response.verified is False


class TestValidationResult:
    """Test ValidationResult schema"""
    
    def test_valid_result(self):
        """Valid validation result should pass"""
        result = ValidationResult(
            valid=True,
            checks={
                "json_schema": True,
                "source_count": True,
                "toxicity": True,
                "pii": True
            }
        )
        assert result.valid is True
        assert result.error is None
    
    def test_invalid_result_with_error(self):
        """Invalid result should include error details"""
        result = ValidationResult(
            valid=False,
            error="toxicity_detected",
            details="Toxicity score 0.91 exceeds threshold 0.8",
            checks={
                "json_schema": True,
                "source_count": True,
                "toxicity": False,
                "pii": True
            }
        )
        assert result.valid is False
        assert result.error == "toxicity_detected"
        assert "0.91" in result.details


class TestGuardrailsConfig:
    """Test GuardrailsConfig schema"""
    
    def test_default_config(self):
        """Default config should enforce strict RAG-only"""
        config = GuardrailsConfig()
        assert config.rag_only is True
        assert config.min_sources == 2
        assert config.max_temperature == 0.2
        assert config.toxicity_threshold == 0.8
        assert config.require_source_ids is True
    
    def test_custom_config(self):
        """Custom config should override defaults"""
        config = GuardrailsConfig(
            min_sources=3,
            toxicity_threshold=0.7,
            max_retries=5
        )
        assert config.min_sources == 3
        assert config.toxicity_threshold == 0.7
        assert config.max_retries == 5
        # Defaults still apply
        assert config.rag_only is True
        assert config.max_temperature == 0.2


class TestEndToEndFlow:
    """Test complete LLM request-response flow"""
    
    def test_complete_flow(self):
        """Test full LLM interaction flow"""
        # 1. Create request
        request = LLMRequest(
            task="crisis_brief",
            prompt="Analyze brand crisis...",
            context_sources=[
                SourceReference(id="src_1", type="ssot_doc", quote="Brand mentioned negatively"),
                SourceReference(id="src_2", type="web_search", url="https://example.com")
            ],
            model="claude-sonnet-4.5",
            temperature=0.2
        )
        
        # 2. Simulate response
        response = LLMResponse(
            task=request.task,
            content='{"stance": "against", "risk_score": 0.82}',
            source_ids=["src_1", "src_2"],
            model=request.model,
            input_tokens=1500,
            output_tokens=350,
            latency_ms=2300,
            cost_usd=0.015,
            verified=True
        )
        
        # 3. Validate
        validation = ValidationResult(
            valid=True,
            checks={
                "json_schema": True,
                "source_count": True,
                "source_verification": True,
                "toxicity": True,
                "pii": True
            }
        )
        
        assert request.task == response.task
        assert len(response.source_ids) >= request.context_sources.__len__()
        assert validation.valid is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
