"""Tests for agents.llm_guardrails - Temperature and source validation"""
import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from agents.llm_guardrails import LLMGuardrails, validate_llm_output
from core.exceptions import ConfigurationError, ValidationError
from core.contracts import LLMConfig


class TestTemperatureValidation:
    """Test temperature ≤ 0.2 enforcement (Q_008)"""
    
    def test_temperature_exceeds_max(self):
        """
        T002 Acceptance: temperature > 0.2 raises ConfigurationError at startup
        Tests fail-fast validation before any API calls
        """
        with pytest.raises(ConfigurationError) as exc_info:
            LLMGuardrails(
                model="claude-sonnet-4.5",
                temperature=0.5,  # Exceeds 0.2 limit
                max_tokens=800
            )
        
        assert "temperature" in str(exc_info.value).lower()
        assert "0.2" in str(exc_info.value)
    
    def test_temperature_at_max_allowed(self):
        """Test temperature = 0.2 (boundary) is accepted"""
        guardrails = LLMGuardrails(
            model="claude-sonnet-4.5",
            temperature=0.2,  # Exactly at limit
            max_tokens=800
        )
        
        assert guardrails.temperature == 0.2
    
    def test_temperature_zero_allowed(self):
        """Test temperature = 0 (most deterministic) is accepted"""
        guardrails = LLMGuardrails(
            model="claude-sonnet-4.5",
            temperature=0.0,
            max_tokens=800
        )
        
        assert guardrails.temperature == 0.0
    
    def test_temperature_negative_rejected(self):
        """Test negative temperature is rejected"""
        with pytest.raises(ConfigurationError) as exc_info:
            LLMGuardrails(
                model="claude-sonnet-4.5",
                temperature=-0.1,
                max_tokens=800
            )
        
        assert "temperature" in str(exc_info.value).lower()
    
    def test_llm_config_schema_validates_temperature(self):
        """Contract: LLMConfig Pydantic schema enforces temperature ∈ [0, 0.2]"""
        from pydantic import ValidationError as PydanticValidationError
        
        # Valid temperature
        config = LLMConfig(
            model="claude-sonnet-4.5",
            temperature=0.15,
            max_tokens=800,
            source_ids=["src_1", "src_2"]
        )
        assert config.temperature == 0.15
        
        # Invalid temperature > 0.2
        with pytest.raises(PydanticValidationError) as exc_info:
            LLMConfig(
                model="claude-sonnet-4.5",
                temperature=0.3,
                max_tokens=800,
                source_ids=["src_1", "src_2"]
            )
        
        assert "temperature" in str(exc_info.value).lower()


class TestSourceValidation:
    """Test source_ids ≥ 2 enforcement (Q_009)"""
    
    def test_insufficient_sources(self):
        """
        T002 Acceptance: source_ids < 2 raises ValidationError
        Tests RAG-only policy enforcement
        """
        with pytest.raises(ValidationError) as exc_info:
            validate_llm_output(
                output={
                    "text": "Analysis result",
                    "source_ids": ["src_1"]  # Only 1 source
                },
                min_sources=2
            )
        
        assert "source" in str(exc_info.value).lower()
        assert "2" in str(exc_info.value)
    
    def test_exactly_two_sources_accepted(self):
        """Test minimum 2 sources (boundary) is accepted"""
        result = validate_llm_output(
            output={
                "text": "Analysis result",
                "source_ids": ["src_1", "src_2"]
            },
            min_sources=2
        )
        
        assert result is True
    
    def test_zero_sources_rejected(self):
        """Test no sources is rejected"""
        with pytest.raises(ValidationError) as exc_info:
            validate_llm_output(
                output={
                    "text": "Analysis result",
                    "source_ids": []
                },
                min_sources=2
            )
        
        assert "source" in str(exc_info.value).lower()
    
    def test_missing_source_ids_field_rejected(self):
        """Test missing source_ids field is rejected"""
        with pytest.raises(ValidationError) as exc_info:
            validate_llm_output(
                output={
                    "text": "Analysis result"
                    # Missing source_ids
                },
                min_sources=2
            )
        
        assert "source" in str(exc_info.value).lower()
    
    def test_llm_config_schema_validates_sources(self):
        """Contract: LLMConfig Pydantic schema enforces min_items=2 for source_ids"""
        from pydantic import ValidationError as PydanticValidationError
        
        # Valid sources
        config = LLMConfig(
            model="claude-sonnet-4.5",
            temperature=0.2,
            max_tokens=800,
            source_ids=["src_1", "src_2", "src_3"]
        )
        assert len(config.source_ids) == 3
        
        # Invalid sources < 2
        with pytest.raises(PydanticValidationError) as exc_info:
            LLMConfig(
                model="claude-sonnet-4.5",
                temperature=0.2,
                max_tokens=800,
                source_ids=["src_1"]  # Only 1 source
            )
        
        assert "source" in str(exc_info.value).lower()


class TestLLMGuardrails:
    """Integration tests for LLMGuardrails"""
    
    def test_guardrails_initialization_success(self):
        """Test successful initialization with valid config"""
        guardrails = LLMGuardrails(
            model="claude-sonnet-4.5",
            temperature=0.1,
            max_tokens=800
        )
        
        assert guardrails.model == "claude-sonnet-4.5"
        assert guardrails.temperature == 0.1
        assert guardrails.max_tokens == 800
    
    def test_guardrails_prevents_api_call_on_invalid_config(self):
        """Test no API call attempted if config invalid (fail-fast)"""
        with patch('agents.llm_guardrails.anthropic_client') as mock_client:
            with pytest.raises(ConfigurationError):
                guardrails = LLMGuardrails(
                    model="claude-sonnet-4.5",
                    temperature=0.5,  # Invalid
                    max_tokens=800
                )
            
            # Verify no API call was made
            mock_client.messages.create.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_output_validation_integrated(self):
        """Test output validation rejects insufficient sources"""
        guardrails = LLMGuardrails(
            model="claude-sonnet-4.5",
            temperature=0.2,
            max_tokens=800
        )
        
        # Mock LLM response with insufficient sources
        mock_output = {
            "text": "Result",
            "source_ids": ["src_1"]  # Only 1 source
        }
        
        with pytest.raises(ValidationError):
            await guardrails.validate_and_return(mock_output)
    
    def test_temperature_logging_on_startup(self):
        """Test temperature value logged on startup for audit"""
        with patch('agents.llm_guardrails.logger') as mock_logger:
            guardrails = LLMGuardrails(
                model="claude-sonnet-4.5",
                temperature=0.15,
                max_tokens=800
            )
            
            mock_logger.info.assert_called()
            call_args = str(mock_logger.info.call_args)
            assert "0.15" in call_args or "temperature" in call_args.lower()


class TestFailFast:
    """Test fail-fast behavior (startup validation)"""
    
    def test_startup_validation_before_api_call(self):
        """Test configuration validated at startup, not at API call time"""
        # This should fail immediately at instantiation
        with pytest.raises(ConfigurationError):
            LLMGuardrails(
                model="claude-sonnet-4.5",
                temperature=1.0,  # Way over limit
                max_tokens=800
            )
    
    def test_misconfiguration_prevents_instantiation(self):
        """Test misconfigured guardrails cannot be instantiated"""
        invalid_configs = [
            {"temperature": 0.5, "max_tokens": 800},
            {"temperature": -0.1, "max_tokens": 800},
            {"temperature": 0.2, "max_tokens": -100},
        ]
        
        for config in invalid_configs:
            with pytest.raises(ConfigurationError):
                LLMGuardrails(model="claude-sonnet-4.5", **config)
