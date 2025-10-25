"""
Test suite for LLM Guardrails
Validates: JSON schema, source citation >=2, temperature <=0.2
"""
import pytest
from pydantic import BaseModel, ValidationError
from typing import List, Dict, Optional
from agents.llm_guardrails import LLMGuardrails, ValidationResult


class MockDocument(BaseModel):
    """Mock document for testing"""
    id: str
    text: str


class MockLLMOutput(BaseModel):
    """Mock LLM output schema"""
    text: str
    source_ids: List[str]


class TestJSONSchemaValidation:
    """Test JSON schema validation at LLM boundary"""
    
    def test_valid_json_passes(self):
        """Valid JSON with proper schema passes validation"""
        output = '{"text": "Test", "source_ids": ["src1", "src2"]}'
        sources = [
            MockDocument(id="src1", text="Source 1"),
            MockDocument(id="src2", text="Source 2")
        ]
        
        result = LLMGuardrails.validate_llm_output(
            output=output,
            schema=MockLLMOutput,
            sources=sources
        )
        
        assert result.valid is True
        assert result.error is None
    
    def test_invalid_json_rejected(self):
        """Invalid JSON structure is rejected"""
        output = '{"text": "Test", "source_ids": "not_a_list"}'
        sources = [MockDocument(id="src1", text="Source 1")]
        
        result = LLMGuardrails.validate_llm_output(
            output=output,
            schema=MockLLMOutput,
            sources=sources
        )
        
        assert result.valid is False
        assert result.error == 'json_schema_violation'
        assert 'source_ids' in result.details
    
    def test_malformed_json_rejected(self):
        """Malformed JSON string is rejected"""
        output = '{"text": "Test", "source_ids": ["src1"'  # Missing closing brackets
        sources = [MockDocument(id="src1", text="Source 1")]
        
        result = LLMGuardrails.validate_llm_output(
            output=output,
            schema=MockLLMOutput,
            sources=sources
        )
        
        assert result.valid is False
        assert result.error == 'json_parse_error'
    
    def test_missing_required_field_rejected(self):
        """Missing required fields are rejected"""
        output = '{"text": "Test"}'  # Missing source_ids
        sources = [MockDocument(id="src1", text="Source 1")]
        
        result = LLMGuardrails.validate_llm_output(
            output=output,
            schema=MockLLMOutput,
            sources=sources
        )
        
        assert result.valid is False
        assert result.error == 'json_schema_violation'


class TestSourceCitationRequirement:
    """Test source citation >=2 requirement (RAG-only principle)"""
    
    def test_zero_sources_rejected(self):
        """Output with zero sources is rejected"""
        output = '{"text": "Test", "source_ids": []}'
        sources = []
        
        result = LLMGuardrails.validate_llm_output(
            output=output,
            schema=MockLLMOutput,
            sources=sources
        )
        
        assert result.valid is False
        assert result.error == 'insufficient_sources'
        assert result.details == 'Require >=2 sources'
    
    def test_one_source_rejected(self):
        """Output with only one source is rejected"""
        output = '{"text": "Test", "source_ids": ["src1"]}'
        sources = [MockDocument(id="src1", text="Source 1")]
        
        result = LLMGuardrails.validate_llm_output(
            output=output,
            schema=MockLLMOutput,
            sources=sources
        )
        
        assert result.valid is False
        assert result.error == 'insufficient_sources'
        assert result.details == 'Require >=2 sources'
    
    def test_two_sources_passes(self):
        """Output with two sources passes"""
        output = '{"text": "Test", "source_ids": ["src1", "src2"]}'
        sources = [
            MockDocument(id="src1", text="Source 1"),
            MockDocument(id="src2", text="Source 2")
        ]
        
        result = LLMGuardrails.validate_llm_output(
            output=output,
            schema=MockLLMOutput,
            sources=sources
        )
        
        assert result.valid is True
    
    def test_invalid_source_id_rejected(self):
        """Source IDs not in provided context are rejected"""
        output = '{"text": "Test", "source_ids": ["src1", "src_invalid"]}'
        sources = [
            MockDocument(id="src1", text="Source 1"),
            MockDocument(id="src2", text="Source 2")
        ]
        
        result = LLMGuardrails.validate_llm_output(
            output=output,
            schema=MockLLMOutput,
            sources=sources
        )
        
        assert result.valid is False
        assert result.error == 'invalid_source_id'
        assert 'src_invalid' in result.details


class TestTemperatureEnforcement:
    """Test temperature <=0.2 enforcement (determinism requirement)"""
    
    def test_temperature_within_limit_passes(self):
        """Temperature <=0.2 passes validation"""
        for temp in [0.0, 0.1, 0.2]:
            result = LLMGuardrails.validate_temperature(temp)
            assert result.valid is True, f"Temperature {temp} should pass"
    
    def test_temperature_above_limit_rejected(self):
        """Temperature >0.2 is rejected"""
        for temp in [0.21, 0.5, 0.7, 1.0]:
            result = LLMGuardrails.validate_temperature(temp)
            assert result.valid is False, f"Temperature {temp} should be rejected"
            assert result.error == 'temperature_exceeds_limit'
            assert result.details == 'Max temperature is 0.2 for determinism'
    
    def test_negative_temperature_rejected(self):
        """Negative temperature is rejected"""
        result = LLMGuardrails.validate_temperature(-0.1)
        assert result.valid is False
        assert result.error == 'invalid_temperature'


class TestToxicityCheck:
    """Test toxicity detection"""
    
    @pytest.mark.asyncio
    async def test_non_toxic_content_passes(self):
        """Non-toxic content passes"""
        guardrails = LLMGuardrails()
        
        class MockOutput:
            text = "This is a helpful response about marketing strategies."
        
        result = await guardrails.validate_llm_output(
            output='{"text": "This is a helpful response"}',
            schema=MockLLMOutput,
            sources=[MockDocument(id="s1", text="test"), MockDocument(id="s2", text="test2")]
        )
        
        assert result.valid is True
    
    @pytest.mark.asyncio
    async def test_toxic_content_rejected(self):
        """Toxic content is rejected"""
        guardrails = LLMGuardrails()
        
        # Mock toxicity detector to return high score
        guardrails.toxicity_detector = lambda text: 0.9
        
        output = '{"text": "Offensive content here", "source_ids": ["s1", "s2"]}'
        sources = [
            MockDocument(id="s1", text="test"),
            MockDocument(id="s2", text="test2")
        ]
        
        result = await guardrails.validate_llm_output(
            output=output,
            schema=MockLLMOutput,
            sources=sources
        )
        
        assert result.valid is False
        assert result.error == 'toxicity_detected'


class TestPIIDetection:
    """Test PII detection in outputs"""
    
    @pytest.mark.asyncio
    async def test_content_without_pii_passes(self):
        """Content without PII passes"""
        guardrails = LLMGuardrails()
        
        output = '{"text": "The campaign performed well", "source_ids": ["s1", "s2"]}'
        sources = [
            MockDocument(id="s1", text="test"),
            MockDocument(id="s2", text="test2")
        ]
        
        result = await guardrails.validate_llm_output(
            output=output,
            schema=MockLLMOutput,
            sources=sources
        )
        
        assert result.valid is True
    
    @pytest.mark.asyncio
    async def test_content_with_pii_rejected(self):
        """Content containing PII is rejected"""
        guardrails = LLMGuardrails()
        
        # Mock PII detector
        guardrails.pii_detector = lambda text: ['email']
        
        output = '{"text": "Contact john@example.com", "source_ids": ["s1", "s2"]}'
        sources = [
            MockDocument(id="s1", text="test"),
            MockDocument(id="s2", text="test2")
        ]
        
        result = await guardrails.validate_llm_output(
            output=output,
            schema=MockLLMOutput,
            sources=sources
        )
        
        assert result.valid is False
        assert result.error == 'pii_detected'
        assert 'email' in result.details


class TestIntegrationScenarios:
    """Integration tests for complete validation flow"""
    
    @pytest.mark.asyncio
    async def test_complete_valid_flow(self):
        """Complete validation flow with all checks passing"""
        guardrails = LLMGuardrails()
        
        output = '{"text": "Based on the analysis, we recommend action X.", "source_ids": ["doc1", "doc2"]}'
        sources = [
            MockDocument(id="doc1", text="Data shows X is effective"),
            MockDocument(id="doc2", text="Previous campaigns confirm this")
        ]
        
        result = await guardrails.validate_llm_output(
            output=output,
            schema=MockLLMOutput,
            sources=sources,
            temperature=0.2
        )
        
        assert result.valid is True
    
    @pytest.mark.asyncio
    async def test_multiple_validation_failures(self):
        """Multiple validation failures are captured"""
        guardrails = LLMGuardrails()
        
        # Invalid JSON + insufficient sources + high temperature
        output = '{"text": "Test"'  # Malformed
        sources = [MockDocument(id="s1", text="only one source")]
        
        result = await guardrails.validate_llm_output(
            output=output,
            schema=MockLLMOutput,
            sources=sources,
            temperature=0.5
        )
        
        assert result.valid is False
        # Should capture first failure encountered
        assert result.error in ['json_parse_error', 'json_schema_violation']
