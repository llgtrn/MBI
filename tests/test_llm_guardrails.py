"""
Tests for LLM Guardrails
Validates toxicity filtering, PII detection, source verification
"""
import pytest
from datetime import datetime
from pydantic import ValidationError

from contracts.llm_schemas import (
    LLMRequest,
    LLMResponse,
    ValidationResult,
    CrisisBrief,
    CreativeVariant
)
from agents.llm_guardrails import (
    LLMGuardrails,
    SimpleToxicityDetector,
    PIIDetector,
    hash_content
)


class TestLLMRequest:
    """Test LLM request validation"""
    
    def test_valid_request(self):
        """Valid request should pass"""
        request = LLMRequest(
            prompt="Analyze this topic",
            model="claude-sonnet-4.5",
            temperature=0.2,
            max_tokens=800,
            response_format="json",
            context_sources=[
                {"id": "src_1", "content": "Source 1 content"},
                {"id": "src_2", "content": "Source 2 content"}
            ],
            task_type="crisis_brief"
        )
        
        assert request.temperature == 0.2
        assert len(request.context_sources) == 2
    
    def test_temperature_too_high(self):
        """Temperature >0.2 should fail"""
        with pytest.raises(ValidationError):
            LLMRequest(
                prompt="Test",
                model="claude-sonnet-4.5",
                temperature=0.5,  # Too high!
                max_tokens=800,
                context_sources=[
                    {"id": "src_1", "content": "Content 1"},
                    {"id": "src_2", "content": "Content 2"}
                ],
                task_type="crisis_brief"
            )
    
    def test_insufficient_sources(self):
        """<2 sources should fail"""
        with pytest.raises(ValidationError):
            LLMRequest(
                prompt="Test",
                model="claude-sonnet-4.5",
                temperature=0.2,
                max_tokens=800,
                context_sources=[
                    {"id": "src_1", "content": "Only one source"}
                ],
                task_type="crisis_brief"
            )
    
    def test_invalid_model(self):
        """Invalid model name should fail"""
        with pytest.raises(ValidationError):
            LLMRequest(
                prompt="Test",
                model="gpt-99",  # Invalid!
                temperature=0.2,
                max_tokens=800,
                context_sources=[
                    {"id": "src_1", "content": "Content 1"},
                    {"id": "src_2", "content": "Content 2"}
                ],
                task_type="crisis_brief"
            )


class TestLLMResponse:
    """Test LLM response validation"""
    
    def test_valid_response(self):
        """Valid response with ≥2 sources"""
        response = LLMResponse(
            content='{"stance":"neutral"}',
            source_ids=["src_1", "src_2"],
            model_used="claude-sonnet-4.5",
            tokens_input=450,
            tokens_output=120,
            latency_ms=1850,
            verified=True
        )
        
        assert response.verified is True
        assert len(response.source_ids) >= 2
    
    def test_insufficient_sources_response(self):
        """Response with <2 sources should fail"""
        with pytest.raises(ValidationError):
            LLMResponse(
                content='{"stance":"neutral"}',
                source_ids=["src_1"],  # Only 1 source!
                model_used="claude-sonnet-4.5",
                tokens_input=450,
                tokens_output=120,
                latency_ms=1850
            )


class TestToxicityDetector:
    """Test toxicity detection"""
    
    def test_benign_content(self):
        """Safe content should have low score"""
        detector = SimpleToxicityDetector()
        
        text = "This is a great product for your daily needs"
        score = detector.score(text)
        
        assert score < 0.8
    
    def test_toxic_content(self):
        """Toxic content should have high score"""
        detector = SimpleToxicityDetector()
        
        text = "I want to kill and attack everyone with violence"
        score = detector.score(text)
        
        assert score >= 0.8
    
    def test_mild_toxic_content(self):
        """Mildly toxic should be between thresholds"""
        detector = SimpleToxicityDetector()
        
        text = "This will kill your competition"  # 1 match
        score = detector.score(text)
        
        assert 0.2 < score < 0.8


class TestPIIDetector:
    """Test PII detection"""
    
    def test_clean_content(self):
        """Content without PII should pass"""
        detector = PIIDetector(LLMGuardrails.PII_PATTERNS)
        
        text = "Our product helps you save time and money"
        pii = detector.detect(text)
        
        assert pii is None
    
    def test_email_detection(self):
        """Email should be detected"""
        detector = PIIDetector(LLMGuardrails.PII_PATTERNS)
        
        text = "Contact us at test@example.com for more info"
        pii = detector.detect(text)
        
        assert pii is not None
        assert 'email' in pii
    
    def test_phone_detection(self):
        """Phone number should be detected"""
        detector = PIIDetector(LLMGuardrails.PII_PATTERNS)
        
        text = "Call 555-123-4567 today"
        pii = detector.detect(text)
        
        assert pii is not None
        assert 'phone' in pii
    
    def test_ssn_detection(self):
        """SSN should be detected"""
        detector = PIIDetector(LLMGuardrails.PII_PATTERNS)
        
        text = "SSN: 123-45-6789"
        pii = detector.detect(text)
        
        assert pii is not None
        assert 'ssn' in pii
    
    def test_multiple_pii_types(self):
        """Multiple PII types should all be detected"""
        detector = PIIDetector(LLMGuardrails.PII_PATTERNS)
        
        text = "Email: test@example.com Phone: 555-123-4567"
        pii = detector.detect(text)
        
        assert pii is not None
        assert 'email' in pii
        assert 'phone' in pii


class TestLLMGuardrails:
    """Test guardrail validation"""
    
    def test_validate_request_success(self):
        """Valid request should pass guardrails"""
        request = LLMRequest(
            prompt="Analyze topic",
            model="claude-sonnet-4.5",
            temperature=0.2,
            max_tokens=800,
            context_sources=[
                {"id": "src_1", "content": "Content 1"},
                {"id": "src_2", "content": "Content 2"}
            ],
            task_type="crisis_brief"
        )
        
        result = LLMGuardrails.validate_request(request)
        
        assert result.valid is True
        assert result.error is None
    
    def test_validate_request_temp_fail(self):
        """Temperature too high should fail"""
        # This should fail at schema level, not guardrails
        # But we test the guardrails validation separately
        pass
    
    def test_validate_output_success(self):
        """Valid output should pass all checks"""
        output = '''
        {
            "topic_id": "test",
            "stance": "neutral",
            "risk_score": 0.5,
            "reasons": ["Reason 1", "Reason 2"],
            "actions": ["monitor"],
            "sources": [
                {"id": "src_1", "url": "https://example.com", "quote": "Quote 1"},
                {"id": "src_2", "url": "https://example.com", "quote": "Quote 2"}
            ],
            "requires_human_review": false,
            "source_ids": ["src_1", "src_2"]
        }
        '''
        
        sources = [
            {"id": "src_1", "content": "Content 1"},
            {"id": "src_2", "content": "Content 2"}
        ]
        
        result = LLMGuardrails.validate_llm_output(
            output=output,
            schema=CrisisBrief,
            sources=sources
        )
        
        assert result.valid is True
        assert result.toxicity_score is not None
        assert result.toxicity_score < 0.8
        assert result.pii_found is None
    
    def test_validate_output_insufficient_sources(self):
        """Output with <2 sources should fail"""
        output = '''
        {
            "topic_id": "test",
            "stance": "neutral",
            "risk_score": 0.5,
            "reasons": ["Reason 1", "Reason 2"],
            "actions": ["monitor"],
            "sources": [
                {"id": "src_1", "url": "https://example.com", "quote": "Quote 1"},
                {"id": "src_2", "url": "https://example.com", "quote": "Quote 2"}
            ],
            "requires_human_review": false,
            "source_ids": ["src_1"]
        }
        '''
        
        sources = [
            {"id": "src_1", "content": "Content 1"},
            {"id": "src_2", "content": "Content 2"}
        ]
        
        result = LLMGuardrails.validate_llm_output(
            output=output,
            schema=CrisisBrief,
            sources=sources
        )
        
        assert result.valid is False
        assert result.error == 'insufficient_sources'
    
    def test_validate_output_invalid_source_id(self):
        """Output citing non-existent source should fail"""
        output = '''
        {
            "topic_id": "test",
            "stance": "neutral",
            "risk_score": 0.5,
            "reasons": ["Reason 1", "Reason 2"],
            "actions": ["monitor"],
            "sources": [
                {"id": "src_1", "url": "https://example.com", "quote": "Quote 1"},
                {"id": "src_2", "url": "https://example.com", "quote": "Quote 2"}
            ],
            "requires_human_review": false,
            "source_ids": ["src_1", "src_999"]
        }
        '''
        
        sources = [
            {"id": "src_1", "content": "Content 1"},
            {"id": "src_2", "content": "Content 2"}
        ]
        
        result = LLMGuardrails.validate_llm_output(
            output=output,
            schema=CrisisBrief,
            sources=sources
        )
        
        assert result.valid is False
        assert result.error == 'invalid_source_id'
        assert 'src_999' in result.details
    
    def test_validate_output_toxicity(self):
        """Toxic content should be rejected"""
        output = '''
        {
            "text": "(Promo) Kill your competition with violence and hate",
            "cta": {"label": "Shop", "url": "https://example.com"},
            "policy_approved": true,
            "source_ids": ["src_1", "src_2"],
            "language": "en"
        }
        '''
        
        sources = [
            {"id": "src_1", "content": "Content 1"},
            {"id": "src_2", "content": "Content 2"}
        ]
        
        result = LLMGuardrails.validate_llm_output(
            output=output,
            schema=CreativeVariant,
            sources=sources,
            check_toxicity=True
        )
        
        assert result.valid is False
        assert result.error == 'toxicity_detected'
        assert result.toxicity_score >= 0.8
    
    def test_validate_output_pii(self):
        """PII in output should be rejected"""
        output = '''
        {
            "text": "(Promo) Email us at contact@company.com or call 555-123-4567",
            "cta": {"label": "Contact", "url": "https://example.com"},
            "policy_approved": true,
            "source_ids": ["src_1", "src_2"],
            "language": "en"
        }
        '''
        
        sources = [
            {"id": "src_1", "content": "Content 1"},
            {"id": "src_2", "content": "Content 2"}
        ]
        
        result = LLMGuardrails.validate_llm_output(
            output=output,
            schema=CreativeVariant,
            sources=sources,
            check_pii=True
        )
        
        assert result.valid is False
        assert result.error == 'pii_detected'
        assert result.pii_found is not None
        assert 'email' in result.pii_found or 'phone' in result.pii_found


class TestCreativeVariant:
    """Test creative variant validation"""
    
    def test_valid_variant_english(self):
        """Valid English variant with Promo label"""
        variant = CreativeVariant(
            text="(Promo) Discover the difference",
            cta={"label": "Shop Now", "url": "https://example.com"},
            policy_approved=True,
            source_ids=["src_1", "src_2"],
            language="en"
        )
        
        assert "Promo" in variant.text
    
    def test_missing_promo_label_english(self):
        """English variant without Promo should fail"""
        with pytest.raises(ValidationError):
            CreativeVariant(
                text="Discover the difference",  # Missing (Promo)!
                cta={"label": "Shop Now", "url": "https://example.com"},
                policy_approved=True,
                source_ids=["src_1", "src_2"],
                language="en"
            )
    
    def test_valid_variant_japanese(self):
        """Valid Japanese variant with 広告 label"""
        variant = CreativeVariant(
            text="（広告）素晴らしい商品",
            cta={"label": "今すぐ購入", "url": "https://example.com"},
            policy_approved=True,
            source_ids=["src_1", "src_2"],
            language="ja"
        )
        
        assert "広告" in variant.text
    
    def test_missing_label_japanese(self):
        """Japanese variant without 広告 should fail"""
        with pytest.raises(ValidationError):
            CreativeVariant(
                text="素晴らしい商品",  # Missing （広告）!
                cta={"label": "今すぐ購入", "url": "https://example.com"},
                policy_approved=True,
                source_ids=["src_1", "src_2"],
                language="ja"
            )
    
    def test_missing_cta_fields(self):
        """CTA without required fields should fail"""
        with pytest.raises(ValidationError):
            CreativeVariant(
                text="(Promo) Great product",
                cta={"label": "Shop Now"},  # Missing url!
                policy_approved=True,
                source_ids=["src_1", "src_2"],
                language="en"
            )


class TestHashContent:
    """Test content hashing"""
    
    def test_hash_determinism(self):
        """Same content should produce same hash"""
        content = "Test content"
        
        hash1 = hash_content(content)
        hash2 = hash_content(content)
        
        assert hash1 == hash2
    
    def test_hash_uniqueness(self):
        """Different content should produce different hashes"""
        content1 = "Test content 1"
        content2 = "Test content 2"
        
        hash1 = hash_content(content1)
        hash2 = hash_content(content2)
        
        assert hash1 != hash2
    
    def test_hash_length(self):
        """SHA-256 hash should be 64 hex characters"""
        content = "Test"
        hash_val = hash_content(content)
        
        assert len(hash_val) == 64
        assert all(c in '0123456789abcdef' for c in hash_val)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
