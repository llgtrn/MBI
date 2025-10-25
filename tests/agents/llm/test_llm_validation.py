"""
LLM Guardrails Validation Tests

Tests for JSON schema validation, source verification, and safety checks.
Related: Q_004, A_004
"""

import pytest
import json
from pydantic import ValidationError
from typing import List

from src.agents.llm.llm_guardrails import LLMGuardrails, ValidationResult
from src.agents.llm.schemas import CrisisBrief, CreativeVariant, Document


class TestLLMGuardrails:
    """Test LLM output validation and safety checks"""
    
    @pytest.fixture
    def guardrails(self):
        return LLMGuardrails()
    
    @pytest.fixture
    def sample_sources(self):
        return [
            Document(id="src_1", url="https://example.com/news1", text="Sample news 1"),
            Document(id="src_2", url="https://example.com/news2", text="Sample news 2"),
            Document(id="src_3", url="https://official.com/pr", text="Official statement")
        ]
    
    def test_valid_json_schema_passes(self, guardrails, sample_sources):
        """ACCEPTANCE: Valid Pydantic schema output passes validation"""
        output = json.dumps({
            "topic_id": "crisis_001",
            "stance": "neutral",
            "risk_score": 0.5,
            "reasons": ["Reason 1", "Reason 2"],
            "actions": ["verify_official"],
            "sources": [
                {"id": "src_1", "url": "https://example.com/news1", "quote": "Quote 1"},
                {"id": "src_2", "url": "https://example.com/news2", "quote": "Quote 2"}
            ],
            "requires_human_review": False
        })
        
        result = guardrails.validate_llm_output(
            output=output,
            schema=CrisisBrief,
            sources=sample_sources
        )
        
        assert result.valid is True
        assert result.error is None
    
    def test_invalid_json_schema_fails(self, guardrails, sample_sources):
        """ACCEPTANCE: Invalid schema rejected with clear error"""
        output = json.dumps({
            "topic_id": "crisis_001",
            "risk_score": 0.5,
            "reasons": ["Reason 1"],
            "sources": [{"id": "src_1", "url": "https://example.com", "quote": "Q"}]
        })
        
        result = guardrails.validate_llm_output(
            output=output,
            schema=CrisisBrief,
            sources=sample_sources
        )
        
        assert result.valid is False
        assert result.error == 'json_schema_violation'
    
    def test_minimum_sources_enforced(self, guardrails, sample_sources):
        """ACCEPTANCE: Require ≥2 sources for factual claims"""
        output = json.dumps({
            "topic_id": "crisis_001",
            "stance": "neutral",
            "risk_score": 0.3,
            "reasons": ["Single reason"],
            "actions": ["monitor"],
            "sources": [
                {"id": "src_1", "url": "https://example.com", "quote": "Quote"}
            ],
            "requires_human_review": False
        })
        
        result = guardrails.validate_llm_output(
            output=output,
            schema=CrisisBrief,
            sources=sample_sources
        )
        
        assert result.valid is False
        assert result.error == 'insufficient_sources'
    
    def test_source_verification_invalid_id(self, guardrails, sample_sources):
        """ACCEPTANCE: Reject outputs referencing non-existent source IDs"""
        output = json.dumps({
            "topic_id": "crisis_001",
            "stance": "neutral",
            "risk_score": 0.4,
            "reasons": ["R1", "R2"],
            "actions": ["verify"],
            "sources": [
                {"id": "src_1", "url": "https://example.com", "quote": "Q1"},
                {"id": "src_999", "url": "https://fake.com", "quote": "Q2"}
            ],
            "requires_human_review": False
        })
        
        result = guardrails.validate_llm_output(
            output=output,
            schema=CrisisBrief,
            sources=sample_sources
        )
        
        assert result.valid is False
        assert result.error == 'invalid_source_id'


class TestLLMSchemas:
    """Test Pydantic schemas for LLM outputs"""
    
    def test_crisis_brief_schema_complete(self):
        """ACCEPTANCE: CrisisBrief schema has all required fields"""
        brief = CrisisBrief(
            topic_id="crisis_001",
            stance="neutral",
            risk_score=0.5,
            reasons=["Reason 1", "Reason 2"],
            actions=["verify_official"],
            sources=[
                {"id": "src_1", "url": "https://example.com", "quote": "Quote"}
            ],
            requires_human_review=False
        )
        
        assert brief.topic_id == "crisis_001"
        assert brief.stance == "neutral"
        assert 0 <= brief.risk_score <= 1
    
    def test_creative_variant_schema(self):
        """ACCEPTANCE: CreativeVariant schema validates correctly"""
        variant = CreativeVariant(
            text="(Promo/広告) Great product!",
            cta={"label": "Buy Now", "url": "https://shop.example.com"},
            policy_approved=True,
            source_ids=["src_1", "src_2"]
        )
        
        assert "(Promo" in variant.text or "(広告)" in variant.text
        assert len(variant.source_ids) >= 2
