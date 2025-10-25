"""
LLM Schemas - Contracts for LLM Council Architecture
Version: 1.0.0
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime


class LLMRequest(BaseModel):
    """Request schema for LLM API calls"""
    
    prompt: str = Field(..., min_length=1, max_length=8000)
    model: str = Field(..., regex="^(claude-sonnet-4.5|claude-opus-4|local_.*|gpt-4.*)$")
    temperature: float = Field(0.2, ge=0.0, le=0.2, description="Max 0.2 for determinism")
    max_tokens: int = Field(800, ge=100, le=4000)
    response_format: str = Field("json", regex="^(json|text)$")
    context_sources: List[Dict[str, str]] = Field(
        ..., 
        min_items=2,
        description="RAG sources with id and content"
    )
    task_type: str = Field(
        ..., 
        regex="^(crisis_brief|creative_variants|stance_analysis|qa_internal|policy_check|mmm_explainer|briefing|tagging)$"
    )
    
    @validator('context_sources')
    def validate_sources(cls, v):
        """Ensure sources have required fields"""
        for source in v:
            if 'id' not in source or 'content' not in source:
                raise ValueError("Each source must have 'id' and 'content'")
            if len(source['id']) == 0:
                raise ValueError("Source id cannot be empty")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "SYSTEM: Analyze stance...",
                "model": "claude-sonnet-4.5",
                "temperature": 0.2,
                "max_tokens": 800,
                "response_format": "json",
                "context_sources": [
                    {"id": "src_1", "content": "Brand X announced..."},
                    {"id": "src_2", "content": "Competitor Y reported..."}
                ],
                "task_type": "stance_analysis"
            }
        }


class LLMResponse(BaseModel):
    """Response schema from LLM with verification"""
    
    content: str = Field(..., min_length=1)
    source_ids: List[str] = Field(
        ..., 
        min_items=2,
        description="Source citations (≥2 required)"
    )
    model_used: str
    tokens_input: int = Field(..., ge=0)
    tokens_output: int = Field(..., ge=0)
    latency_ms: int = Field(..., ge=0)
    verified: bool = Field(False, description="Set by Verifier LLM")
    verification_notes: Optional[str] = None
    timestamp_utc: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('source_ids')
    def validate_source_count(cls, v):
        """Enforce ≥2 sources for factual claims"""
        if len(v) < 2:
            raise ValueError("Factual claims require ≥2 source citations")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "content": '{"stance":"neutral","risk_score":0.45}',
                "source_ids": ["src_1", "src_2"],
                "model_used": "claude-sonnet-4.5",
                "tokens_input": 450,
                "tokens_output": 120,
                "latency_ms": 1850,
                "verified": True,
                "verification_notes": "Facts cross-checked",
                "timestamp_utc": "2025-10-19T19:00:00Z"
            }
        }


class ValidationResult(BaseModel):
    """Result of LLM output validation"""
    
    valid: bool
    error: Optional[str] = Field(
        None,
        regex="^(json_schema_violation|insufficient_sources|invalid_source_id|toxicity_detected|pii_detected)?$"
    )
    details: Optional[str] = None
    toxicity_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    pii_found: Optional[List[str]] = Field(None, description="Types of PII detected")
    source_verification: Optional[Dict[str, bool]] = Field(
        None,
        description="Map of source_id to exists_in_context"
    )
    
    class Config:
        schema_extra = {
            "example_pass": {
                "valid": True,
                "error": None,
                "toxicity_score": 0.12,
                "pii_found": None,
                "source_verification": {"src_1": True, "src_2": True}
            },
            "example_fail": {
                "valid": False,
                "error": "toxicity_detected",
                "details": "Score: 0.87",
                "toxicity_score": 0.87,
                "pii_found": None,
                "source_verification": {"src_1": True, "src_2": True}
            }
        }


class CrisisBrief(BaseModel):
    """LLM output for crisis detection"""
    
    topic_id: str
    stance: str = Field(..., regex="^(against|for|neutral|unclear)$")
    risk_score: float = Field(..., ge=0.0, le=1.0)
    reasons: List[str] = Field(..., min_items=2, max_items=4)
    actions: List[str] = Field(..., min_items=1)
    sources: List[Dict[str, str]] = Field(..., min_items=2)
    requires_human_review: bool
    source_ids: List[str] = Field(..., min_items=2, description="RAG sources used")
    
    @validator('sources')
    def validate_sources(cls, v):
        """Ensure sources have required fields"""
        for source in v:
            required = {'id', 'url', 'quote'}
            if not required.issubset(source.keys()):
                raise ValueError(f"Source must have fields: {required}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "topic_id": "cl_20251012_001",
                "stance": "neutral",
                "risk_score": 0.45,
                "reasons": [
                    "No official statement from brand",
                    "Unverified claims from low-trust sources"
                ],
                "actions": ["verify_official", "monitor"],
                "sources": [
                    {"id": "src_1", "url": "https://...", "quote": "Brand X..."},
                    {"id": "src_2", "url": "https://...", "quote": "Report states..."}
                ],
                "requires_human_review": False,
                "source_ids": ["src_1", "src_2"]
            }
        }


class CreativeVariant(BaseModel):
    """LLM-generated creative variant"""
    
    text: str = Field(..., min_length=10, max_length=220)
    cta: Dict[str, str] = Field(..., description="Call-to-action with label and url")
    policy_approved: bool
    source_ids: List[str] = Field(..., min_items=2)
    language: str = Field("en", regex="^(en|ja|vi|zh|ko)$")
    
    @validator('text')
    def validate_promo_label(cls, v, values):
        """Ensure promotional content has proper labeling"""
        lang = values.get('language', 'en')
        labels = {
            'en': 'Promo',
            'ja': '広告',
            'vi': 'Quảng cáo',
            'zh': '广告',
            'ko': '광고'
        }
        if lang in labels and labels[lang] not in v:
            raise ValueError(f"Promotional content must include '{labels[lang]}' label")
        return v
    
    @validator('cta')
    def validate_cta(cls, v):
        """Ensure CTA has required fields"""
        if 'label' not in v or 'url' not in v:
            raise ValueError("CTA must have 'label' and 'url'")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "text": "(Promo) Discover the difference...",
                "cta": {"label": "Shop Now", "url": "https://..."},
                "policy_approved": True,
                "source_ids": ["ssot_doc_3", "ssot_doc_7"],
                "language": "en"
            }
        }


class LLMCallLog(BaseModel):
    """Audit log entry for LLM API calls"""
    
    call_id: str
    timestamp_utc: datetime
    agent_name: str
    model: str
    task_type: str
    prompt_hash: str = Field(..., description="SHA-256 hash of prompt")
    output_hash: str = Field(..., description="SHA-256 hash of output")
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_usd: float = Field(..., ge=0.0)
    source_ids: List[str]
    verified: bool
    validation_result: Optional[ValidationResult] = None
    
    class Config:
        schema_extra = {
            "example": {
                "call_id": "llm_call_abc123",
                "timestamp_utc": "2025-10-19T19:00:00Z",
                "agent_name": "CrisisDetectionAgent",
                "model": "claude-sonnet-4.5",
                "task_type": "crisis_brief",
                "prompt_hash": "a1b2c3...",
                "output_hash": "d4e5f6...",
                "input_tokens": 450,
                "output_tokens": 120,
                "latency_ms": 1850,
                "cost_usd": 0.015,
                "source_ids": ["src_1", "src_2"],
                "verified": True,
                "validation_result": None
            }
        }
