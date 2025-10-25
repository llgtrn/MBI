"""
LLM Output Schemas - Pydantic Models

Strict schemas for all LLM outputs to ensure type safety and validation.
Related: Q_004, A_004, C05_LLMCouncil
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Literal, Optional


class Document(BaseModel):
    """Source document for RAG"""
    id: str
    url: str
    text: str
    title: Optional[str] = None


class SourceReference(BaseModel):
    """Reference to a source with quote"""
    id: str = Field(description="Source document ID")
    url: str = Field(description="Source URL")
    quote: str = Field(description="Relevant quote from source")


class CrisisBrief(BaseModel):
    """
    LLM output schema for crisis detection.
    
    Enforces:
    - Valid stance enum
    - Risk score in [0, 1]
    - Minimum 2 sources
    - All required fields present
    """
    
    topic_id: str = Field(description="Unique topic identifier")
    
    stance: Literal["against", "for", "neutral", "unclear"] = Field(
        description="Brand stance in the discourse"
    )
    
    risk_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Risk score from 0 (no risk) to 1 (critical)"
    )
    
    reasons: List[str] = Field(
        min_items=2,
        max_items=4,
        description="2-4 concise reasons with evidence"
    )
    
    actions: List[str] = Field(
        min_items=1,
        description="Recommended actions (e.g., verify_official, pause_promo)"
    )
    
    sources: List[SourceReference] = Field(
        min_items=2,
        description="Source references (minimum 2 for factual claims)"
    )
    
    requires_human_review: bool = Field(
        description="Whether human review is required"
    )
    
    @validator('reasons')
    def validate_reasons_length(cls, v):
        """Ensure reasons are concise"""
        for reason in v:
            if len(reason) > 200:
                raise ValueError("Each reason must be ≤200 characters")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "topic_id": "crisis_001",
                "stance": "neutral",
                "risk_score": 0.65,
                "reasons": [
                    "No official statement from company",
                    "Multiple unverified social media posts"
                ],
                "actions": ["verify_official", "monitor"],
                "sources": [
                    {
                        "id": "src_1",
                        "url": "https://example.com/news",
                        "quote": "Reports suggest..."
                    },
                    {
                        "id": "src_2",
                        "url": "https://twitter.com/source",
                        "quote": "According to users..."
                    }
                ],
                "requires_human_review": True
            }
        }


class CreativeVariant(BaseModel):
    """
    LLM output schema for creative variant generation.
    
    Enforces:
    - Promo label present
    - Valid CTA structure
    - Minimum 2 sources
    """
    
    text: str = Field(
        max_length=220,
        description="Creative copy text (max 220 chars, must include Promo/広告 label)"
    )
    
    cta: Dict[str, str] = Field(
        description="Call to action with label and url"
    )
    
    policy_approved: bool = Field(
        description="Whether content passes policy checks"
    )
    
    source_ids: List[str] = Field(
        min_items=2,
        description="Source document IDs used (minimum 2)"
    )
    
    @validator('text')
    def validate_promo_label(cls, v):
        """Ensure promotional label is present"""
        if not any(label in v for label in ["Promo", "広告", "Quảng cáo", "推广"]):
            raise ValueError("Creative text must include promotional label (Promo/広告/etc.)")
        return v
    
    @validator('cta')
    def validate_cta_structure(cls, v):
        """Ensure CTA has required fields"""
        if 'label' not in v or 'url' not in v:
            raise ValueError("CTA must have 'label' and 'url' fields")
        if not v['url'].startswith('http'):
            raise ValueError("CTA URL must be valid HTTP(S) URL")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "text": "(Promo/広告) Discover amazing products at unbeatable prices!",
                "cta": {
                    "label": "Shop Now",
                    "url": "https://shop.example.com"
                },
                "policy_approved": True,
                "source_ids": ["src_3", "src_7"]
            }
        }


class ExecutiveBrief(BaseModel):
    """Executive summary output schema"""
    
    week_id: str = Field(description="Week identifier (e.g., 2025-W42)")
    
    summary: List[str] = Field(
        min_items=3,
        max_items=5,
        description="3-5 bullet point executive summary"
    )
    
    key_wins: List[str] = Field(
        description="Key accomplishments"
    )
    
    concerns: List[str] = Field(
        description="Areas requiring attention"
    )
    
    action_items: List[str] = Field(
        description="Recommended actions for next week"
    )
    
    source_ids: List[str] = Field(
        min_items=1,
        description="Source references from SSOT"
    )


class QAResponse(BaseModel):
    """Internal Q&A response schema"""
    
    answer: str = Field(
        description="Answer to user's question"
    )
    
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score in answer (0-1)"
    )
    
    source_ids: List[str] = Field(
        min_items=1,
        description="Source references from SSOT"
    )
    
    action: Optional[str] = Field(
        default=None,
        description="Recommended action if answer is unclear (e.g., consult_expert)"
    )
