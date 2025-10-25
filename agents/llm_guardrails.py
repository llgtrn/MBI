"""
LLM Guardrails - Safety and validation layer for LLM outputs
Enforces RAG-only, toxicity filtering, PII detection, source verification
Version: 1.0.0
"""
import re
import hashlib
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, ValidationError
import numpy as np

from contracts.llm_schemas import (
    LLMRequest,
    LLMResponse,
    ValidationResult,
    CrisisBrief,
    CreativeVariant
)


class LLMGuardrails:
    """
    Enforce safety constraints on all LLM calls
    
    Rules:
    - RAG-only: Never use LLM memory
    - min_sources: Require ≥2 sources for factual claims
    - max_temperature: ≤0.2 for determinism
    - toxicity_check: Reject if score >0.8
    - pii_detection: Block email, phone, SSN patterns
    - json_schema_validation: Strict output format
    """
    
    # Mandatory configuration
    RULES = {
        'rag_only': True,
        'min_sources': 2,
        'max_temperature': 0.2,
        'timeout_seconds': 30,
        'max_retries': 2,
        'require_source_ids': True,
        'json_schema_validation': True,
        'toxicity_check': True,
        'pii_detection': True,
        'toxicity_threshold': 0.8,  # >0.8 = reject
    }
    
    # PII regex patterns
    PII_PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
    }
    
    # Toxicity keywords (simple heuristic; in production use ML model)
    TOXIC_KEYWORDS = [
        'kill', 'suicide', 'violence', 'hate', 'attack', 
        'bomb', 'weapon', 'hurt', 'die', 'death'
    ]
    
    def __init__(self):
        """Initialize guardrails with strict validation"""
        self.toxicity_detector = SimpleToxicityDetector()
        self.pii_detector = PIIDetector(self.PII_PATTERNS)
    
    @staticmethod
    def validate_request(request: LLMRequest) -> ValidationResult:
        """
        Validate LLM request before sending
        
        Args:
            request: LLM request to validate
            
        Returns:
            ValidationResult with pass/fail and details
        """
        # Check temperature
        if request.temperature > LLMGuardrails.RULES['max_temperature']:
            return ValidationResult(
                valid=False,
                error='temperature_too_high',
                details=f'Temperature {request.temperature} > {LLMGuardrails.RULES["max_temperature"]}'
            )
        
        # Check minimum sources
        if len(request.context_sources) < LLMGuardrails.RULES['min_sources']:
            return ValidationResult(
                valid=False,
                error='insufficient_sources',
                details=f'Need ≥{LLMGuardrails.RULES["min_sources"]} sources, got {len(request.context_sources)}'
            )
        
        # Verify sources have content
        for source in request.context_sources:
            if not source.get('content') or len(source['content']) == 0:
                return ValidationResult(
                    valid=False,
                    error='empty_source_content',
                    details=f'Source {source.get("id")} has empty content'
                )
        
        return ValidationResult(valid=True)
    
    @staticmethod
    def validate_llm_output(
        output: str,
        schema: type[BaseModel],
        sources: List[Dict[str, str]],
        check_toxicity: bool = True,
        check_pii: bool = True
    ) -> ValidationResult:
        """
        Validate LLM output before use
        
        Args:
            output: Raw LLM output string
            schema: Pydantic schema to validate against
            sources: Context sources provided in request
            check_toxicity: Whether to run toxicity check
            check_pii: Whether to run PII detection
            
        Returns:
            ValidationResult with validation status
        """
        guardrails = LLMGuardrails()
        
        # 1. JSON Schema validation
        try:
            parsed = schema.parse_raw(output)
        except ValidationError as e:
            return ValidationResult(
                valid=False,
                error='json_schema_violation',
                details=str(e)
            )
        
        # 2. Source citation check
        if hasattr(parsed, 'source_ids'):
            if not parsed.source_ids or len(parsed.source_ids) < LLMGuardrails.RULES['min_sources']:
                return ValidationResult(
                    valid=False,
                    error='insufficient_sources',
                    details=f'Require ≥{LLMGuardrails.RULES["min_sources"]} sources'
                )
            
            # 3. Source verification - ensure cited sources exist in context
            source_ids_available = {s['id'] for s in sources}
            source_verification = {}
            
            for source_id in parsed.source_ids:
                exists = source_id in source_ids_available
                source_verification[source_id] = exists
                
                if not exists:
                    return ValidationResult(
                        valid=False,
                        error='invalid_source_id',
                        details=f'Source {source_id} not in provided context',
                        source_verification=source_verification
                    )
        
        # 4. Toxicity check
        toxicity_score = None
        if check_toxicity:
            # Get text content for toxicity check
            text_content = output
            if hasattr(parsed, 'text'):
                text_content = parsed.text
            elif hasattr(parsed, 'reasons'):
                text_content = ' '.join(parsed.reasons)
            
            toxicity_score = guardrails.toxicity_detector.score(text_content)
            
            if toxicity_score > LLMGuardrails.RULES['toxicity_threshold']:
                return ValidationResult(
                    valid=False,
                    error='toxicity_detected',
                    details=f'Score: {toxicity_score:.2f}',
                    toxicity_score=toxicity_score
                )
        
        # 5. PII detection
        pii_found = None
        if check_pii:
            text_content = output
            if hasattr(parsed, 'text'):
                text_content = parsed.text
            
            pii_found = guardrails.pii_detector.detect(text_content)
            
            if pii_found:
                return ValidationResult(
                    valid=False,
                    error='pii_detected',
                    details=f'Found: {", ".join(pii_found)}',
                    pii_found=pii_found
                )
        
        # All checks passed
        return ValidationResult(
            valid=True,
            toxicity_score=toxicity_score,
            pii_found=pii_found,
            source_verification=source_verification if hasattr(parsed, 'source_ids') else None
        )


class SimpleToxicityDetector:
    """
    Simple keyword-based toxicity detector
    In production, use ML model (e.g., Perspective API, Detoxify)
    """
    
    def score(self, text: str) -> float:
        """
        Compute toxicity score [0, 1]
        
        Args:
            text: Text to analyze
            
        Returns:
            Score between 0 (safe) and 1 (toxic)
        """
        text_lower = text.lower()
        
        # Count toxic keyword matches
        matches = sum(1 for keyword in LLMGuardrails.TOXIC_KEYWORDS if keyword in text_lower)
        
        # Normalize to [0, 1] - simple heuristic
        # In production, use calibrated ML model
        score = min(matches / 3.0, 1.0)  # 3+ matches = 1.0
        
        return score


class PIIDetector:
    """Detect PII patterns in text"""
    
    def __init__(self, patterns: Dict[str, str]):
        """
        Initialize with regex patterns
        
        Args:
            patterns: Dict of pattern_name -> regex
        """
        self.patterns = {
            name: re.compile(pattern) 
            for name, pattern in patterns.items()
        }
    
    def detect(self, text: str) -> Optional[List[str]]:
        """
        Detect PII in text
        
        Args:
            text: Text to analyze
            
        Returns:
            List of PII types found, or None if clean
        """
        found = []
        
        for pii_type, pattern in self.patterns.items():
            if pattern.search(text):
                found.append(pii_type)
        
        return found if found else None


# Helper function for hashing (used in audit logs)
def hash_content(content: str) -> str:
    """
    Generate SHA-256 hash of content
    
    Args:
        content: Content to hash
        
    Returns:
        Hex digest of hash
    """
    return hashlib.sha256(content.encode('utf-8')).hexdigest()
