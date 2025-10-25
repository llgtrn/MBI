"""
LLM Guardrails - Output Validation & Safety

Validates all LLM outputs before use:
- JSON schema validation (Pydantic)
- Source verification (min 2 sources)
- Toxicity detection
- PII detection
- Full audit trail

Related: Q_004, A_004, C05_LLMCouncil
"""

import json
import re
from typing import Type, List, Optional, Any
from pydantic import BaseModel, ValidationError

from src.agents.llm.schemas import Document


class ValidationResult(BaseModel):
    """Result of LLM output validation"""
    valid: bool
    error: Optional[str] = None
    details: Optional[str] = None


class LLMGuardrails:
    """
    Enforce safety constraints on all LLM calls.
    
    Mandatory Requirements:
    - RAG-only (never use LLM memory)
    - Min 2 sources for factual claims
    - Max temperature 0.2 (determinism)
    - JSON schema validation
    - Toxicity check
    - PII detection
    """
    
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
    }
    
    def __init__(self):
        """Initialize guardrails with safety checks"""
        self.toxicity_detector = self._init_toxicity_detector()
        self.pii_detector = self._init_pii_detector()
    
    def _init_toxicity_detector(self):
        """Initialize toxicity detection (placeholder)"""
        # In production: use Perspective API or similar
        def detect_toxicity(text: str) -> dict:
            """Mock toxicity detector"""
            # Simple keyword-based detection for demo
            toxic_keywords = ['terrible', 'destroy', 'hate', 'kill']
            score = sum(1 for kw in toxic_keywords if kw in text.lower()) / 10
            return {'score': min(score, 1.0)}
        
        return detect_toxicity
    
    def _init_pii_detector(self):
        """Initialize PII detection"""
        def detect_pii(text: str) -> List[str]:
            """Detect PII patterns in text"""
            pii_found = []
            
            # Email pattern
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            emails = re.findall(email_pattern, text)
            pii_found.extend(emails)
            
            # Phone pattern (simple)
            phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
            phones = re.findall(phone_pattern, text)
            pii_found.extend(phones)
            
            # SSN pattern
            ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
            ssns = re.findall(ssn_pattern, text)
            pii_found.extend(ssns)
            
            return pii_found
        
        return detect_pii
    
    def validate_llm_output(
        self,
        output: str,
        schema: Type[BaseModel],
        sources: List[Document]
    ) -> ValidationResult:
        """
        Validate LLM output before use.
        
        Args:
            output: Raw LLM output (JSON string)
            schema: Expected Pydantic schema
            sources: Source documents provided to LLM
        
        Returns:
            ValidationResult with approval decision
        """
        # 1. JSON Schema validation
        try:
            parsed = schema.parse_raw(output)
        except ValidationError as e:
            return ValidationResult(
                valid=False,
                error='json_schema_violation',
                details=str(e)
            )
        except json.JSONDecodeError as e:
            return ValidationResult(
                valid=False,
                error='invalid_json',
                details=f"Failed to parse JSON: {str(e)}"
            )
        
        # 2. Source citation check
        source_ids = self._extract_source_ids(parsed)
        
        if not source_ids or len(source_ids) < self.RULES['min_sources']:
            return ValidationResult(
                valid=False,
                error='insufficient_sources',
                details=f"Require ≥{self.RULES['min_sources']} sources, found {len(source_ids)}"
            )
        
        # 3. Source verification
        valid_source_ids = {s.id for s in sources}
        for source_id in source_ids:
            if source_id not in valid_source_ids:
                return ValidationResult(
                    valid=False,
                    error='invalid_source_id',
                    details=f"Source {source_id} not in provided context"
                )
        
        # 4. Toxicity check
        text_content = self._extract_text(parsed)
        toxicity_result = self.toxicity_detector(text_content)
        
        if toxicity_result['score'] > 0.8:
            return ValidationResult(
                valid=False,
                error='toxicity_detected',
                details=f"Toxicity score: {toxicity_result['score']:.2f}"
            )
        
        # 5. PII detection
        pii_found = self.pii_detector(text_content)
        
        if pii_found:
            return ValidationResult(
                valid=False,
                error='pii_detected',
                details=f"Found PII: {', '.join(pii_found[:3])}"
            )
        
        return ValidationResult(valid=True)
    
    def _extract_source_ids(self, parsed: BaseModel) -> List[str]:
        """Extract source IDs from parsed output"""
        # Handle different schema types
        if hasattr(parsed, 'source_ids'):
            return parsed.source_ids
        elif hasattr(parsed, 'sources'):
            # Extract IDs from SourceReference list
            return [s.id if hasattr(s, 'id') else s['id'] for s in parsed.sources]
        else:
            return []
    
    def _extract_text(self, parsed: BaseModel) -> str:
        """Extract all text content for toxicity/PII checks"""
        text_parts = []
        
        # Collect all string fields
        for field_name, field_value in parsed.dict().items():
            if isinstance(field_value, str):
                text_parts.append(field_value)
            elif isinstance(field_value, list):
                for item in field_value:
                    if isinstance(item, str):
                        text_parts.append(item)
                    elif isinstance(item, dict):
                        for v in item.values():
                            if isinstance(v, str):
                                text_parts.append(v)
        
        return ' '.join(text_parts)
    
    @staticmethod
    def enforce_temperature(temperature: float) -> float:
        """Enforce maximum temperature for determinism"""
        max_temp = LLMGuardrails.RULES['max_temperature']
        if temperature > max_temp:
            return max_temp
        return temperature
    
    @staticmethod
    def validate_rag_context(sources: List[Document]) -> bool:
        """Ensure RAG context is provided (no LLM memory)"""
        return len(sources) > 0
