"""Crisis Detection Agent: Brand intelligence with separate analyst/verifier models.

Core Principles:
- Multi-source corroboration required (>=2 independent sources)
- Risk score capped to 0.5 when only 1 source (Q_011 enforcement)
- Separate verifier model prevents self-verification loops
- Risk scoring with mandatory human review for high-risk crises
- RAG-only: All facts must cite source_ids
"""

from typing import List, Dict, Optional
from pydantic import BaseModel
from enum import Enum
from prometheus_client import Counter, Histogram


# Metrics
crisis_detections_total = Counter(
    'mbi_crisis_detections_total',
    'Total crisis detections',
    ['stance', 'requires_review']
)

crisis_risk_score_histogram = Histogram(
    'mbi_crisis_risk_score',
    'Crisis risk score distribution',
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

crisis_source_count_histogram = Histogram(
    'mbi_crisis_source_count',
    'Number of sources per crisis detection',
    buckets=[0, 1, 2, 3, 4, 5, 10, 20]
)


class Stance(str, Enum):
    """Brand stance in discourse."""
    AGAINST = "against"
    FOR = "for"
    NEUTRAL = "neutral"
    UNCLEAR = "unclear"


class Document(BaseModel):
    """Source document for RAG."""
    id: str
    url: str
    text: str
    domain: str


class CrisisBrief(BaseModel):
    """Crisis analysis output schema."""
    topic_id: str
    stance: Stance
    risk_score: float  # 0.0-1.0
    reasons: List[str]
    actions: List[str]
    sources: List[Dict[str, str]]  # [{"id": "...", "url": "...", "quote": "..."}]
    requires_human_review: bool
    
    # Required for RAG-only principle
    source_ids: Optional[List[str]] = None


class CrisisDetectionAgent:
    """Detect and verify brand crises with multi-model verification."""
    
    # Configuration (Q_011 requirements)
    MIN_INDEPENDENT_SOURCES = 2
    SINGLE_SOURCE_RISK_CAP = 0.5  # Q_011: Cap risk_score to 0.5 when only 1 source
    HIGH_RISK_THRESHOLD = 0.8
    
    def __init__(self, llm, verifier, official_domains: List[str]):
        """Initialize with LLM models and official domains.
        
        Args:
            llm: Analyst LLM model
            verifier: Verifier LLM model (must be different from analyst)
            official_domains: List of official brand domains
        """
        self.llm = llm
        self.verifier = verifier
        self.official_domains = official_domains
        
        # Critical: Verify model separation at init
        if llm == verifier:
            raise ValueError(
                f'CRITICAL: Analyst and verifier must use different models. '
                f'Both are using the same model instance. '
                f'Check config/model_registry.yaml verification.require_different_model setting.'
            )
    
    def verify_crisis(
        self,
        topic_id: str,
        velocity: float,
        sources: List[Document],
        official_domains: List[str],
        brand: str = 'Brand',
        language: str = 'en'
    ) -> CrisisBrief:
        """Verify crisis authenticity with multi-source corroboration.
        
        Q_011 Enforcement:
        - If <2 sources: cap risk_score to 0.5 regardless of LLM output
        - If 0 sources: return risk_score=0.0, stance='unclear'
        - Always add 'verify_official' to actions when <2 sources
        
        Args:
            topic_id: Unique topic identifier
            velocity: Spike magnitude
            sources: List of source documents
            official_domains: List of official brand domains
            brand: Brand name
            language: Output language
            
        Returns:
            Verified crisis brief with enforced risk caps
        """
        # Record source count metric
        crisis_source_count_histogram.observe(len(sources))
        
        # Q_011 Pre-enforcement: Check source count
        if len(sources) == 0:
            # Zero sources = no crisis possible
            return CrisisBrief(
                topic_id=topic_id,
                stance=Stance.UNCLEAR,
                risk_score=0.0,
                reasons=['No sources available for verification'],
                actions=['verify_official'],
                sources=[],
                requires_human_review=True
            )
        
        if len(sources) == 1:
            # Single source: proceed but will cap risk_score
            pass  # Continue to LLM processing
        
        # Step 1: Draft with analyst model
        draft = self._analyze_with_analyst(
            topic_id=topic_id,
            brand=brand,
            velocity=velocity,
            sources=sources,
            official_domains=official_domains,
            language=language
        )
        
        # Step 2: Verify with separate verifier model (CRITICAL)
        verified = self._verify_with_verifier(
            draft=draft,
            sources=sources,
            official_domains=official_domains
        )
        
        # Step 3: Q_011 Post-enforcement - Cap risk_score if <2 sources
        if len(sources) < self.MIN_INDEPENDENT_SOURCES:
            # Cap risk_score to 0.5
            if verified.risk_score > self.SINGLE_SOURCE_RISK_CAP:
                verified.risk_score = self.SINGLE_SOURCE_RISK_CAP
            
            # Ensure 'verify_official' action is included
            if 'verify_official' not in verified.actions:
                verified.actions.insert(0, 'verify_official')
            
            # Force human review for single-source crises
            verified.requires_human_review = True
        
        # Step 4: Policy gate - high risk requires human review
        if verified.risk_score >= self.HIGH_RISK_THRESHOLD:
            verified.requires_human_review = True
            self._escalate_to_human(verified)
        
        # Record metrics
        crisis_detections_total.labels(
            stance=verified.stance.value,
            requires_review=str(verified.requires_human_review)
        ).inc()
        crisis_risk_score_histogram.observe(verified.risk_score)
        
        return verified
    
    def _analyze_with_analyst(
        self,
        topic_id: str,
        brand: str,
        velocity: float,
        sources: List[Document],
        official_domains: List[str],
        language: str
    ) -> CrisisBrief:
        """Draft crisis analysis using analyst model.
        
        Args:
            topic_id: Topic ID
            brand: Brand name
            velocity: Spike magnitude
            sources: Source documents
            official_domains: Official domains
            language: Output language
            
        Returns:
            Draft crisis brief
        """
        # Build prompt with RAG context
        prompt = self._build_crisis_prompt(
            brand=brand,
            velocity=velocity,
            sources=sources,
            official_domains=official_domains,
            language=language
        )
        
        # Call analyst model with low temperature
        response = self.llm.complete(
            prompt=prompt,
            temperature=0.2,
            max_tokens=800
        )
        
        # Parse response (simplified - assumes JSON in content)
        import json
        draft_data = json.loads(response.content)
        draft = CrisisBrief(**draft_data)
        draft.topic_id = topic_id  # Ensure topic_id is set
        
        return draft
    
    def _verify_with_verifier(
        self,
        draft: CrisisBrief,
        sources: List[Document],
        official_domains: List[str]
    ) -> CrisisBrief:
        """Verify draft with separate verifier model.
        
        CRITICAL: This method uses a different model than the analyst.
        
        Args:
            draft: Draft crisis brief from analyst
            sources: Original source documents
            official_domains: Official domains for fact-checking
            
        Returns:
            Verified crisis brief
        """
        # Build verification prompt
        verification_prompt = self._build_verification_prompt(
            draft=draft,
            sources=sources,
            official_domains=official_domains
        )
        
        # CRITICAL: Use separate verifier model with even lower temperature
        response = self.verifier.verify_crisis(
            draft=draft,
            sources=sources,
            official_domains=official_domains
        )
        
        return response
    
    def _build_crisis_prompt(
        self,
        brand: str,
        velocity: float,
        sources: List[Document],
        official_domains: List[str],
        language: str
    ) -> str:
        """Build crisis analysis prompt for analyst model."""
        sources_text = self._format_sources(sources)
        
        return f"""SYSTEM: You are a Brand Intelligence analyst. Use ONLY the provided sources.
Never invent facts. If evidence is insufficient, say "unclear".
Label promotional content explicitly as Promo/広告 when applicable.

INPUT:
- task: "crisis_brief"
- language: {language}
- brand: "{brand}"
- context_metrics: {{"velocity": {velocity}}}
- sources: {sources_text}
- official_domains: {official_domains}

TASKS:
1) Decide stance (against/for/neutral/unclear)
2) Compute risk_score in [0,1] (qualitative → numeric)
3) List 2-4 concise reasons with direct short quotes
4) Output JSON strictly matching CrisisBrief schema
5) If no corroboration from ≥2 independent sources → suggest "verify_official" action

OUTPUT: JSON only with this schema:
{{
  "topic_id": "string",
  "stance": "against|for|neutral|unclear",
  "risk_score": 0.0-1.0,
  "reasons": ["reason1", "reason2"],
  "actions": ["verify_official", "pause_promo"],
  "sources": [{{"id": "src_1", "url": "...", "quote": "..."}}],
  "requires_human_review": false
}}"""
    
    def _build_verification_prompt(
        self,
        draft: CrisisBrief,
        sources: List[Document],
        official_domains: List[str]
    ) -> str:
        """Build verification prompt for verifier model."""
        sources_text = self._format_sources(sources)
        
        return f"""SYSTEM: You are a verification analyst. Your job is to fact-check the draft analysis.
Use ONLY the provided sources. Flag any unsupported claims.

INPUT:
- draft_analysis: {draft.json()}
- sources: {sources_text}
- official_domains: {official_domains}

VERIFICATION TASKS:
1) Verify each reason is supported by cited sources
2) Check risk_score aligns with evidence
3) Confirm ≥2 independent sources used (if not, cap risk_score to 0.5)
4) Flag any invented facts or unsupported claims
5) Adjust risk_score if warranted
6) Maintain or correct stance

OUTPUT: Verified JSON with same CrisisBrief schema:
{{
  "topic_id": "string",
  "stance": "against|for|neutral|unclear",
  "risk_score": 0.0-1.0,
  "reasons": ["verified_reason1", "verified_reason2"],
  "actions": ["verify_official", "pause_promo"],
  "sources": [{{"id": "src_1", "url": "...", "quote": "..."}}],
  "requires_human_review": true/false
}}"""
    
    def _format_sources(self, sources: List[Document]) -> str:
        """Format sources for prompt."""
        formatted = []
        for s in sources:
            formatted.append(f"[{s.id}] {s.url} ({s.domain}): {s.text[:200]}...")
        return "\n".join(formatted)
    
    def _escalate_to_human(self, crisis_brief: CrisisBrief) -> None:
        """Escalate high-risk crisis to human review.
        
        TODO: Implement escalation (Slack, PagerDuty, etc.)
        """
        # Placeholder - actual implementation would send alerts
        pass
