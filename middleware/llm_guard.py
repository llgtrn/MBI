"""LLM Guard Middleware - Temperature validation and source diversity enforcement"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from urllib.parse import urlparse
from collections import Counter
import structlog
from prometheus_client import Counter as PrometheusCounter

logger = structlog.get_logger()


# Metrics
llm_guard_rejections_total = PrometheusCounter(
    'llm_guard_rejections_total',
    'LLM guard rejections',
    ['reason', 'model']
)
llm_guard_validations_total = PrometheusCounter(
    'llm_guard_validations_total',
    'LLM guard validations passed',
    ['model']
)


class TemperatureViolationError(ValueError):
    """Raised when temperature exceeds maximum allowed value"""
    pass


class SourceDiversityError(ValueError):
    """Raised when source domain diversity is insufficient"""
    pass


@dataclass
class LLMGuardConfig:
    """Configuration for LLM guard middleware"""
    max_temperature: float = 0.2
    default_temperature: float = 0.1
    min_source_domains: int = 2
    enforce_temperature: bool = True
    enforce_source_diversity: bool = True
    log_all_requests: bool = True


@dataclass
class LLMRequest:
    """LLM request schema with validation fields"""
    model: str
    prompt: str
    temperature: Optional[float] = None
    max_tokens: int = 1000
    sources: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Apply defaults after initialization"""
        if self.temperature is None:
            self.temperature = 0.1  # Default to conservative temperature


class LLMGuard:
    """
    LLM Guard Middleware - Pre-API validation layer
    
    Enforces:
    - Temperature ≤ 0.2 (determinism for RAG)
    - Source domain diversity ≥ 2 (prevent single-source bias)
    
    All validations happen BEFORE calling LLM API to:
    - Save API costs on invalid requests
    - Fail fast with clear error messages
    - Enforce MBI best practices at the middleware layer
    """
    
    def __init__(self, config: Optional[LLMGuardConfig] = None):
        self.config = config or LLMGuardConfig()
        logger.info(
            "llm_guard_initialized",
            max_temperature=self.config.max_temperature,
            min_source_domains=self.config.min_source_domains
        )
    
    def validate_temperature(self, request: LLMRequest) -> bool:
        """
        Validate temperature is within allowed range
        
        Args:
            request: LLM request to validate
            
        Returns:
            True if valid
            
        Raises:
            TemperatureViolationError: If temperature exceeds maximum
        """
        # Apply default if not set
        if request.temperature is None:
            request.temperature = self.config.default_temperature
        
        # Check against maximum
        if request.temperature > self.config.max_temperature:
            logger.warning(
                "temperature_violation",
                temperature=request.temperature,
                max_allowed=self.config.max_temperature,
                model=request.model
            )
            
            llm_guard_rejections_total.labels(
                reason='temperature_violation',
                model=request.model
            ).inc()
            
            raise TemperatureViolationError(
                f"Temperature {request.temperature} exceeds maximum {self.config.max_temperature}. "
                f"MBI requires temperature ≤ {self.config.max_temperature} for deterministic RAG outputs."
            )
        
        logger.info(
            "temperature_validation_passed",
            temperature=request.temperature,
            model=request.model
        )
        
        return True
    
    def validate_source_diversity(
        self,
        sources: List[Dict[str, Any]],
        model: str = "unknown"
    ) -> bool:
        """
        Validate source domain diversity to prevent single-source bias
        
        Args:
            sources: List of source documents with 'url' field
            model: Model name for metrics
            
        Returns:
            True if valid
            
        Raises:
            SourceDiversityError: If domain diversity insufficient
        """
        if not sources or len(sources) < self.config.min_source_domains:
            # Skip validation if insufficient sources
            return True
        
        # Extract root domains from URLs
        domains = []
        for source in sources:
            url = source.get('url', '')
            if url:
                parsed = urlparse(url)
                # Get root domain (e.g., example.com from blog.example.com)
                hostname_parts = parsed.hostname.split('.') if parsed.hostname else []
                if len(hostname_parts) >= 2:
                    root_domain = '.'.join(hostname_parts[-2:])
                    domains.append(root_domain)
        
        # Count unique domains
        unique_domains = len(set(domains))
        
        if unique_domains < self.config.min_source_domains:
            logger.warning(
                "source_diversity_violation",
                unique_domains=unique_domains,
                min_required=self.config.min_source_domains,
                domains=list(set(domains)),
                model=model
            )
            
            llm_guard_rejections_total.labels(
                reason='source_diversity',
                model=model
            ).inc()
            
            raise SourceDiversityError(
                f"Insufficient source diversity: only {unique_domains} unique domain(s) found, "
                f"minimum {self.config.min_source_domains} required. "
                f"MBI requires diverse sources to prevent single-source bias."
            )
        
        logger.info(
            "source_diversity_validation_passed",
            unique_domains=unique_domains,
            domains=list(set(domains)),
            model=model
        )
        
        return True
    
    async def execute_with_guard(
        self,
        request: LLMRequest,
        llm_api_fn: Callable
    ) -> Any:
        """
        Execute LLM request with full guard validation
        
        Validation order:
        1. Temperature check
        2. Source diversity check
        3. Call LLM API (only if all validations pass)
        
        Args:
            request: LLM request to execute
            llm_api_fn: Async function to call LLM API
            
        Returns:
            LLM API response
            
        Raises:
            TemperatureViolationError: If temperature invalid
            SourceDiversityError: If source diversity insufficient
        """
        # Validate temperature
        if self.config.enforce_temperature:
            self.validate_temperature(request)
        
        # Validate source diversity
        if self.config.enforce_source_diversity and request.sources:
            self.validate_source_diversity(request.sources, request.model)
        
        # Log request (sanitized)
        if self.config.log_all_requests:
            logger.info(
                "llm_request_validated",
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                source_count=len(request.sources),
                prompt_length=len(request.prompt)
            )
        
        # All validations passed - call API
        llm_guard_validations_total.labels(model=request.model).inc()
        
        return await llm_api_fn(request)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get guard statistics for monitoring"""
        return {
            "config": {
                "max_temperature": self.config.max_temperature,
                "min_source_domains": self.config.min_source_domains,
                "enforce_temperature": self.config.enforce_temperature,
                "enforce_source_diversity": self.config.enforce_source_diversity
            },
            "status": "active"
        }


# Singleton instance for application-wide use
_default_guard: Optional[LLMGuard] = None


def get_default_guard() -> LLMGuard:
    """Get or create default LLM guard instance"""
    global _default_guard
    if _default_guard is None:
        _default_guard = LLMGuard()
    return _default_guard


async def guarded_llm_call(
    model: str,
    prompt: str,
    temperature: Optional[float] = None,
    max_tokens: int = 1000,
    sources: Optional[List[Dict[str, Any]]] = None,
    llm_api_fn: Optional[Callable] = None
) -> Any:
    """
    Convenience function for guarded LLM calls
    
    Example:
        response = await guarded_llm_call(
            model="claude-sonnet-4.5",
            prompt="Analyze this data...",
            temperature=0.2,
            sources=[
                {"url": "https://source1.com/article"},
                {"url": "https://source2.com/article"}
            ],
            llm_api_fn=my_llm_api_function
        )
    """
    guard = get_default_guard()
    
    request = LLMRequest(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        sources=sources or []
    )
    
    if llm_api_fn is None:
        raise ValueError("llm_api_fn is required")
    
    return await guard.execute_with_guard(request, llm_api_fn)
