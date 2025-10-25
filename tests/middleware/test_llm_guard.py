"""Tests for LLM Guard Middleware - temperature validation and source diversity"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from middleware.llm_guard import (
    LLMGuard,
    LLMRequest,
    LLMGuardConfig,
    TemperatureViolationError,
    SourceDiversityError
)


class TestTemperatureGuard:
    """T003: LLM temperature guard pre-API middleware"""
    
    def test_temperature_guard_rejects_high(self):
        """Verify temp>0.2 rejected before LLM API"""
        guard = LLMGuard(
            config=LLMGuardConfig(max_temperature=0.2)
        )
        
        # Valid request (temp=0.2)
        valid_request = LLMRequest(
            model="claude-sonnet-4.5",
            prompt="Test prompt",
            temperature=0.2,
            max_tokens=100
        )
        validated = guard.validate_temperature(valid_request)
        assert validated is True
        
        # Invalid request (temp=0.3)
        invalid_request = LLMRequest(
            model="claude-sonnet-4.5",
            prompt="Test prompt",
            temperature=0.3,
            max_tokens=100
        )
        with pytest.raises(TemperatureViolationError) as exc_info:
            guard.validate_temperature(invalid_request)
        
        assert "temperature 0.3 exceeds maximum 0.2" in str(exc_info.value)
    
    def test_temperature_guard_allows_low(self):
        """Verify temp<=0.2 passes validation"""
        guard = LLMGuard(
            config=LLMGuardConfig(max_temperature=0.2)
        )
        
        valid_temps = [0.0, 0.1, 0.15, 0.2]
        for temp in valid_temps:
            request = LLMRequest(
                model="claude-sonnet-4.5",
                prompt="Test prompt",
                temperature=temp,
                max_tokens=100
            )
            assert guard.validate_temperature(request) is True
    
    def test_temperature_guard_default_max(self):
        """Verify default max_temperature=0.2 when not specified"""
        guard = LLMGuard()
        assert guard.config.max_temperature == 0.2
    
    @pytest.mark.asyncio
    async def test_temperature_guard_no_api_call_on_violation(self):
        """Verify LLM API is not called when temperature validation fails"""
        guard = LLMGuard(
            config=LLMGuardConfig(max_temperature=0.2)
        )
        
        mock_llm_api = AsyncMock()
        
        invalid_request = LLMRequest(
            model="claude-sonnet-4.5",
            prompt="Test prompt",
            temperature=0.5,
            max_tokens=100
        )
        
        with pytest.raises(TemperatureViolationError):
            await guard.execute_with_guard(invalid_request, mock_llm_api)
        
        # Verify API was not called
        mock_llm_api.assert_not_called()
    
    def test_metric_llm_guard_rejections_total(self):
        """Verify llm_guard_rejections_total counter increments on temp violation"""
        from prometheus_client import REGISTRY
        
        guard = LLMGuard()
        
        # Get initial counter value
        initial = REGISTRY.get_sample_value(
            'llm_guard_rejections_total',
            {'reason': 'temperature_violation', 'model': 'claude-sonnet-4.5'}
        ) or 0
        
        # Trigger violation
        invalid_request = LLMRequest(
            model="claude-sonnet-4.5",
            prompt="Test",
            temperature=0.3,
            max_tokens=100
        )
        
        with pytest.raises(TemperatureViolationError):
            guard.validate_temperature(invalid_request)
        
        # Verify counter incremented
        final = REGISTRY.get_sample_value(
            'llm_guard_rejections_total',
            {'reason': 'temperature_violation', 'model': 'claude-sonnet-4.5'}
        )
        assert final == initial + 1
    
    def test_temperature_guard_with_none_temp(self):
        """Verify None temperature defaults to config max"""
        guard = LLMGuard(
            config=LLMGuardConfig(max_temperature=0.2, default_temperature=0.1)
        )
        
        request = LLMRequest(
            model="claude-sonnet-4.5",
            prompt="Test",
            temperature=None,
            max_tokens=100
        )
        
        # Should apply default temperature
        validated = guard.validate_temperature(request)
        assert validated is True
        assert request.temperature == 0.1


class TestSourceDiversityGuard:
    """Source domain diversity validation (>=2 domains)"""
    
    def test_source_diversity_rejects_single_domain(self):
        """Verify 3+ URLs from same domain rejected"""
        guard = LLMGuard(
            config=LLMGuardConfig(min_source_domains=2)
        )
        
        # All URLs from same domain
        sources = [
            {"url": "https://example.com/article1"},
            {"url": "https://example.com/article2"},
            {"url": "https://example.com/article3"}
        ]
        
        with pytest.raises(SourceDiversityError) as exc_info:
            guard.validate_source_diversity(sources)
        
        assert "only 1 unique domain" in str(exc_info.value)
        assert "minimum 2 required" in str(exc_info.value)
    
    def test_source_diversity_allows_diverse_domains(self):
        """Verify >=2 domains passes validation"""
        guard = LLMGuard(
            config=LLMGuardConfig(min_source_domains=2)
        )
        
        # Multiple domains
        sources = [
            {"url": "https://example.com/article1"},
            {"url": "https://another.com/article2"},
            {"url": "https://third.com/article3"}
        ]
        
        validated = guard.validate_source_diversity(sources)
        assert validated is True
    
    def test_source_diversity_subdomain_handling(self):
        """Verify subdomains of same root domain count as single domain"""
        guard = LLMGuard(
            config=LLMGuardConfig(min_source_domains=2)
        )
        
        # Multiple subdomains of same root
        sources = [
            {"url": "https://blog.example.com/post1"},
            {"url": "https://news.example.com/post2"},
            {"url": "https://docs.example.com/post3"}
        ]
        
        with pytest.raises(SourceDiversityError):
            guard.validate_source_diversity(sources)
    
    def test_metric_source_diversity_rejections(self):
        """Verify source_diversity rejection metric"""
        from prometheus_client import REGISTRY
        
        guard = LLMGuard()
        
        initial = REGISTRY.get_sample_value(
            'llm_guard_rejections_total',
            {'reason': 'source_diversity', 'model': 'unknown'}
        ) or 0
        
        sources = [
            {"url": "https://example.com/1"},
            {"url": "https://example.com/2"},
            {"url": "https://example.com/3"}
        ]
        
        with pytest.raises(SourceDiversityError):
            guard.validate_source_diversity(sources, model="test-model")
        
        final = REGISTRY.get_sample_value(
            'llm_guard_rejections_total',
            {'reason': 'source_diversity', 'model': 'test-model'}
        )
        assert final == initial + 1


class TestLLMGuardIntegration:
    """Integration tests for LLMGuard middleware"""
    
    @pytest.mark.asyncio
    async def test_full_guard_pipeline(self):
        """Test complete guard validation pipeline"""
        guard = LLMGuard(
            config=LLMGuardConfig(
                max_temperature=0.2,
                min_source_domains=2
            )
        )
        
        mock_llm_api = AsyncMock(return_value={"content": "test response"})
        
        # Valid request
        request = LLMRequest(
            model="claude-sonnet-4.5",
            prompt="Test prompt",
            temperature=0.2,
            max_tokens=100,
            sources=[
                {"url": "https://example.com/1"},
                {"url": "https://another.com/2"}
            ]
        )
        
        result = await guard.execute_with_guard(request, mock_llm_api)
        
        assert result["content"] == "test response"
        mock_llm_api.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_guard_rejects_before_api(self):
        """Verify guard rejects invalid requests before API call"""
        guard = LLMGuard()
        mock_llm_api = AsyncMock()
        
        # Invalid temperature
        invalid_request = LLMRequest(
            model="claude-sonnet-4.5",
            prompt="Test",
            temperature=0.5,
            max_tokens=100
        )
        
        with pytest.raises(TemperatureViolationError):
            await guard.execute_with_guard(invalid_request, mock_llm_api)
        
        # API should not be called
        mock_llm_api.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_guard_with_logging(self):
        """Verify guard logs all validations"""
        guard = LLMGuard()
        
        with patch('middleware.llm_guard.logger') as mock_logger:
            request = LLMRequest(
                model="claude-sonnet-4.5",
                prompt="Test",
                temperature=0.2,
                max_tokens=100
            )
            
            guard.validate_temperature(request)
            
            # Verify logging
            mock_logger.info.assert_called()
            call_args = mock_logger.info.call_args
            assert "temperature_validation" in str(call_args)
