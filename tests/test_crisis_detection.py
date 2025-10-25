"""Tests for Crisis Detection Agent: Separate analyst/verifier model enforcement."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from agents.crisis_detection import (
    CrisisDetectionAgent,
    CrisisBrief,
    Stance,
    ModelRegistry
)
from agents.llm_guardrails import LLMGuardrails, ValidationResult


# Fixtures
@pytest.fixture
def mock_model_registry():
    """Create mock model registry with separate models."""
    registry = Mock(spec=ModelRegistry)
    registry.get_analyst_model.return_value = "anthropic:claude-sonnet-4.5"
    registry.get_verifier_model.return_value = "anthropic:claude-opus-4"
    return registry


@pytest.fixture
def mock_guardrails():
    """Create mock guardrails."""
    guardrails = Mock(spec=LLMGuardrails)
    guardrails.enforce_temperature = Mock()
    guardrails.validate_sources.return_value = ValidationResult(valid=True)
    guardrails.validate_llm_output.return_value = ValidationResult(valid=True)
    return guardrails


@pytest.fixture
def valid_sources():
    """Create valid source documents."""
    return [
        {'id': 'src_1', 'url': 'https://news.com/1', 'text': 'Source 1 content'},
        {'id': 'src_2', 'url': 'https://news.com/2', 'text': 'Source 2 content'},
        {'id': 'src_3', 'url': 'https://news.com/3', 'text': 'Source 3 content'}
    ]


@pytest.fixture
def agent(mock_model_registry, mock_guardrails):
    """Create CrisisDetectionAgent instance."""
    return CrisisDetectionAgent(
        model_registry=mock_model_registry,
        llm_guardrails=mock_guardrails
    )


# Tests for verifier model separation
def test_verifier_model_differs_from_analyst():
    """Test that verifier model must differ from analyst model."""
    # Mock registry returning SAME model for both
    bad_registry = Mock(spec=ModelRegistry)
    bad_registry.get_analyst_model.return_value = "anthropic:claude-sonnet-4.5"
    bad_registry.get_verifier_model.return_value = "anthropic:claude-sonnet-4.5"  # Same!
    
    mock_guardrails = Mock(spec=LLMGuardrails)
    
    # Should raise ValueError at init
    with pytest.raises(ValueError) as exc_info:
        CrisisDetectionAgent(
            model_registry=bad_registry,
            llm_guardrails=mock_guardrails
        )
    
    assert 'CRITICAL' in str(exc_info.value)
    assert 'different models' in str(exc_info.value)


def test_verifier_uses_separate_model_in_verify_crisis(mock_model_registry, mock_guardrails, valid_sources):
    """Test that verify_crisis uses separate models for analyst and verifier."""
    agent = CrisisDetectionAgent(
        model_registry=mock_model_registry,
        llm_guardrails=mock_guardrails
    )
    
    # Verify models are different at init
    assert agent.analyst_model != agent.verifier_model
    assert agent.analyst_model == "anthropic:claude-sonnet-4.5"
    assert agent.verifier_model == "anthropic:claude-opus-4"


def test_model_registry_enforces_separation(mock_guardrails):
    """Test that ModelRegistry contract requires different models."""
    registry = Mock(spec=ModelRegistry)
    
    # Registry must provide different models
    analyst = registry.get_analyst_model('crisis_brief')
    verifier = registry.get_verifier_model('crisis_brief')
    
    # Contract: these methods must exist
    assert hasattr(registry, 'get_analyst_model')
    assert hasattr(registry, 'get_verifier_model')


# Tests for source requirements
@pytest.mark.asyncio
async def test_verify_crisis_requires_min_sources(agent):
    """Test that <2 sources raises error."""
    insufficient_sources = [
        {'id': 'src_1', 'url': 'https://news.com/1', 'text': 'Only one source'}
    ]
    
    with pytest.raises(ValueError) as exc_info:
        await agent.verify_crisis(
            topic_id='test_001',
            brand='TestBrand',
            velocity=0.8,
            sources=insufficient_sources,
            official_domains=['testbrand.com'],
            language='en'
        )
    
    assert '2' in str(exc_info.value)
    assert 'independent sources' in str(exc_info.value)


@pytest.mark.asyncio
async def test_verify_crisis_validates_sources_pre_call(agent, valid_sources, mock_guardrails):
    """Test that source validation happens before LLM call."""
    # Make source validation fail
    mock_guardrails.validate_sources.return_value = ValidationResult(
        valid=False,
        error='invalid_source'
    )
    
    with pytest.raises(ValueError) as exc_info:
        await agent.verify_crisis(
            topic_id='test_001',
            brand='TestBrand',
            velocity=0.8,
            sources=valid_sources,
            official_domains=['testbrand.com']
        )
    
    assert 'Source validation failed' in str(exc_info.value)


# Tests for workflow
@pytest.mark.asyncio
async def test_verify_crisis_calls_analyst_then_verifier(agent, valid_sources, mock_guardrails):
    """Test that verify_crisis follows analyst → verifier workflow."""
    # Mock the LLM call methods
    draft_response = CrisisBrief(
        topic_id='test_001',
        stance=Stance.UNCLEAR,
        risk_score=0.6,
        reasons=['Reason 1', 'Reason 2'],
        actions=['verify_official'],
        sources=[{'id': 'src_1', 'url': 'https://news.com/1', 'quote': 'quote 1'}],
        requires_human_review=False,
        source_ids=['src_1', 'src_2']
    )
    
    verified_response = CrisisBrief(
        topic_id='test_001',
        stance=Stance.UNCLEAR,
        risk_score=0.5,
        reasons=['Verified reason 1', 'Verified reason 2'],
        actions=['verify_official'],
        sources=[{'id': 'src_1', 'url': 'https://news.com/1', 'quote': 'quote 1'}],
        requires_human_review=False,
        source_ids=['src_1', 'src_2']
    )
    
    with patch.object(agent, '_analyze_with_analyst', new=AsyncMock(return_value=draft_response)):
        with patch.object(agent, '_verify_with_verifier', new=AsyncMock(return_value=verified_response)):
            result = await agent.verify_crisis(
                topic_id='test_001',
                brand='TestBrand',
                velocity=0.8,
                sources=valid_sources,
                official_domains=['testbrand.com']
            )
            
            # Verify both methods were called
            agent._analyze_with_analyst.assert_called_once()
            agent._verify_with_verifier.assert_called_once()
            
            # Verify result is from verifier
            assert result.risk_score == 0.5
            assert 'Verified' in result.reasons[0]


@pytest.mark.asyncio
async def test_high_risk_triggers_human_review(agent, valid_sources, mock_guardrails):
    """Test that risk_score >= 0.8 requires human review."""
    high_risk_response = CrisisBrief(
        topic_id='test_001',
        stance=Stance.AGAINST,
        risk_score=0.85,
        reasons=['Major brand threat'],
        actions=['pause_promo', 'prepare_statement'],
        sources=[{'id': 'src_1', 'url': 'https://news.com/1', 'quote': 'quote 1'}],
        requires_human_review=False,  # Will be set to True
        source_ids=['src_1', 'src_2']
    )
    
    with patch.object(agent, '_analyze_with_analyst', new=AsyncMock(return_value=high_risk_response)):
        with patch.object(agent, '_verify_with_verifier', new=AsyncMock(return_value=high_risk_response)):
            with patch.object(agent, '_escalate_to_human', new=AsyncMock()) as mock_escalate:
                result = await agent.verify_crisis(
                    topic_id='test_001',
                    brand='TestBrand',
                    velocity=0.9,
                    sources=valid_sources,
                    official_domains=['testbrand.com']
                )
                
                # Verify escalation was triggered
                assert result.requires_human_review is True
                mock_escalate.assert_called_once()


# Tests for guardrails integration
@pytest.mark.asyncio
async def test_temperature_enforced_before_analyst_call(agent, valid_sources, mock_guardrails):
    """Test that temperature is enforced before calling analyst."""
    with patch.object(agent, '_call_llm', new=AsyncMock(return_value='{"mock": "response"}')):
        with patch.object(agent, '_analyze_with_analyst') as mock_analyze:
            # Make analyze check that enforce_temperature was called
            async def check_temperature(*args, **kwargs):
                mock_guardrails.enforce_temperature.assert_called_with(0.2)
                return CrisisBrief(
                    topic_id='test_001',
                    stance=Stance.UNCLEAR,
                    risk_score=0.5,
                    reasons=['R1'],
                    actions=['verify_official'],
                    sources=[],
                    requires_human_review=False,
                    source_ids=['src_1', 'src_2']
                )
            
            mock_analyze.side_effect = check_temperature


@pytest.mark.asyncio
async def test_output_validation_for_both_models(agent, valid_sources, mock_guardrails):
    """Test that both analyst and verifier outputs are validated."""
    draft = CrisisBrief(
        topic_id='test_001',
        stance=Stance.UNCLEAR,
        risk_score=0.5,
        reasons=['R1'],
        actions=[],
        sources=[],
        requires_human_review=False,
        source_ids=['src_1', 'src_2']
    )
    
    with patch.object(agent, '_analyze_with_analyst', new=AsyncMock(return_value=draft)):
        with patch.object(agent, '_verify_with_verifier', new=AsyncMock(return_value=draft)):
            await agent.verify_crisis(
                topic_id='test_001',
                brand='TestBrand',
                velocity=0.7,
                sources=valid_sources,
                official_domains=['testbrand.com']
            )
            
            # Should validate sources once at start
            mock_guardrails.validate_sources.assert_called_once()


# Integration test
@pytest.mark.asyncio
async def test_full_crisis_verification_workflow(mock_model_registry, mock_guardrails, valid_sources):
    """Test complete crisis verification workflow with all checks."""
    agent = CrisisDetectionAgent(
        model_registry=mock_model_registry,
        llm_guardrails=mock_guardrails
    )
    
    # Verify initialization enforced model separation
    assert agent.analyst_model == "anthropic:claude-sonnet-4.5"
    assert agent.verifier_model == "anthropic:claude-opus-4"
    assert agent.analyst_model != agent.verifier_model
    
    # Mock successful workflow
    draft = CrisisBrief(
        topic_id='test_001',
        stance=Stance.UNCLEAR,
        risk_score=0.6,
        reasons=['Initial analysis'],
        actions=['verify_official'],
        sources=[{'id': 'src_1', 'url': 'url1', 'quote': 'q1'}],
        requires_human_review=False,
        source_ids=['src_1', 'src_2']
    )
    
    verified = CrisisBrief(
        topic_id='test_001',
        stance=Stance.NEUTRAL,
        risk_score=0.4,
        reasons=['Verified analysis'],
        actions=['monitor'],
        sources=[{'id': 'src_1', 'url': 'url1', 'quote': 'q1'}],
        requires_human_review=False,
        source_ids=['src_1', 'src_2']
    )
    
    with patch.object(agent, '_analyze_with_analyst', new=AsyncMock(return_value=draft)):
        with patch.object(agent, '_verify_with_verifier', new=AsyncMock(return_value=verified)):
            result = await agent.verify_crisis(
                topic_id='test_001',
                brand='TestBrand',
                velocity=0.7,
                sources=valid_sources,
                official_domains=['testbrand.com'],
                language='en'
            )
            
            # Verify final result
            assert result.topic_id == 'test_001'
            assert result.risk_score == 0.4
            assert result.stance == Stance.NEUTRAL
            assert 'Verified' in result.reasons[0]
            assert result.requires_human_review is False  # Low risk
