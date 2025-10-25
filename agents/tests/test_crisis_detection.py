"""
Test suite for Crisis Detection Agent
Validates: Separate verifier model enforcement
"""
import pytest
from unittest.mock import Mock, patch
from agents.crisis_detection import CrisisDetectionAgent, CrisisBrief
from config.model_registry import ModelRegistry


class TestVerifierModelSeparation:
    """Test that verifier model differs from analyst model"""
    
    def test_model_registry_has_separate_verifier(self):
        """ModelRegistry must define separate verifier model"""
        registry = ModelRegistry()
        
        analyst_model = registry.get_analyst_model()
        verifier_model = registry.get_verifier_model()
        
        # Models must be different
        assert analyst_model != verifier_model, \
            "Verifier model must be different from analyst model"
        
        # Both must be valid model identifiers
        assert analyst_model in registry.available_models()
        assert verifier_model in registry.available_models()
    
    def test_crisis_detection_uses_separate_models(self):
        """CrisisDetectionAgent must use separate models for analysis and verification"""
        agent = CrisisDetectionAgent()
        
        # Get models configured for agent
        analyst = agent.analyst_llm_model
        verifier = agent.verifier_llm_model
        
        assert analyst != verifier, \
            "Crisis detection must use separate analyst and verifier models"
    
    def test_verify_crisis_calls_different_model(self):
        """verify_crisis() must call verifier model, not analyst"""
        agent = CrisisDetectionAgent()
        
        # Mock the LLM clients
        with patch.object(agent, 'analyst_llm') as mock_analyst, \
             patch.object(agent, 'verifier_llm') as mock_verifier:
            
            # Set up mock responses
            mock_analyst.complete.return_value = Mock(
                content='{"stance": "against", "risk_score": 0.8}'
            )
            mock_verifier.complete.return_value = Mock(
                content='{"stance": "against", "risk_score": 0.8, "verified": true}'
            )
            
            # Call verify_crisis
            result = agent.verify_crisis(
                topic_id="test_topic",
                velocity=2.5,
                sources=[
                    {"id": "s1", "url": "https://test1.com", "text": "Test source 1"},
                    {"id": "s2", "url": "https://test2.com", "text": "Test source 2"}
                ]
            )
            
            # Analyst should be called once for initial analysis
            assert mock_analyst.complete.call_count == 1
            
            # Verifier should be called once for verification
            assert mock_verifier.complete.call_count == 1
            
            # Verify they were called with different model configs
            analyst_call = mock_analyst.complete.call_args
            verifier_call = mock_verifier.complete.call_args
            
            assert analyst_call != verifier_call, \
                "Analyst and verifier must be called with different configurations"
    
    def test_same_model_rejected(self):
        """Attempting to use same model for both should raise error"""
        with pytest.raises(ValueError, match=".*same model.*"):
            agent = CrisisDetectionAgent(
                analyst_model="claude-sonnet-4",
                verifier_model="claude-sonnet-4"  # Same as analyst - should fail
            )


class TestCrisisVerificationFlow:
    """Test complete crisis verification flow"""
    
    @pytest.mark.asyncio
    async def test_two_stage_verification(self):
        """Crisis verification must use two-stage analyst→verifier flow"""
        agent = CrisisDetectionAgent()
        
        sources = [
            {"id": "s1", "url": "https://news1.com", "text": "Company faces criticism"},
            {"id": "s2", "url": "https://news2.com", "text": "Negative social media spike"}
        ]
        
        with patch.object(agent, 'analyst_llm') as mock_analyst, \
             patch.object(agent, 'verifier_llm') as mock_verifier:
            
            # Stage 1: Analyst draft
            mock_analyst.complete.return_value = Mock(
                content='''{
                    "topic_id": "test",
                    "stance": "against",
                    "risk_score": 0.85,
                    "reasons": ["reason1", "reason2"],
                    "sources": [{"id": "s1"}, {"id": "s2"}]
                }'''
            )
            
            # Stage 2: Verifier validation
            mock_verifier.complete.return_value = Mock(
                content='''{
                    "verified": true,
                    "stance": "against",
                    "risk_score": 0.82,
                    "reasons": ["verified reason1", "verified reason2"],
                    "sources": [{"id": "s1"}, {"id": "s2"}],
                    "verification_notes": "Sources corroborated"
                }'''
            )
            
            result = await agent.verify_crisis(
                topic_id="test",
                velocity=2.0,
                sources=sources
            )
            
            # Verify flow
            assert mock_analyst.complete.call_count == 1, "Analyst should be called once"
            assert mock_verifier.complete.call_count == 1, "Verifier should be called once"
            
            # Result should come from verifier
            assert result.risk_score == 0.82  # Verifier's score
            assert "verified" in str(result.reasons).lower() or result.requires_human_review
    
    def test_verifier_can_downgrade_risk(self):
        """Verifier can downgrade risk score if sources insufficient"""
        agent = CrisisDetectionAgent()
        
        with patch.object(agent, 'analyst_llm') as mock_analyst, \
             patch.object(agent, 'verifier_llm') as mock_verifier:
            
            # Analyst says high risk
            mock_analyst.complete.return_value = Mock(
                content='{"stance": "against", "risk_score": 0.9, "sources": [{"id": "s1"}]}'
            )
            
            # Verifier downgrades due to insufficient corroboration
            mock_verifier.complete.return_value = Mock(
                content='''{
                    "verified": false,
                    "stance": "unclear",
                    "risk_score": 0.4,
                    "reasons": ["Only one source, cannot verify"],
                    "actions": ["verify_official"],
                    "requires_human_review": true
                }'''
            )
            
            result = agent.verify_crisis(
                topic_id="test",
                velocity=1.5,
                sources=[{"id": "s1", "text": "Single source"}]
            )
            
            # Final risk should be verifier's downgraded score
            assert result.risk_score < 0.5
            assert result.requires_human_review is True


class TestModelRegistryIntegration:
    """Test integration with ModelRegistry configuration"""
    
    def test_model_registry_loaded_from_yaml(self):
        """ModelRegistry should load configuration from YAML"""
        registry = ModelRegistry.from_yaml("config/model_registry.yaml")
        
        # Check managed_verifier exists and differs from managed_main
        assert "managed_verifier" in registry.routing_policy
        assert "managed_main" in registry.routing_policy
        
        verifier_model = registry.routing_policy["managed_verifier"]["models"][0]
        main_model = registry.routing_policy["managed_main"]["models"][0]
        
        assert verifier_model != main_model
    
    def test_crisis_brief_uses_verifier_routing(self):
        """Crisis brief task should route to verifier model"""
        registry = ModelRegistry()
        
        # Get routing for crisis_brief task
        routing = registry.get_routing_for_task("crisis_brief")
        
        assert "verify" in routing, "Crisis brief must have verify routing"
        assert routing["verify"] == "managed_verifier"


class TestRiskGates:
    """Test risk gates and safety checks"""
    
    def test_high_risk_requires_human_review(self):
        """Risk score >=0.8 must trigger human review"""
        agent = CrisisDetectionAgent()
        
        with patch.object(agent, 'analyst_llm') as mock_analyst, \
             patch.object(agent, 'verifier_llm') as mock_verifier:
            
            # High risk scenario
            mock_analyst.complete.return_value = Mock(
                content='{"risk_score": 0.95, "stance": "against"}'
            )
            mock_verifier.complete.return_value = Mock(
                content='{"risk_score": 0.92, "stance": "against", "verified": true}'
            )
            
            result = agent.verify_crisis(
                topic_id="high_risk",
                velocity=3.0,
                sources=[{"id": "s1"}, {"id": "s2"}]
            )
            
            assert result.risk_score >= 0.8
            assert result.requires_human_review is True
    
    def test_official_domain_check(self):
        """Should check for official domain sources"""
        agent = CrisisDetectionAgent()
        
        official_sources = [
            {"id": "s1", "url": "https://company.com/press-release", "text": "Official statement"}
        ]
        
        result = agent._check_official_sources(
            sources=official_sources,
            official_domains=["company.com"]
        )
        
        assert result.has_official_source is True
        
        # Test without official sources
        unofficial_sources = [
            {"id": "s1", "url": "https://random-blog.com", "text": "Rumor"}
        ]
        
        result = agent._check_official_sources(
            sources=unofficial_sources,
            official_domains=["company.com"]
        )
        
        assert result.has_official_source is False
