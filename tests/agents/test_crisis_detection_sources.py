"""
Crisis Detection Agent - RAG-only LLM with multi-source verification
Enforces >=2 independent sources and caps risk_score to 0.5 when only 1 source
"""
import pytest
from unittest.mock import Mock, patch
from agents.crisis_detection import CrisisDetectionAgent, CrisisBrief, Document

@pytest.fixture
def agent():
    """Crisis detection agent fixture"""
    return CrisisDetectionAgent(
        llm=Mock(),
        verifier=Mock(),
        official_domains=['company.com', 'officialblog.com']
    )

@pytest.fixture
def mock_sources_two():
    """Mock two independent sources"""
    return [
        Document(id='src_1', url='https://news1.com/article', text='Brand X faces criticism...', domain='news1.com'),
        Document(id='src_2', url='https://news2.com/report', text='Independent report confirms...', domain='news2.com')
    ]

@pytest.fixture
def mock_sources_one():
    """Mock single source only"""
    return [
        Document(id='src_1', url='https://socialpost.com/123', text='Rumor about Brand X...', domain='socialpost.com')
    ]

class TestCrisisSourceVerification:
    """Test >=2 sources requirement and risk_score capping"""
    
    def test_two_sources_allows_high_risk_score(self, agent, mock_sources_two):
        """With >=2 independent sources, risk_score can be >0.5"""
        # Arrange
        agent.llm.complete = Mock(return_value=Mock(
            content='{"topic_id":"c1","stance":"against","risk_score":0.85,"reasons":["r1","r2"],"actions":["verify_official"],"sources":[{"id":"src_1","url":"...","quote":"..."},{"id":"src_2","url":"...","quote":"..."}],"requires_human_review":true}'
        ))
        agent.verifier.verify_crisis = Mock(return_value=CrisisBrief(
            topic_id='c1', stance='against', risk_score=0.85, 
            reasons=['r1','r2'], actions=['verify_official'],
            sources=[{'id':'src_1','url':'...','quote':'...'},{'id':'src_2','url':'...','quote':'...'}],
            requires_human_review=True
        ))
        
        # Act
        result = agent.verify_crisis(
            topic_id='c1',
            velocity=5.0,
            sources=mock_sources_two,
            official_domains=['company.com']
        )
        
        # Assert
        assert result.risk_score == 0.85
        assert len(result.sources) >= 2
        assert result.requires_human_review is True
    
    def test_one_source_caps_risk_score_to_0_5(self, agent, mock_sources_one):
        """With only 1 source, risk_score must be capped to 0.5 regardless of LLM output"""
        # Arrange - LLM tries to output 0.9 but should be capped
        agent.llm.complete = Mock(return_value=Mock(
            content='{"topic_id":"c2","stance":"unclear","risk_score":0.9,"reasons":["r1"],"actions":["verify_official"],"sources":[{"id":"src_1","url":"...","quote":"..."}],"requires_human_review":true}'
        ))
        agent.verifier.verify_crisis = Mock(return_value=CrisisBrief(
            topic_id='c2', stance='unclear', risk_score=0.9,
            reasons=['r1'], actions=['verify_official'],
            sources=[{'id':'src_1','url':'...','quote':'...'}],
            requires_human_review=True
        ))
        
        # Act
        result = agent.verify_crisis(
            topic_id='c2',
            velocity=3.0,
            sources=mock_sources_one,
            official_domains=['company.com']
        )
        
        # Assert - risk_score capped to 0.5
        assert result.risk_score <= 0.5
        assert len(result.sources) == 1
        assert 'verify_official' in result.actions
        assert result.requires_human_review is True
    
    def test_no_sources_returns_risk_0_0(self, agent):
        """With 0 sources, risk_score must be 0.0"""
        # Act
        result = agent.verify_crisis(
            topic_id='c3',
            velocity=1.0,
            sources=[],
            official_domains=['company.com']
        )
        
        # Assert
        assert result.risk_score == 0.0
        assert result.stance == 'unclear'
        assert 'verify_official' in result.actions
    
    def test_verify_official_action_when_one_source(self, agent, mock_sources_one):
        """When <2 sources, actions must include 'verify_official'"""
        # Arrange
        agent.llm.complete = Mock(return_value=Mock(
            content='{"topic_id":"c4","stance":"neutral","risk_score":0.3,"reasons":["r1"],"actions":["pause_promo"],"sources":[{"id":"src_1","url":"...","quote":"..."}],"requires_human_review":false}'
        ))
        agent.verifier.verify_crisis = Mock(return_value=CrisisBrief(
            topic_id='c4', stance='neutral', risk_score=0.3,
            reasons=['r1'], actions=['pause_promo'],
            sources=[{'id':'src_1','url':'...','quote':'...'}],
            requires_human_review=False
        ))
        
        # Act
        result = agent.verify_crisis(
            topic_id='c4',
            velocity=2.0,
            sources=mock_sources_one,
            official_domains=['company.com']
        )
        
        # Assert
        assert 'verify_official' in result.actions
        assert result.risk_score <= 0.5

class TestCrisisMetricsEmission:
    """Test that crisis detection emits proper metrics"""
    
    @patch('agents.crisis_detection.prometheus_client')
    def test_risk_score_histogram_emitted(self, mock_prom, agent, mock_sources_two):
        """Risk score should be recorded in histogram"""
        # Arrange
        agent.llm.complete = Mock(return_value=Mock(
            content='{"topic_id":"c5","stance":"for","risk_score":0.6,"reasons":["r1","r2"],"actions":[],"sources":[{"id":"src_1","url":"...","quote":"..."},{"id":"src_2","url":"...","quote":"..."}],"requires_human_review":false}'
        ))
        agent.verifier.verify_crisis = Mock(return_value=CrisisBrief(
            topic_id='c5', stance='for', risk_score=0.6,
            reasons=['r1','r2'], actions=[],
            sources=[{'id':'src_1','url':'...','quote':'...'},{'id':'src_2','url':'...','quote':'...'}],
            requires_human_review=False
        ))
        
        # Act
        result = agent.verify_crisis(
            topic_id='c5',
            velocity=4.0,
            sources=mock_sources_two,
            official_domains=['company.com']
        )
        
        # Assert
        mock_prom.Histogram.assert_called()
        assert result.risk_score == 0.6
