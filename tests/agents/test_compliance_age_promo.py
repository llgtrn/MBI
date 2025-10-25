"""
Compliance Agent - Age verification and promotional labeling enforcement
Enforces regulatory requirements: age <18 block 403, Promo/広告 labeling
"""
import pytest
from unittest.mock import Mock
from agents.compliance_agent import ComplianceAgent, ComplianceResult

@pytest.fixture
def agent():
    """Compliance agent fixture"""
    return ComplianceAgent(
        policy_pack={'japan': {'promo_label_required': True, 'min_age': 18}}
    )

class TestAgeVerification:
    """Test age <18 block 403 enforcement (Q_012)"""
    
    def test_age_17_returns_403(self, agent):
        """Age 17 must return 403 status"""
        # Act
        result = agent.check_compliance(
            text='Buy our product now!',
            claims=[],
            language='ja',
            country='japan',
            user_age=17
        )
        
        # Assert
        assert result.approved is False
        assert result.status_code == 403
        assert 'age_requirement' in result.violations
        assert any('18' in v for v in result.violations)
    
    def test_age_18_allowed(self, agent):
        """Age 18 and above should be allowed"""
        # Act
        result = agent.check_compliance(
            text='(広告) Buy our product now!',
            claims=[],
            language='ja',
            country='japan',
            user_age=18
        )
        
        # Assert
        assert result.approved is True
        assert result.status_code == 200
    
    def test_missing_age_defaults_to_403(self, agent):
        """Missing age should default to blocking for safety"""
        # Act
        result = agent.check_compliance(
            text='(広告) Buy our product now!',
            claims=[],
            language='ja',
            country='japan',
            user_age=None
        )
        
        # Assert
        assert result.approved is False
        assert result.status_code == 403
        assert 'age_verification_required' in result.violations

class TestPromoLabeling:
    """Test Promo/広告 label enforcement (Q_013)"""
    
    def test_no_promo_label_rejected(self, agent):
        """Content without Promo/広告 label must be rejected"""
        # Act
        result = agent.check_compliance(
            text='Buy our amazing product now! Limited offer!',
            claims=[],
            language='ja',
            country='japan',
            user_age=25
        )
        
        # Assert
        assert result.approved is False
        assert 'promo_label_missing' in result.violations
        assert any('Promo' in v or '広告' in v for v in result.violations)
    
    def test_promo_label_katakana_accepted(self, agent):
        """(Promo/広告) label should be accepted"""
        # Act
        result = agent.check_compliance(
            text='(Promo/広告) Buy our amazing product now!',
            claims=[],
            language='ja',
            country='japan',
            user_age=25
        )
        
        # Assert
        assert result.approved is True
    
    def test_promo_only_accepted(self, agent):
        """(Promo) alone should be accepted"""
        # Act
        result = agent.check_compliance(
            text='(Promo) Buy our product!',
            claims=[],
            language='en',
            country='us',
            user_age=25
        )
        
        # Assert
        assert result.approved is True
    
    def test_kouokoku_only_accepted(self, agent):
        """(広告) alone should be accepted"""
        # Act
        result = agent.check_compliance(
            text='(広告) 今すぐ購入！',
            claims=[],
            language='ja',
            country='japan',
            user_age=25
        )
        
        # Assert
        assert result.approved is True
    
    def test_promo_label_not_at_start_rejected(self, agent):
        """Promo label must be at the beginning of text"""
        # Act
        result = agent.check_compliance(
            text='Buy our product now! (Promo/広告)',
            claims=[],
            language='ja',
            country='japan',
            user_age=25
        )
        
        # Assert
        assert result.approved is False
        assert 'promo_label_position' in result.violations

class TestMetricsEmission:
    """Test compliance metrics are emitted"""
    
    def test_compliance_check_counter_incremented(self, agent, monkeypatch):
        """Each compliance check should increment counter"""
        # Arrange
        mock_counter = Mock()
        monkeypatch.setattr('agents.compliance_agent.compliance_checks_total', mock_counter)
        
        # Act
        agent.check_compliance(
            text='(広告) Test',
            claims=[],
            language='ja',
            country='japan',
            user_age=25
        )
        
        # Assert
        mock_counter.labels.assert_called()
        mock_counter.labels().inc.assert_called_once()
    
    def test_violation_counter_on_failure(self, agent, monkeypatch):
        """Violations should be recorded in metrics"""
        # Arrange
        mock_counter = Mock()
        monkeypatch.setattr('agents.compliance_agent.compliance_violations_total', mock_counter)
        
        # Act
        agent.check_compliance(
            text='Buy now!',  # No promo label
            claims=[],
            language='ja',
            country='japan',
            user_age=25
        )
        
        # Assert
        mock_counter.labels.assert_called()
        assert mock_counter.labels().inc.call_count >= 1
