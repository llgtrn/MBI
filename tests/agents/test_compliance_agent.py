# Compliance Agent Test Suite
"""
Tests for Compliance Agent - COPPA & Japan Promo
Priority: P0 (Critical Path)
Capsule Refs: Q_020, Q_021, Q_413, Q_414, A_023
"""

import pytest
from datetime import datetime, timedelta
from agents.compliance_agent import (
    ComplianceAgent, COPPAViolationError,
    PromoLabelError, ComplianceResult
)


class TestCOPPAParentVerification:
    """Q_020: Block orders from users <13 without parent email"""
    
    def test_minor_without_parent_rejected(self):
        """Given user age=12, no parent_email → COPPAViolationError"""
        agent = ComplianceAgent()
        
        with pytest.raises(COPPAViolationError) as exc:
            agent.verify_age_compliance(
                user_age=12,
                parent_email_verified=False
            )
        
        assert "parent" in str(exc.value).lower()
        assert "13" in str(exc.value)
    
    def test_minor_with_parent_accepted(self):
        """Given user age=12, parent_email_verified=True → accepted"""
        agent = ComplianceAgent()
        
        # Should not raise
        result = agent.verify_age_compliance(
            user_age=12,
            parent_email_verified=True
        )
        
        assert result['approved'] is True
    
    def test_adult_no_parent_required(self):
        """Given user age>=13 → no parent verification required"""
        agent = ComplianceAgent()
        
        result = agent.verify_age_compliance(
            user_age=13,
            parent_email_verified=False
        )
        
        assert result['approved'] is True


class TestParentEmailChallenge:
    """Q_413: 24h token parent email verification"""
    
    def test_generate_verification_token(self):
        """Test token generation for parent email"""
        agent = ComplianceAgent()
        
        token = agent.generate_parent_verification_token(
            user_id="user_123",
            parent_email="parent@example.com"
        )
        
        assert token is not None
        assert len(token) > 20  # Secure token
    
    def test_token_expiration(self):
        """Verify token expires after 24 hours"""
        agent = ComplianceAgent()
        
        # Generate token
        token = agent.generate_parent_verification_token(
            user_id="user_123",
            parent_email="parent@example.com"
        )
        
        # Simulate 25 hours passing
        agent.simulate_time_passage(hours=25)
        
        # Try to verify with expired token
        with pytest.raises(COPPAViolationError) as exc:
            agent.verify_parent_token(token)
        
        assert "expired" in str(exc.value).lower()
    
    def test_valid_token_verification(self):
        """Test successful parent verification within 24h"""
        agent = ComplianceAgent()
        
        token = agent.generate_parent_verification_token(
            user_id="user_123",
            parent_email="parent@example.com"
        )
        
        # Verify within 24h
        result = agent.verify_parent_token(token)
        
        assert result['verified'] is True
        assert result['user_id'] == "user_123"


class TestJapanPromoVisual:
    """Q_021: Japan ads verify visual has "広告" label"""
    
    def test_japan_ad_no_visual_label(self):
        """Given Japan ad, no visual "広告" → PromoLabelError"""
        agent = ComplianceAgent()
        
        with pytest.raises(PromoLabelError) as exc:
            agent.verify_promo_compliance(
                country="JP",
                text="Buy now!",
                visual_text=None,
                metadata={"promo": True}
            )
        
        assert "広告" in str(exc.value) or "promo label" in str(exc.value).lower()
    
    def test_japan_ad_with_visual_label(self):
        """Given Japan ad, visual has "広告" → accepted"""
        agent = ComplianceAgent()
        
        result = agent.verify_promo_compliance(
            country="JP",
            text="Buy now!",
            visual_text="(広告) Special offer",
            metadata={"promo": True}
        )
        
        assert result['approved'] is True
    
    def test_non_japan_different_rules(self):
        """Verify non-Japan countries use different rules"""
        agent = ComplianceAgent()
        
        # US requires "Promo" or "Ad"
        result = agent.verify_promo_compliance(
            country="US",
            text="(Promo) Buy now!",
            visual_text=None,
            metadata={"promo": True}
        )
        
        assert result['approved'] is True


class TestVisualMetadataPrecedence:
    """Q_414: Visual label overrides metadata"""
    
    def test_visual_overrides_empty_metadata(self):
        """Given visual="広告", metadata=None → accepted"""
        agent = ComplianceAgent()
        
        result = agent.verify_promo_compliance(
            country="JP",
            text="Special offer",
            visual_text="(広告) Limited time",
            metadata={}  # No metadata
        )
        
        assert result['approved'] is True
    
    def test_visual_overrides_conflicting_metadata(self):
        """Given visual="広告", metadata=non_promo → visual wins"""
        agent = ComplianceAgent()
        
        result = agent.verify_promo_compliance(
            country="JP",
            text="Info",
            visual_text="(広告) Content",
            metadata={"promo": False}  # Conflicting
        )
        
        # Visual takes precedence
        assert result['approved'] is True
    
    def test_metadata_used_when_no_visual(self):
        """Given no visual, metadata used as fallback"""
        agent = ComplianceAgent()
        
        with pytest.raises(PromoLabelError):
            agent.verify_promo_compliance(
                country="JP",
                text="Content",
                visual_text=None,
                metadata={"promo": True}  # Not sufficient without visual
            )


class TestComplianceRuleset:
    """A_023: Age, Promo, GDPR, Medical compliance"""
    
    def test_age_compliance(self):
        """Test COPPA age restrictions"""
        agent = ComplianceAgent()
        
        # Test various ages
        test_cases = [
            (12, False, False),  # Minor without parent → reject
            (12, True, True),    # Minor with parent → accept
            (13, False, True),   # Teen → accept
            (18, False, True)    # Adult → accept
        ]
        
        for age, parent_verified, expected_approved in test_cases:
            result = agent.verify_age_compliance(age, parent_verified)
            assert result['approved'] == expected_approved
    
    def test_promo_compliance(self):
        """Test promo labeling across countries"""
        agent = ComplianceAgent()
        
        # Japan requires "広告"
        assert agent.get_promo_label_requirement("JP") == "広告"
        
        # US/EU require "Promo"/"Ad"
        assert "Promo" in agent.get_promo_label_requirement("US")
    
    def test_gdpr_consent(self):
        """Test GDPR consent requirements"""
        agent = ComplianceAgent()
        
        # EU countries require explicit consent
        result = agent.verify_gdpr_compliance(
            country="DE",
            consent_given=False
        )
        
        assert result['approved'] is False
    
    def test_medical_claims_blocked(self):
        """Test medical claims are rejected"""
        agent = ComplianceAgent()
        
        medical_claims = [
            "cures diabetes",
            "treats cancer",
            "prevents heart disease"
        ]
        
        for claim in medical_claims:
            with pytest.raises(Exception):  # Medical claim error
                agent.verify_claims_compliance([claim])


class TestNegationParsing:
    """Q_434: Medical negation NLI parsing"""
    
    def test_negation_detection(self):
        """Test negation is properly parsed"""
        agent = ComplianceAgent()
        
        # "Does NOT cure" should be allowed
        result = agent.verify_claims_compliance([
            "This product does NOT cure diabetes"
        ])
        
        assert result['approved'] is True
    
    def test_positive_claim_rejected(self):
        """Test positive medical claim is rejected"""
        agent = ComplianceAgent()
        
        with pytest.raises(Exception):
            agent.verify_claims_compliance([
                "This product cures diabetes"
            ])
    
    def test_complex_negation(self):
        """Test complex negation patterns"""
        agent = ComplianceAgent()
        
        test_cases = [
            ("not intended to treat", True),   # Negated → allowed
            ("cannot cure", True),              # Negated → allowed
            ("will treat", False),              # Positive → rejected
            ("proven to cure", False)           # Positive → rejected
        ]
        
        for claim, should_pass in test_cases:
            try:
                result = agent.verify_claims_compliance([claim])
                assert result['approved'] == should_pass
            except Exception:
                assert not should_pass


class TestGDPRCascade:
    """Q_433: GDPR FK exception handling"""
    
    def test_gdpr_deletion_cascade(self):
        """Test GDPR deletion cascades properly"""
        agent = ComplianceAgent()
        
        # Request deletion
        result = agent.process_gdpr_deletion(user_id="user_123")
        
        assert result['deleted_stores'] == [
            'user_profiles',
            'order_history',
            'session_data',
            'audit_logs'
        ]
    
    def test_foreign_key_exception_caught(self):
        """Test FK constraint violations are caught and alerted"""
        agent = ComplianceAgent()
        
        # Simulate FK constraint violation
        agent.simulate_fk_violation(user_id="user_123")
        
        alerts = agent.get_gdpr_alerts()
        
        assert len(alerts) > 0
        assert any('foreign key' in a['message'].lower() for a in alerts)


class TestComplianceMetrics:
    """Overall compliance metrics"""
    
    def test_compliance_rejection_rate(self):
        """Track compliance rejection rate"""
        agent = ComplianceAgent()
        
        # Process 100 requests
        rejections = 0
        for i in range(100):
            try:
                agent.verify_age_compliance(
                    user_age=10 + (i % 10),
                    parent_email_verified=(i % 3 == 0)
                )
            except COPPAViolationError:
                rejections += 1
        
        rejection_rate = rejections / 100
        assert rejection_rate > 0  # Some should be rejected


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
