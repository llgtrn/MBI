# LLM Council Test Suite
"""
Tests for LLM Council Safety Guards
Priority: P0 (Critical Path)
Capsule Refs: Q_015, Q_016, Q_409, Q_410, A_021
"""

import pytest
from datetime import datetime, timedelta
from agents.llm_council import (
    LLMCouncil, TemperatureViolationError, 
    DomainDiversityError, RAGOutput
)


class TestTemperatureGuard:
    """Q_015: Block LLM calls with temperature>0.2"""
    
    def test_reject_high_temperature(self):
        """Given temperature=0.3 → TemperatureViolationError"""
        council = LLMCouncil()
        
        with pytest.raises(TemperatureViolationError) as exc:
            council.complete(
                prompt="Test prompt",
                temperature=0.3
            )
        
        assert "temperature" in str(exc.value).lower()
        assert "0.2" in str(exc.value)
    
    def test_accept_valid_temperature(self):
        """Given temperature<=0.2 → accepted"""
        council = LLMCouncil()
        
        # Should not raise
        result = council.complete(
            prompt="Test prompt",
            temperature=0.2
        )
        
        assert result is not None
    
    def test_default_temperature(self):
        """Verify default temperature is <=0.2"""
        council = LLMCouncil()
        
        assert council.default_temperature <= 0.2


class TestDomainDiversity:
    """Q_016: Reject outputs citing <2 unique domains"""
    
    def test_single_domain_rejected(self):
        """Given sources from same domain → DomainDiversityError"""
        council = LLMCouncil()
        
        sources = [
            {"id": "src_1", "url": "https://example.com/page1"},
            {"id": "src_2", "url": "https://example.com/page2"},
            {"id": "src_3", "url": "https://example.com/page3"}
        ]
        
        with pytest.raises(DomainDiversityError) as exc:
            council.validate_domain_diversity(sources)
        
        assert "unique domains" in str(exc.value).lower()
    
    def test_multiple_domains_accepted(self):
        """Given sources from ≥2 domains → accepted"""
        council = LLMCouncil()
        
        sources = [
            {"id": "src_1", "url": "https://domain1.com/page"},
            {"id": "src_2", "url": "https://domain2.com/page"}
        ]
        
        # Should not raise
        council.validate_domain_diversity(sources)
    
    def test_subdomain_deduplication(self):
        """Test that subdomains are deduplicated correctly"""
        council = LLMCouncil()
        
        sources = [
            {"id": "src_1", "url": "https://blog.example.com/post1"},
            {"id": "src_2", "url": "https://www.example.com/page"},
            {"id": "src_3", "url": "https://example.com/other"}
        ]
        
        # All same base domain (example.com)
        with pytest.raises(DomainDiversityError):
            council.validate_domain_diversity(sources)


class TestDomainExtraction:
    """Q_409: Extract FQDN, dedupe source domains"""
    
    def test_extract_fqdn(self):
        """Test FQDN extraction from various URLs"""
        council = LLMCouncil()
        
        test_cases = [
            ("https://example.com/path", "example.com"),
            ("https://www.example.com/path", "example.com"),
            ("https://subdomain.example.com/path", "example.com"),
            ("https://example.co.uk/path", "example.co.uk"),
            ("http://localhost:8080/path", "localhost")
        ]
        
        for url, expected_domain in test_cases:
            domain = council.extract_base_domain(url)
            assert domain == expected_domain, f"URL: {url}"
    
    def test_dedupe_domains(self):
        """Test domain deduplication"""
        council = LLMCouncil()
        
        sources = [
            {"url": "https://a.example.com/1"},
            {"url": "https://b.example.com/2"},
            {"url": "https://other.com/3"},
            {"url": "https://www.other.com/4"}
        ]
        
        domains = council.get_unique_domains(sources)
        
        assert len(domains) == 2
        assert "example.com" in domains
        assert "other.com" in domains


class TestKeyRotation:
    """Q_410: Key rotation 30d/45d"""
    
    def test_analyst_key_rotation_schedule(self):
        """Verify analyst key rotation every 30 days"""
        from config.llm_config import get_key_rotation_config
        
        config = get_key_rotation_config()
        
        assert config['analyst_rotation_days'] == 30
        assert config['verifier_rotation_days'] == 45
    
    @pytest.mark.integration
    def test_key_rotation_alert(self):
        """Test alert when key age exceeds threshold"""
        council = LLMCouncil()
        
        # Simulate old keys
        council.set_key_age('analyst', days=31)
        council.set_key_age('verifier', days=46)
        
        alerts = council.check_key_rotation_status()
        
        assert len(alerts) == 2
        assert any('analyst' in a['message'].lower() for a in alerts)
        assert any('verifier' in a['message'].lower() for a in alerts)


class TestRAGSafety:
    """A_021: All RAG outputs include source_ids with ≥2 domains"""
    
    def test_rag_output_schema(self):
        """Verify RAGOutput includes required fields"""
        output = RAGOutput(
            text="Crisis detected",
            source_ids=["src_1", "src_2"],
            source_domains=["domain1.com", "domain2.com"],
            temperature=0.2
        )
        
        assert hasattr(output, 'source_ids')
        assert hasattr(output, 'source_domains')
        assert len(output.source_ids) >= 2
        assert len(output.source_domains) >= 2
    
    def test_rag_output_validation(self):
        """Test RAG output fails validation if <2 domains"""
        with pytest.raises(DomainDiversityError):
            RAGOutput(
                text="Output",
                source_ids=["src_1", "src_2"],
                source_domains=["domain1.com"],  # Only 1 domain
                temperature=0.2
            )
    
    def test_source_id_domain_alignment(self):
        """Verify source_ids align with source_domains"""
        council = LLMCouncil()
        
        sources = [
            {"id": "src_1", "url": "https://domain1.com/page1"},
            {"id": "src_2", "url": "https://domain2.com/page2"}
        ]
        
        output = council.create_rag_output(
            text="Test output",
            sources=sources,
            temperature=0.2
        )
        
        assert len(output.source_ids) == 2
        assert "domain1.com" in output.source_domains
        assert "domain2.com" in output.source_domains


class TestVerifierSeparation:
    """Q_017: Verifier uses different model/key"""
    
    def test_different_models(self):
        """Verify analyst and verifier use different models"""
        council = LLMCouncil()
        
        analyst_model = council.get_analyst_model()
        verifier_model = council.get_verifier_model()
        
        assert analyst_model != verifier_model
    
    def test_different_api_keys(self):
        """Verify analyst and verifier use different API keys"""
        council = LLMCouncil()
        
        analyst_key = council.get_analyst_api_key()
        verifier_key = council.get_verifier_api_key()
        
        assert analyst_key != verifier_key


class TestLLMAuditTrail:
    """Q_427: LLM retrieval audit with query+source_ids"""
    
    def test_audit_log_creation(self):
        """Verify audit log includes all required fields"""
        council = LLMCouncil()
        
        result = council.complete_with_audit(
            prompt="Test query",
            sources=[
                {"id": "src_1", "url": "https://domain1.com/page"},
                {"id": "src_2", "url": "https://domain2.com/page"}
            ],
            temperature=0.2
        )
        
        audit = result['audit']
        assert 'query' in audit
        assert 'source_ids' in audit
        assert 'timestamp' in audit
        assert 'model' in audit
        assert len(audit['source_ids']) >= 2


class TestVerifierRejection:
    """Q_428: Verifier rejection schema with reason"""
    
    def test_rejection_includes_reason(self):
        """Verify verifier rejection includes detailed reason"""
        council = LLMCouncil()
        
        analyst_output = {
            "text": "Output with insufficient sources",
            "source_ids": ["src_1"]  # Only 1 source
        }
        
        verification = council.verify(analyst_output)
        
        assert verification['approved'] is False
        assert 'verifier_reason' in verification
        assert 'domain diversity' in verification['verifier_reason'].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
