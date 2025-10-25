"""
Test suite for secrets audit and environment variable security
Tests Q_021: Zero env var secrets audit with CI pattern detection
Component: DataOps_Secrets | Priority: P0 | Owner: Infra
"""

import pytest
import os
import re
from typing import Dict, List, Optional
from unittest.mock import Mock, patch, MagicMock
from core.security import (
    SecretsScanner,
    SecretPattern,
    SecretViolation,
    SecretScanResult,
    BANNED_ENV_PREFIXES,
    SECRET_PATTERNS
)


class TestSecretsScanner:
    """Test suite for SecretsScanner class"""
    
    @pytest.fixture
    def scanner(self):
        """Create a SecretsScanner instance for testing"""
        return SecretsScanner(
            patterns=SECRET_PATTERNS,
            banned_prefixes=BANNED_ENV_PREFIXES,
            raise_on_violation=True
        )
    
    def test_scanner_initialization(self, scanner):
        """Test SecretsScanner initializes with correct patterns"""
        assert len(scanner.patterns) > 0
        assert scanner.raise_on_violation is True
        assert scanner.violation_count == 0
    
    def test_detect_aws_access_key_in_env(self, scanner):
        """Test detection of AWS access key in environment variable"""
        code = 'os.environ["AWS_ACCESS_KEY_ID"] = "AKIAIOSFODNN7EXAMPLE"'
        
        with pytest.raises(SecretViolation) as exc:
            scanner.scan_code(code, filename="bad_config.py")
        
        assert "AWS_ACCESS_KEY" in str(exc.value)
        assert scanner.violation_count == 1
    
    def test_detect_api_key_pattern(self, scanner):
        """Test detection of API key pattern"""
        code = 'API_KEY = "sk-1234567890abcdef1234567890abcdef"'
        
        with pytest.raises(SecretViolation) as exc:
            scanner.scan_code(code, filename="config.py")
        
        assert "api" in str(exc.value).lower()
        assert scanner.violation_count == 1
    
    def test_detect_database_password(self, scanner):
        """Test detection of database password"""
        code = 'DB_PASSWORD = "mysecretpassword123"'
        
        with pytest.raises(SecretViolation) as exc:
            scanner.scan_code(code, filename="db_config.py")
        
        assert "password" in str(exc.value).lower()
    
    def test_detect_private_key(self, scanner):
        """Test detection of private key"""
        code = '''
        PRIVATE_KEY = """-----BEGIN RSA PRIVATE KEY-----
        MIIEpAIBAAKCAQEA0Z3VS5JJcds3xfn/ygWyF0K...
        -----END RSA PRIVATE KEY-----"""
        '''
        
        with pytest.raises(SecretViolation) as exc:
            scanner.scan_code(code, filename="keys.py")
        
        assert "PRIVATE KEY" in str(exc.value)
    
    def test_detect_jwt_token(self, scanner):
        """Test detection of JWT token"""
        code = 'token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"'
        
        with pytest.raises(SecretViolation) as exc:
            scanner.scan_code(code, filename="auth.py")
        
        assert "jwt" in str(exc.value).lower() or "token" in str(exc.value).lower()
    
    def test_detect_banned_env_prefix_api_key(self, scanner):
        """Test detection of banned API_KEY prefix"""
        code = 'os.environ["API_KEY"] = "some_value"'
        
        with pytest.raises(SecretViolation) as exc:
            scanner.scan_code(code, filename="config.py")
        
        assert "API_KEY" in str(exc.value)
        assert "banned" in str(exc.value).lower()
    
    def test_detect_banned_env_prefix_secret(self, scanner):
        """Test detection of banned SECRET_ prefix"""
        code = 'os.environ["SECRET_TOKEN"] = "abc123"'
        
        with pytest.raises(SecretViolation) as exc:
            scanner.scan_code(code, filename="config.py")
        
        assert "SECRET_" in str(exc.value)
    
    def test_detect_multiple_violations(self, scanner):
        """Test detection of multiple violations in same file"""
        code = '''
        API_KEY = "sk-test123"
        DB_PASSWORD = "admin123"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        '''
        
        with pytest.raises(SecretViolation):
            scanner.scan_code(code, filename="multiple_secrets.py")
        
        # Should detect all 3 violations
        assert scanner.violation_count >= 3
    
    def test_safe_code_no_violations(self, scanner):
        """Test that safe code with no secrets passes"""
        code = '''
        import os
        from core.config import get_secret
        
        # Correct: Using Secret Manager
        api_key = get_secret("api_key")
        db_password = get_secret("db_password")
        
        # Correct: Placeholder in config
        config = {
            "api_endpoint": "https://api.example.com",
            "timeout": 30,
            "api_key": "${API_KEY}"  # Template variable, not hardcoded
        }
        '''
        
        # Should not raise
        result = scanner.scan_code(code, filename="safe_config.py")
        assert result.violations == 0
        assert result.safe is True
    
    def test_environment_variable_reference_allowed(self, scanner):
        """Test that os.getenv() is allowed (reading, not writing)"""
        code = 'api_key = os.getenv("API_KEY", default=None)'
        
        result = scanner.scan_code(code, filename="config.py")
        assert result.safe is True
        assert result.violations == 0
    
    def test_template_variables_allowed(self, scanner):
        """Test that template variables ${VAR} are allowed"""
        code = '''
        config = {
            "db_url": "postgresql://${DB_USER}:${DB_PASSWORD}@localhost/db",
            "api_key": "${API_KEY}"
        }
        '''
        
        result = scanner.scan_code(code, filename="config.py")
        assert result.safe is True
    
    def test_scan_multiple_files(self, scanner):
        """Test scanning multiple files"""
        files = {
            "safe.py": "config = {'timeout': 30}",
            "unsafe.py": 'API_KEY = "sk-test123"'
        }
        
        results = scanner.scan_files(files)
        
        assert len(results) == 2
        assert results["safe.py"].safe is True
        assert results["unsafe.py"].safe is False
        assert results["unsafe.py"].violations > 0
    
    def test_scan_result_metadata(self, scanner):
        """Test SecretScanResult contains correct metadata"""
        code = 'API_KEY = "sk-test"'
        
        with pytest.raises(SecretViolation) as exc:
            result = scanner.scan_code(code, filename="test.py")
        
        violation = exc.value
        assert violation.filename == "test.py"
        assert violation.line_number > 0
        assert violation.pattern_name
        assert violation.matched_text


class TestSecretPatterns:
    """Test secret pattern detection rules"""
    
    def test_aws_access_key_pattern(self):
        """Test AWS access key pattern matches correctly"""
        pattern = SECRET_PATTERNS["aws_access_key"]
        
        valid_keys = [
            "AKIAIOSFODNN7EXAMPLE",
            "AKIAJG7H3LKJSDFLKJSD"
        ]
        invalid_keys = [
            "AKIA123",  # Too short
            "BKIAIOSFODNN7EXAMPLE"  # Wrong prefix
        ]
        
        for key in valid_keys:
            assert re.search(pattern.regex, key) is not None
        
        for key in invalid_keys:
            assert re.search(pattern.regex, key) is None
    
    def test_generic_api_key_pattern(self):
        """Test generic API key pattern"""
        pattern = SECRET_PATTERNS["generic_api_key"]
        
        matches = [
            'api_key = "sk-1234567890abcdef"',
            'apiKey: "sk_live_abc123def456"',
            'API_KEY = "test_abc123xyz789"'
        ]
        
        for text in matches:
            assert re.search(pattern.regex, text, re.IGNORECASE) is not None
    
    def test_password_pattern(self):
        """Test password detection pattern"""
        pattern = SECRET_PATTERNS["password"]
        
        matches = [
            'password = "mysecret123"',
            'PASSWORD: "Admin@123"',
            'db_password = "p@ssw0rd"'
        ]
        
        for text in matches:
            assert re.search(pattern.regex, text, re.IGNORECASE) is not None
    
    def test_private_key_pattern(self):
        """Test private key detection"""
        pattern = SECRET_PATTERNS["private_key"]
        
        text = "-----BEGIN RSA PRIVATE KEY-----"
        assert re.search(pattern.regex, text) is not None


class TestCIIntegration:
    """Test CI/CD pipeline integration for secrets scanning"""
    
    @pytest.fixture
    def mock_ci_env(self):
        """Mock CI environment"""
        with patch.dict(os.environ, {"CI": "true", "GITHUB_ACTIONS": "true"}):
            yield
    
    def test_ci_fails_on_secret_detection(self, mock_ci_env):
        """Test that CI pipeline fails when secrets detected"""
        scanner = SecretsScanner(raise_on_violation=True)
        
        code_with_secret = 'API_KEY = "sk-test123"'
        
        with pytest.raises(SecretViolation) as exc:
            scanner.scan_code(code_with_secret, filename="config.py")
        
        # In CI, this should cause exit code 1
        assert exc.value.exit_code == 1
    
    def test_ci_passes_on_clean_code(self, mock_ci_env):
        """Test that CI pipeline passes with clean code"""
        scanner = SecretsScanner(raise_on_violation=True)
        
        clean_code = '''
        from core.config import get_secret
        api_key = get_secret("api_key")
        '''
        
        result = scanner.scan_code(clean_code, filename="config.py")
        assert result.safe is True
        # Exit code 0 on success
        assert result.exit_code == 0
    
    def test_generate_ci_report(self):
        """Test generating CI-friendly violation report"""
        scanner = SecretsScanner(raise_on_violation=False)
        
        files = {
            "config1.py": 'API_KEY = "sk-test"',
            "config2.py": 'PASSWORD = "admin123"',
            "safe.py": "timeout = 30"
        }
        
        results = scanner.scan_files(files)
        report = scanner.generate_report(results, format="github_actions")
        
        assert "::error" in report  # GitHub Actions error format
        assert "config1.py" in report
        assert "config2.py" in report
        assert "safe.py" not in report or "✓" in report


class TestSecretManagerIntegration:
    """Test proper Secret Manager usage patterns"""
    
    def test_get_secret_from_manager(self):
        """Test correct pattern: fetching from Secret Manager"""
        from core.config import get_secret
        
        with patch('core.config.SecretManagerClient') as mock_sm:
            mock_sm.return_value.get_secret.return_value = "actual_secret_value"
            
            secret = get_secret("api_key")
            
            assert secret == "actual_secret_value"
            mock_sm.return_value.get_secret.assert_called_once_with("api_key")
    
    def test_secret_caching_with_ttl(self):
        """Test secret caching with TTL"""
        from core.config import SecretCache
        
        cache = SecretCache(ttl_seconds=300)
        
        # First call: cache miss
        with patch('core.config.SecretManagerClient') as mock_sm:
            mock_sm.return_value.get_secret.return_value = "secret_value"
            
            secret1 = cache.get("api_key")
            assert secret1 == "secret_value"
            assert mock_sm.return_value.get_secret.call_count == 1
            
            # Second call: cache hit
            secret2 = cache.get("api_key")
            assert secret2 == "secret_value"
            assert mock_sm.return_value.get_secret.call_count == 1  # Not called again
    
    def test_secret_rotation_dual_valid(self):
        """Test secret rotation with dual-valid window"""
        from core.security import SecretRotator
        
        rotator = SecretRotator(dual_valid_hours=24)
        
        with patch('core.config.SecretManagerClient') as mock_sm:
            mock_sm.return_value.get_secret.side_effect = [
                "current_secret",
                "previous_secret"
            ]
            
            # Both current and previous should be valid
            assert rotator.validate("current_secret") is True
            assert rotator.validate("previous_secret") is True
            
            # After 24h, previous should be invalid
            with patch('time.time', return_value=time.time() + 25*3600):
                assert rotator.validate("previous_secret") is False


# Acceptance Tests (Q_021)

def test_acceptance_q021_ci_fails_on_secret_pattern():
    """
    ACCEPTANCE Q_021: CI fails when secret patterns detected
    
    Given: Code with hardcoded API key
    When: CI scanner runs
    Then: Exit code 1, error message contains pattern name
    """
    scanner = SecretsScanner(raise_on_violation=True)
    
    code = '''
    # BAD: Hardcoded secret
    OPENAI_API_KEY = "sk-1234567890abcdef1234567890abcdef"
    '''
    
    with pytest.raises(SecretViolation) as exc:
        scanner.scan_code(code, filename="ci_test.py")
    
    assert exc.value.exit_code == 1
    assert "api" in str(exc.value).lower() or "key" in str(exc.value).lower()
    print("✓ Q_021 ACCEPTANCE: CI fails on secret pattern with exit 1")


def test_acceptance_q021_env_var_assignment_blocked():
    """
    ACCEPTANCE Q_021: Environment variable secret assignment blocked
    
    Given: Code with os.environ["SECRET_*"] = "..."
    When: Scanner runs
    Then: SecretViolation raised with banned prefix
    """
    scanner = SecretsScanner(raise_on_violation=True)
    
    code = '''
    import os
    os.environ["SECRET_API_KEY"] = "my_secret"
    '''
    
    with pytest.raises(SecretViolation) as exc:
        scanner.scan_code(code, filename="env_test.py")
    
    assert "SECRET_" in str(exc.value) or "banned" in str(exc.value).lower()
    print("✓ Q_021 ACCEPTANCE: Banned env var prefix detected")


def test_acceptance_q021_safe_code_passes():
    """
    ACCEPTANCE Q_021: Safe code using Secret Manager passes
    
    Given: Code using get_secret() from Secret Manager
    When: Scanner runs
    Then: No violations, safe=True, exit code 0
    """
    scanner = SecretsScanner(raise_on_violation=True)
    
    code = '''
    from core.config import get_secret
    
    # GOOD: Using Secret Manager
    api_key = get_secret("openai_api_key")
    db_password = get_secret("postgres_password")
    
    # GOOD: Template variables
    config = {
        "api_key": "${API_KEY}",
        "timeout": 30
    }
    '''
    
    result = scanner.scan_code(code, filename="safe_test.py")
    assert result.safe is True
    assert result.violations == 0
    assert result.exit_code == 0
    print("✓ Q_021 ACCEPTANCE: Safe code passes with exit 0")


if __name__ == "__main__":
    # Run acceptance tests
    test_acceptance_q021_ci_fails_on_secret_pattern()
    test_acceptance_q021_env_var_assignment_blocked()
    test_acceptance_q021_safe_code_passes()
    print("\n✓ All Q_021 acceptance criteria met")
