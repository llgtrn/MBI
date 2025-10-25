"""
Test suite for LLM Council model registry configuration
Component: C10_LLMCouncil
Capsule: Q_010
Priority: P0 (CRITICAL - Self-verification prevention)
"""

import pytest
from core.config import Config
from core.exceptions import ConfigurationError, StartupError


class TestLLMModelRegistryValidation:
    """Test LLM Analyst != Verifier enforcement"""
    
    @pytest.fixture
    def valid_config(self):
        """Valid config with different Analyst and Verifier models"""
        return {
            'llm_council': {
                'analyst': {
                    'model': 'anthropic:claude-sonnet-4.5',
                    'temperature': 0.2,
                    'max_tokens': 800
                },
                'verifier': {
                    'model': 'anthropic:claude-opus-4',  # Different model
                    'temperature': 0.1,
                    'max_tokens': 600
                }
            }
        }
    
    def test_different_models_accepted(self, valid_config):
        """Test: Analyst and Verifier with different models should succeed"""
        config = Config(valid_config)
        
        result = config.validate_llm_council()
        
        assert result.is_valid is True
        assert config.llm_council['analyst']['model'] != config.llm_council['verifier']['model']
    
    def test_same_model_raises_startup_error(self):
        """Test: Same model for Analyst and Verifier should raise StartupError"""
        invalid_config = {
            'llm_council': {
                'analyst': {
                    'model': 'anthropic:claude-sonnet-4.5',
                    'temperature': 0.2
                },
                'verifier': {
                    'model': 'anthropic:claude-sonnet-4.5',  # Same as Analyst
                    'temperature': 0.1
                }
            }
        }
        
        with pytest.raises(StartupError) as exc_info:
            config = Config(invalid_config)
            config.validate_llm_council()
        
        assert "analyst and verifier must use different models" in str(exc_info.value).lower()
        assert "self-verification" in str(exc_info.value).lower()
    
    def test_same_provider_different_model_accepted(self):
        """Test: Same provider (e.g. Anthropic) but different models is OK"""
        config_data = {
            'llm_council': {
                'analyst': {
                    'model': 'anthropic:claude-sonnet-4.5'
                },
                'verifier': {
                    'model': 'anthropic:claude-opus-4.1'  # Different model, same provider
                }
            }
        }
        
        config = Config(config_data)
        result = config.validate_llm_council()
        
        assert result.is_valid is True
    
    def test_validation_on_config_load(self):
        """Test: Config validation runs automatically on load"""
        invalid_config = {
            'llm_council': {
                'analyst': {'model': 'openai:gpt-4'},
                'verifier': {'model': 'openai:gpt-4'}  # Same
            }
        }
        
        with pytest.raises(StartupError):
            Config(invalid_config, validate_on_load=True)
    
    def test_kill_switch_bypass_validation(self):
        """Test: Kill switch ENFORCE_SEPARATE_VERIFIER can bypass (for testing)"""
        config_data = {
            'llm_council': {
                'analyst': {'model': 'anthropic:claude-sonnet-4.5'},
                'verifier': {'model': 'anthropic:claude-sonnet-4.5'}  # Same
            },
            'kill_switches': {
                'ENFORCE_SEPARATE_VERIFIER': False  # Bypass
            }
        }
        
        config = Config(config_data)
        result = config.validate_llm_council()
        
        # Should pass with warning, not error
        assert result.is_valid is True
        assert result.bypass_reason == "kill_switch_disabled"
        assert result.warnings is not None


class TestModelRegistryYAMLValidation:
    """Test model_registry.yaml schema validation"""
    
    def test_valid_registry_schema(self, tmp_path):
        """Test: Valid model_registry.yaml passes validation"""
        registry_content = """
registry:
  managed_main:
    tasks: [summary, creative, complex_reasoning]
    models: ["anthropic:claude-sonnet-4.5"]
    max_tokens: 800
    temperature: 0.2
    
  managed_verifier:
    tasks: [policy_check, factual_verify]
    models: ["anthropic:claude-opus-4"]  # Different from managed_main
    max_tokens: 600
    temperature: 0.1

routing_policy:
  crisis_brief:
    primary: managed_main
    verify: managed_verifier  # Different model enforced
"""
        
        registry_file = tmp_path / "model_registry.yaml"
        registry_file.write_text(registry_content)
        
        config = Config(registry_path=str(registry_file))
        result = config.validate_model_registry()
        
        assert result.is_valid is True
    
    def test_invalid_registry_same_model_both_roles(self, tmp_path):
        """Test: Same model in both primary and verify roles should fail"""
        invalid_registry = """
registry:
  managed_main:
    models: ["anthropic:claude-sonnet-4.5"]
    
routing_policy:
  crisis_brief:
    primary: managed_main
    verify: managed_main  # Same registry entry = same model
"""
        
        registry_file = tmp_path / "model_registry.yaml"
        registry_file.write_text(invalid_registry)
        
        with pytest.raises(ConfigurationError) as exc_info:
            config = Config(registry_path=str(registry_file))
            config.validate_model_registry()
        
        assert "same model for analyst and verifier" in str(exc_info.value).lower()
    
    def test_metrics_emitted_on_validation_failure(self, mocker):
        """Test: Config validation failure emits prometheus metric"""
        mock_counter = mocker.patch('core.config.config_error_counter')
        
        invalid_config = {
            'llm_council': {
                'analyst': {'model': 'anthropic:claude-sonnet-4.5'},
                'verifier': {'model': 'anthropic:claude-sonnet-4.5'}
            }
        }
        
        with pytest.raises(StartupError):
            Config(invalid_config, validate_on_load=True)
        
        mock_counter.labels.assert_called_with(error_type='model_registry_violation')
        mock_counter.labels().inc.assert_called_once()
