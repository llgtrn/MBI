"""
Core configuration management with LLM Council validation
Component: C10_LLMCouncil
Enhancement: Q_010 - Analyst != Verifier enforcement
"""
import os
import yaml
from typing import Any, Dict, Optional
from dataclasses import dataclass
from pathlib import Path

from prometheus_client import Counter

from core.exceptions import ConfigurationError, StartupError


# Prometheus metrics
config_error_counter = Counter(
    'config_validation_errors_total',
    'Total configuration validation errors',
    ['error_type']
)


@dataclass
class ValidationResult:
    """Result of configuration validation"""
    is_valid: bool
    errors: Optional[list] = None
    warnings: Optional[list] = None
    bypass_reason: Optional[str] = None


class Config:
    """
    Configuration management with validation
    
    LLM Council Rules (Q_010):
    - Analyst and Verifier MUST use different models
    - Same model for both roles raises StartupError
    - Kill switch: ENFORCE_SEPARATE_VERIFIER (default: True)
    """
    
    def __init__(
        self,
        config_data: Optional[Dict] = None,
        registry_path: Optional[str] = None,
        validate_on_load: bool = True
    ):
        # Base configuration
        self._config = {
            # MTA settings
            'MTA_MIN_USERS_PER_PATH': int(os.getenv('MTA_MIN_USERS_PER_PATH', '10')),
            'MTA_PRIVACY_ENFORCEMENT': os.getenv('MTA_PRIVACY_ENFORCEMENT', 'true').lower() == 'true',
            
            # Budget allocation settings
            'BUDGET_ALLOCATION_ENABLED': os.getenv('BUDGET_ALLOCATION_ENABLED', 'true').lower() == 'true',
            
            # Compliance settings
            'PROMO_LABEL_ENFORCEMENT': os.getenv('PROMO_LABEL_ENFORCEMENT', 'true').lower() == 'true',
            
            # LLM settings
            'LLM_MAX_TEMPERATURE': float(os.getenv('LLM_MAX_TEMPERATURE', '0.2')),
            'LLM_MIN_SOURCES': int(os.getenv('LLM_MIN_SOURCES', '2')),
        }
        
        # Kill switches (default all enabled)
        self.kill_switches = {
            'ENABLE_MTA_PRIVACY_CHECK': True,
            'ENABLE_SALT_ROTATION': True,
            'ENABLE_DRIFT_ALERTS': True,
            'ENABLE_FRESHNESS_CHECK': True,
            'ENFORCE_SEPARATE_VERIFIER': True,  # Q_010
        }
        
        # LLM Council configuration
        self.llm_council = {}
        
        # Load from config_data if provided
        if config_data:
            if 'llm_council' in config_data:
                self.llm_council = config_data['llm_council']
            if 'kill_switches' in config_data:
                self.kill_switches.update(config_data['kill_switches'])
        
        # Load model registry if path provided
        if registry_path:
            self._load_model_registry(registry_path)
        
        # Validate on load if requested
        if validate_on_load and self.llm_council:
            self.validate_llm_council()
    
    def _load_model_registry(self, path: str):
        """Load model registry from YAML file"""
        registry_file = Path(path)
        if not registry_file.exists():
            raise ConfigurationError(f"Model registry not found: {path}")
        
        with open(registry_file, 'r') as f:
            registry = yaml.safe_load(f)
        
        self.model_registry = registry
    
    def validate_llm_council(self) -> ValidationResult:
        """
        Validate LLM Council configuration
        
        Enforcement (Q_010):
        - Analyst model != Verifier model (MUST be different)
        - Raises StartupError if same model detected
        - Emit metric on violation
        
        Kill Switch:
        - ENFORCE_SEPARATE_VERIFIER=False bypasses (for testing only)
        
        Returns:
            ValidationResult with is_valid flag
            
        Raises:
            StartupError: If models are the same and enforcement enabled
        """
        # Kill switch bypass
        if not self.kill_switches.get('ENFORCE_SEPARATE_VERIFIER', True):
            return ValidationResult(
                is_valid=True,
                bypass_reason="kill_switch_disabled",
                warnings=["LLM Council validation bypassed - not for production use"]
            )
        
        if not self.llm_council:
            return ValidationResult(is_valid=True)
        
        # Extract models
        analyst_model = self.llm_council.get('analyst', {}).get('model')
        verifier_model = self.llm_council.get('verifier', {}).get('model')
        
        if not analyst_model or not verifier_model:
            return ValidationResult(
                is_valid=False,
                errors=["Both analyst and verifier models must be configured"]
            )
        
        # Validate models are different
        if analyst_model == verifier_model:
            error_msg = (
                f"LLM Council configuration error: Analyst and Verifier must use "
                f"different models to prevent self-verification. "
                f"Current: both using '{analyst_model}'. "
                f"This violates security policy Q_010."
            )
            
            # Emit metric
            config_error_counter.labels(
                error_type='model_registry_violation'
            ).inc()
            
            raise StartupError(error_msg)
        
        return ValidationResult(is_valid=True)
    
    def validate_model_registry(self) -> ValidationResult:
        """
        Validate model_registry.yaml routing policies
        
        Checks that routing policies using both 'primary' and 'verify'
        reference different model registry entries (which must have
        different models).
        
        Returns:
            ValidationResult
            
        Raises:
            ConfigurationError: If same model used for both roles
        """
        if not hasattr(self, 'model_registry'):
            return ValidationResult(is_valid=True)
        
        registry = self.model_registry.get('registry', {})
        routing = self.model_registry.get('routing_policy', {})
        
        errors = []
        
        for task_name, policy in routing.items():
            if 'primary' in policy and 'verify' in policy:
                primary_entry = policy['primary']
                verify_entry = policy['verify']
                
                # Same registry entry = same model
                if primary_entry == verify_entry:
                    error_msg = (
                        f"Task '{task_name}': Same model for Analyst and Verifier. "
                        f"Both use registry entry '{primary_entry}'. "
                        f"Violates Q_010 policy."
                    )
                    errors.append(error_msg)
                    
                    # Emit metric
                    config_error_counter.labels(
                        error_type='model_registry_violation'
                    ).inc()
        
        if errors:
            raise ConfigurationError("\n".join(errors))
        
        return ValidationResult(is_valid=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        self._config[key] = value


_global_config = Config()


def get_config() -> Config:
    """Get global configuration instance"""
    return _global_config
