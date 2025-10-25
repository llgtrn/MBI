"""
Model Registry Configuration Loader and Validator

Implements Q_074: Verifier model separation enforcement
"""

import yaml
from typing import Dict, Any
from pathlib import Path


class ConfigError(Exception):
    """Configuration validation error"""
    pass


class ModelRegistry:
    """
    Model registry configuration manager
    
    Enforces Q_074: Analyst and Verifier must use different models
    """
    
    @staticmethod
    def load(config_path: str = "config/model_registry.yaml") -> Dict[str, Any]:
        """
        Load model registry configuration
        
        Args:
            config_path: Path to config file
            
        Returns:
            Parsed configuration dict
            
        Raises:
            ConfigError: If validation fails
        """
        path = Path(config_path)
        if not path.exists():
            raise ConfigError(f"Config file not found: {config_path}")
        
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate council configuration (Q_074)
        ModelRegistry.validate_council(config)
        
        return config
    
    @staticmethod
    def validate_council(config: Dict[str, Any]) -> bool:
        """
        Validate LLM Council configuration
        
        Implements Q_074: Enforce analyst != verifier models
        
        Args:
            config: Configuration dict
            
        Returns:
            True if validation passes
            
        Raises:
            ConfigError: If analyst and verifier use same model
        """
        if 'council' not in config:
            raise ConfigError("Missing 'council' section in config")
        
        council = config['council']
        
        if 'analyst' not in council or 'verifier' not in council:
            raise ConfigError("Council must have both 'analyst' and 'verifier' sections")
        
        analyst_model = council['analyst'].get('model')
        verifier_model = council['verifier'].get('model')
        
        if not analyst_model or not verifier_model:
            raise ConfigError("Both analyst and verifier must specify 'model'")
        
        # Q_074: Enforce separation
        if analyst_model == verifier_model:
            raise ConfigError(
                f"Q_074 VIOLATION: Analyst and Verifier cannot use the same model. "
                f"Got analyst={analyst_model}, verifier={verifier_model}. "
                f"Self-verification is not allowed. Use different models."
            )
        
        # Check validation rules
        validation_rules = config.get('validation_rules', {})
        if validation_rules.get('analyst_verifier_separation', True):
            # Additional check: ensure model_keys are also different
            analyst_key = council['analyst'].get('model_key')
            verifier_key = council['verifier'].get('model_key')
            
            if analyst_key and verifier_key and analyst_key == verifier_key:
                raise ConfigError(
                    f"Analyst and Verifier model_keys must be different. "
                    f"Got both={analyst_key}"
                )
        
        return True
    
    @staticmethod
    def get_model_config(config: Dict[str, Any], role: str) -> Dict[str, Any]:
        """
        Get model configuration for a specific role
        
        Args:
            config: Configuration dict
            role: Role name (analyst, verifier, retriever)
            
        Returns:
            Model configuration for the role
        """
        if role not in config.get('council', {}):
            raise ConfigError(f"Role '{role}' not found in council config")
        
        return config['council'][role]


# Startup validation
def validate_config_on_startup(config_path: str = "config/model_registry.yaml"):
    """
    Validate configuration on application startup
    
    This should be called during app initialization to catch
    configuration errors early (fail-fast principle)
    """
    try:
        config = ModelRegistry.load(config_path)
        print(f"✓ Model registry configuration validated successfully")
        print(f"  - Analyst: {config['council']['analyst']['model']}")
        print(f"  - Verifier: {config['council']['verifier']['model']}")
        return config
    except ConfigError as e:
        print(f"✗ Configuration validation FAILED: {e}")
        raise


if __name__ == "__main__":
    # Test validation
    validate_config_on_startup()
