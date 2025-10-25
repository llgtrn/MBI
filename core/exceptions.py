"""Core exceptions for MBI system"""


class DuplicateEventError(Exception):
    """Raised when duplicate event_id detected within TTL window"""
    pass


class ConfigurationError(Exception):
    """Raised when system configuration is invalid"""
    pass


class ValidationError(Exception):
    """Raised when data validation fails"""
    pass


class PermissionDeniedError(Exception):
    """Raised when RBAC check fails"""
    pass
