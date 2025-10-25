"""
Custom Exceptions for MBI System
"""

from typing import Any, Dict, Optional
from fastapi import status


class MBIException(Exception):
    """Base exception for MBI system"""
    
    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: str = "internal_error",
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class NotFoundException(MBIException):
    """Resource not found exception"""
    
    def __init__(self, message: str = "Resource not found", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="not_found",
            details=details,
        )


class ValidationException(MBIException):
    """Validation error exception"""
    
    def __init__(self, message: str = "Validation error", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code="validation_error",
            details=details,
        )


class UnauthorizedException(MBIException):
    """Authentication error exception"""
    
    def __init__(self, message: str = "Unauthorized", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code="unauthorized",
            details=details,
        )


class ForbiddenException(MBIException):
    """Authorization error exception"""
    
    def __init__(self, message: str = "Forbidden", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
            error_code="forbidden",
            details=details,
        )


class ConflictException(MBIException):
    """Resource conflict exception"""
    
    def __init__(self, message: str = "Resource conflict", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_409_CONFLICT,
            error_code="conflict",
            details=details,
        )


class RateLimitException(MBIException):
    """Rate limit exceeded exception"""
    
    def __init__(self, message: str = "Rate limit exceeded", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_code="rate_limit_exceeded",
            details=details,
        )


class ExternalAPIException(MBIException):
    """External API error exception"""
    
    def __init__(
        self,
        message: str = "External API error",
        provider: str = "unknown",
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        details["provider"] = provider
        super().__init__(
            message=message,
            status_code=status.HTTP_502_BAD_GATEWAY,
            error_code="external_api_error",
            details=details,
        )


class ModelException(MBIException):
    """ML model error exception"""
    
    def __init__(
        self,
        message: str = "Model error",
        model_name: str = "unknown",
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        details["model_name"] = model_name
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="model_error",
            details=details,
        )


class DatabaseException(MBIException):
    """Database error exception"""
    
    def __init__(self, message: str = "Database error", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="database_error",
            details=details,
        )


class CacheException(MBIException):
    """Cache error exception"""
    
    def __init__(self, message: str = "Cache error", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="cache_error",
            details=details,
        )


class MessageBusException(MBIException):
    """Message bus error exception"""
    
    def __init__(self, message: str = "Message bus error", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="message_bus_error",
            details=details,
        )
