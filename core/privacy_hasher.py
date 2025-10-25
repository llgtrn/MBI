"""
Privacy Hasher - PII hashing with salt rotation support
Uses security module for salt management
"""
import hashlib
from typing import Dict, Optional
from core.security import SecurityManager


class PrivacyHasher:
    """
    Hash PII with rotated salt for GDPR compliance
    Supports email, phone, and other PII fields
    """
    
    def __init__(self, security_manager: Optional[SecurityManager] = None):
        self.security_manager = security_manager or SecurityManager()
        self._current_salt: Optional[str] = None
    
    async def hash_pii(self, data: Dict) -> Dict:
        """
        Hash all PII fields in data dictionary
        
        Args:
            data: Dictionary containing PII fields (email, phone, name, etc.)
            
        Returns:
            Dictionary with PII fields replaced by hashes
        """
        # Get current salt (with rotation support)
        salt = await self.security_manager.get_pii_salt()
        self._current_salt = salt
        
        hashed_data = data.copy()
        pii_fields = ['email', 'phone', 'name', 'address']
        
        for field in pii_fields:
            if field in data and data[field]:
                # Hash: SHA256(value + salt)
                hashed_value = self._hash_value(data[field], salt)
                hashed_data[f'{field}_hash'] = hashed_value
                
                # Remove plaintext PII
                del hashed_data[field]
        
        return hashed_data
    
    def _hash_value(self, value: str, salt: str) -> str:
        """
        Hash a single value with salt
        
        Args:
            value: Plain text value
            salt: Salt from Secret Manager
            
        Returns:
            SHA256 hash as hex string
        """
        combined = f"{value}{salt}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    async def hash_email(self, email: str) -> str:
        """Hash email address"""
        salt = await self.security_manager.get_pii_salt()
        return self._hash_value(email.lower().strip(), salt)
    
    async def hash_phone(self, phone: str) -> str:
        """Hash phone number"""
        salt = await self.security_manager.get_pii_salt()
        # Normalize phone (remove non-digits)
        normalized = ''.join(c for c in phone if c.isdigit())
        return self._hash_value(normalized, salt)
    
    async def verify_hash(self, value: str, hash_to_verify: str) -> bool:
        """
        Verify if a value matches a hash
        
        Args:
            value: Plain text value to verify
            hash_to_verify: Hash to check against
            
        Returns:
            True if value hashes to the same value
        """
        salt = await self.security_manager.get_pii_salt()
        computed_hash = self._hash_value(value, salt)
        return computed_hash == hash_to_verify
