"""
Audit logging for compliance and decision tracking
"""
from typing import Dict, List, Optional
from datetime import datetime
import json


class AuditLogger:
    """
    Audit logger for all automated decisions
    
    Records all actions for compliance, debugging, and rollback capability
    """
    
    def __init__(self):
        self._logs: List[Dict] = []
    
    def log(self, entry: Dict):
        """
        Log an audit entry
        
        Args:
            entry: Audit log entry with timestamp, action, details
        """
        if 'timestamp' not in entry:
            entry['timestamp'] = datetime.utcnow().isoformat()
        
        self._logs.append(entry)
        
        # In production: write to immutable storage
        # self._write_to_storage(entry)
    
    def get_last_entry(self) -> Optional[Dict]:
        """Get most recent audit entry"""
        return self._logs[-1] if self._logs else None
    
    def get_entries(self, limit: int = 100) -> List[Dict]:
        """Get recent audit entries"""
        return self._logs[-limit:]
    
    def _write_to_storage(self, entry: Dict):
        """Write to persistent immutable storage (production)"""
        # TODO: Implement write to audit database with 7-year retention
        pass
