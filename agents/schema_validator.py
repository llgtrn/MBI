"""
Schema validation agent with backward compatibility checking
Detects breaking changes in API schemas and triggers alerts

Features:
- Schema registry with version tracking
- Backward compatibility validation
- Breaking change detection (field removal, type change, required field addition)
- Migration guide generation
- CI integration (fail on breaking changes)
- Alert emission within 5min
- Prometheus metrics
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
import logging

from core.contracts import SchemaChangeEvent


logger = logging.getLogger(__name__)


class ChangeSeverity(Enum):
    """Severity levels for schema changes"""
    BREAKING = "BREAKING"
    WARNING = "WARNING"
    INFO = "INFO"


class ChangeType(Enum):
    """Types of schema changes"""
    FIELD_REMOVED = "field_removed"
    FIELD_ADDED = "field_added"
    TYPE_CHANGED = "type_changed"
    REQUIRED_FIELD_ADDED = "required_field_added"
    REQUIRED_FIELD_REMOVED = "required_field_removed"
    ENUM_VALUE_ADDED = "enum_value_added"
    ENUM_VALUE_REMOVED = "enum_value_removed"
    CONSTRAINT_CHANGED = "constraint_changed"


@dataclass
class BreakingChange:
    """Detected schema change"""
    change_type: str
    field_path: str
    severity: str
    description: str
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    suggested_action: Optional[str] = None


@dataclass
class CompatibilityResult:
    """Result of backward compatibility check"""
    is_compatible: bool
    breaking_changes: List[BreakingChange]
    warnings: List[str]
    migration_required: bool
    migration_guide: Optional[List[str]] = None
    ci_should_fail: bool = False
    exit_code: int = 0
    
    def __post_init__(self):
        """Set CI failure based on compatibility"""
        if not self.is_compatible:
            self.ci_should_fail = True
            self.exit_code = 1


@dataclass
class SchemaVersion:
    """Schema version metadata"""
    provider: str
    version: str
    schema: Dict[str, Any]
    registered_at: datetime
    deprecated: bool = False


class SchemaValidator:
    """
    Schema validation with backward compatibility checking
    
    Features:
    - Register schemas with version tracking
    - Validate data against schemas
    - Detect breaking changes between versions
    - Generate migration guides
    - Emit alerts on drift
    - Prometheus metrics
    """
    
    def __init__(
        self,
        registry_path: str = "SSOT/COVERAGE/schema_registry.json",
        enable_strict_validation: Optional[bool] = None
    ):
        self.registry_path = Path(registry_path)
        self.schemas: Dict[str, Dict[str, SchemaVersion]] = {}
        self.change_handlers: List[Callable] = []
        self.prometheus_counter = None
        
        # Kill switch: ENABLE_STRICT_SCHEMA_VALIDATION
        if enable_strict_validation is None:
            import os
            enable_strict_validation = os.getenv("ENABLE_STRICT_SCHEMA_VALIDATION", "true").lower() == "true"
        
        self.strict_validation = enable_strict_validation
        
        # Initialize prometheus if available
        try:
            from prometheus_client import Counter
            self.prometheus_counter = Counter(
                'schema_drift_detected_total',
                'Total schema drift detections',
                ['provider', 'old_version', 'new_version', 'severity']
            )
        except ImportError:
            logger.warning("Prometheus client not available; metrics disabled")
    
    def register_schema(
        self,
        provider: str,
        version: str,
        schema: Dict[str, Any]
    ) -> None:
        """Register a schema version"""
        if provider not in self.schemas:
            self.schemas[provider] = {}
        
        self.schemas[provider][version] = SchemaVersion(
            provider=provider,
            version=version,
            schema=schema,
            registered_at=datetime.utcnow()
        )
        
        logger.info(f"Registered schema {provider} {version}")
    
    def has_schema(self, provider: str, version: str) -> bool:
        """Check if schema version is registered"""
        return provider in self.schemas and version in self.schemas[provider]
    
    def get_schema(self, provider: str, version: str) -> Optional[SchemaVersion]:
        """Get registered schema version"""
        if self.has_schema(provider, version):
            return self.schemas[provider][version]
        return None
    
    def validate_data(
        self,
        provider: str,
        version: str,
        data: Dict[str, Any],
        strict: bool = True
    ) -> bool:
        """
        Validate data against schema version
        
        Args:
            provider: Provider name (e.g., 'meta')
            version: Schema version (e.g., 'v18.0')
            data: Data to validate
            strict: Raise exception on validation failure
        
        Returns:
            True if valid, False otherwise
        
        Raises:
            ValueError: If strict=True and validation fails
        """
        schema_version = self.get_schema(provider, version)
        if not schema_version:
            if strict:
                raise ValueError(f"Schema not found: {provider} {version}")
            return False
        
        schema = schema_version.schema
        fields = schema.get("fields", {})
        
        # Check required fields
        for field_name, field_def in fields.items():
            if field_def.get("required", False) and field_name not in data:
                if strict:
                    raise ValueError(f"Required field missing: {field_name}")
                return False
        
        # Check field types
        for field_name, value in data.items():
            if field_name not in fields:
                continue  # Extra fields OK in non-strict mode
            
            field_def = fields[field_name]
            expected_type = field_def.get("type")
            
            if expected_type == "string" and not isinstance(value, str):
                if strict:
                    raise ValueError(f"Field {field_name}: expected string, got {type(value).__name__}")
                return False
            elif expected_type == "number" and not isinstance(value, (int, float)):
                if strict:
                    raise ValueError(f"Field {field_name}: expected number, got {type(value).__name__}")
                return False
            elif expected_type == "integer" and not isinstance(value, int):
                if strict:
                    raise ValueError(f"Field {field_name}: expected integer, got {type(value).__name__}")
                return False
        
        return True
    
    def validate_backward_compatibility(
        self,
        provider: str,
        old_version: str,
        new_version: str,
        new_schema: Dict[str, Any]
    ) -> CompatibilityResult:
        """
        Validate backward compatibility between schema versions
        
        Detects breaking changes:
        - Field removal
        - Type change
        - New required field
        - Enum value removal
        
        Non-breaking changes:
        - New optional field
        - Enum value addition
        
        Returns:
            CompatibilityResult with compatibility status and detected changes
        """
        old_schema_version = self.get_schema(provider, old_version)
        if not old_schema_version:
            raise ValueError(f"Old schema not found: {provider} {old_version}")
        
        old_schema = old_schema_version.schema
        old_fields = old_schema.get("fields", {})
        new_fields = new_schema.get("fields", {})
        
        breaking_changes: List[BreakingChange] = []
        warnings: List[str] = []
        
        # Check for removed fields
        for field_name in old_fields:
            if field_name not in new_fields:
                breaking_changes.append(BreakingChange(
                    change_type=ChangeType.FIELD_REMOVED.value,
                    field_path=field_name,
                    severity=ChangeSeverity.BREAKING.value,
                    description=f"Field removed: {field_name}",
                    old_value=old_fields[field_name],
                    suggested_action=f"Add migration for {field_name} removal or provide default value"
                ))
        
        # Check for type changes and new required fields
        for field_name, new_field_def in new_fields.items():
            if field_name not in old_fields:
                # New field
                if new_field_def.get("required", False):
                    # New required field is BREAKING
                    breaking_changes.append(BreakingChange(
                        change_type=ChangeType.REQUIRED_FIELD_ADDED.value,
                        field_path=field_name,
                        severity=ChangeSeverity.BREAKING.value,
                        description=f"New required field added: {field_name}",
                        new_value=new_field_def,
                        suggested_action=f"Make {field_name} optional or provide default value"
                    ))
                else:
                    # New optional field is OK (warning only)
                    warnings.append(f"New optional field: {field_name}")
            else:
                # Existing field
                old_field_def = old_fields[field_name]
                
                # Check type change
                old_type = old_field_def.get("type")
                new_type = new_field_def.get("type")
                
                if old_type != new_type:
                    breaking_changes.append(BreakingChange(
                        change_type=ChangeType.TYPE_CHANGED.value,
                        field_path=field_name,
                        severity=ChangeSeverity.BREAKING.value,
                        description=f"Type changed: {field_name} ({old_type} → {new_type})",
                        old_value=old_type,
                        new_value=new_type,
                        suggested_action=f"Add type conversion or keep {old_type}"
                    ))
                
                # Check enum changes
                if "enum" in old_field_def or "enum" in new_field_def:
                    old_enum = set(old_field_def.get("enum", []))
                    new_enum = set(new_field_def.get("enum", []))
                    
                    removed_values = old_enum - new_enum
                    added_values = new_enum - old_enum
                    
                    if removed_values:
                        breaking_changes.append(BreakingChange(
                            change_type=ChangeType.ENUM_VALUE_REMOVED.value,
                            field_path=field_name,
                            severity=ChangeSeverity.BREAKING.value,
                            description=f"Enum values removed from {field_name}: {removed_values}",
                            old_value=list(old_enum),
                            new_value=list(new_enum),
                            suggested_action=f"Keep removed enum values or add migration"
                        ))
                    
                    if added_values:
                        warnings.append(f"Enum values added to {field_name}: {added_values}")
        
        # Determine compatibility
        is_compatible = len(breaking_changes) == 0
        
        # In non-strict mode, treat breaking changes as warnings
        if not self.strict_validation:
            warnings.extend([f"[NON-STRICT] {c.description}" for c in breaking_changes])
            breaking_changes = []
            is_compatible = True
        
        # Generate migration guide if needed
        migration_guide = None
        if breaking_changes:
            migration_guide = self._generate_migration_guide(
                provider,
                old_version,
                new_version,
                breaking_changes
            )
        
        result = CompatibilityResult(
            is_compatible=is_compatible,
            breaking_changes=breaking_changes,
            warnings=warnings,
            migration_required=len(breaking_changes) > 0,
            migration_guide=migration_guide
        )
        
        # Emit schema change event
        if breaking_changes or warnings:
            event = SchemaChangeEvent(
                provider=provider,
                old_version=old_version,
                new_version=new_version,
                is_breaking=not is_compatible,
                changes=[c.description for c in breaking_changes],
                warnings=warnings,
                timestamp=datetime.utcnow()
            )
            self._emit_schema_change(event)
            
            # Increment Prometheus metric
            if self.prometheus_counter:
                severity = "breaking" if not is_compatible else "warning"
                self.prometheus_counter.labels(
                    provider=provider,
                    old_version=old_version,
                    new_version=new_version,
                    severity=severity
                ).inc()
        
        return result
    
    def _generate_migration_guide(
        self,
        provider: str,
        old_version: str,
        new_version: str,
        breaking_changes: List[BreakingChange]
    ) -> List[str]:
        """Generate migration guide for breaking changes"""
        guide = [
            f"Migration Guide: {provider} {old_version} → {new_version}",
            f"Generated: {datetime.utcnow().isoformat()}",
            "",
            "Breaking Changes:"
        ]
        
        for i, change in enumerate(breaking_changes, 1):
            guide.append(f"{i}. {change.description}")
            if change.suggested_action:
                guide.append(f"   Action: {change.suggested_action}")
            guide.append("")
        
        guide.append("Recommended Migration Steps:")
        guide.append("1. Review all breaking changes above")
        guide.append("2. Update data models to handle changes")
        guide.append("3. Add backward compatibility layer if needed")
        guide.append(f"4. Test with both {old_version} and {new_version}")
        guide.append("5. Update documentation")
        
        return guide
    
    def on_schema_change(self, handler: Callable[[SchemaChangeEvent], None]) -> None:
        """Register schema change event handler"""
        self.change_handlers.append(handler)
    
    def _emit_schema_change(self, event: SchemaChangeEvent) -> None:
        """Emit schema change event to all handlers"""
        for handler in self.change_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Schema change handler failed: {e}")
    
    def save_registry(self) -> None:
        """Save schema registry to disk"""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        registry_data = {}
        for provider, versions in self.schemas.items():
            registry_data[provider] = {}
            for version, schema_version in versions.items():
                registry_data[provider][version] = {
                    "schema": schema_version.schema,
                    "registered_at": schema_version.registered_at.isoformat(),
                    "deprecated": schema_version.deprecated
                }
        
        with open(self.registry_path, 'w') as f:
            json.dump(registry_data, f, indent=2)
        
        logger.info(f"Saved schema registry to {self.registry_path}")
    
    def load_registry(self) -> None:
        """Load schema registry from disk"""
        if not self.registry_path.exists():
            logger.warning(f"Schema registry not found: {self.registry_path}")
            return
        
        with open(self.registry_path, 'r') as f:
            registry_data = json.load(f)
        
        for provider, versions in registry_data.items():
            for version, data in versions.items():
                self.schemas.setdefault(provider, {})[version] = SchemaVersion(
                    provider=provider,
                    version=version,
                    schema=data["schema"],
                    registered_at=datetime.fromisoformat(data["registered_at"]),
                    deprecated=data.get("deprecated", False)
                )
        
        logger.info(f"Loaded {len(self.schemas)} providers from registry")
