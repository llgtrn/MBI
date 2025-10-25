"""
Compliance Agent - Policy enforcement for age, labeling, and claims validation
Implements Q_012 (age <18 → 403) and Q_013 (Promo/広告 labeling)
"""

import re
from typing import List, Dict, Optional
from pydantic import BaseModel
from prometheus_client import Counter


# Metrics
compliance_checks_total = Counter(
    'mbi_compliance_checks_total',
    'Total compliance checks',
    ['country', 'language', 'result']
)

compliance_violations_total = Counter(
    'mbi_compliance_violations_total',
    'Total compliance violations',
    ['violation_type', 'country']
)


class ComplianceResult(BaseModel):
    """Compliance check result"""
    approved: bool
    status_code: int  # 200 OK, 403 Forbidden
    violations: List[str]
    policy_checks: Dict[str, bool]
    notes: Optional[str] = None


class ComplianceAgent:
    """Policy enforcement agent for regulatory compliance"""
    
    # Q_012: Age requirement
    MIN_AGE = 18
    
    # Q_013: Promo label patterns (must be at start of text)
    PROMO_LABEL_PATTERNS = [
        r'^\(Promo/広告\)',  # (Promo/広告)
        r'^\(Promo\)',       # (Promo)
        r'^\(広告\)',        # (広告)
        r'^\(プロモ\)',      # (プロモ)
    ]
    
    def __init__(self, policy_pack: Dict):
        """Initialize with policy pack
        
        Args:
            policy_pack: Policy configurations by country
                e.g., {'japan': {'promo_label_required': True, 'min_age': 18}}
        """
        self.policy_pack = policy_pack
    
    def check_compliance(
        self,
        text: str,
        claims: List[str],
        language: str,
        country: str,
        user_age: Optional[int] = None
    ) -> ComplianceResult:
        """Check content compliance with policy requirements
        
        Q_012 Enforcement: age < 18 → status_code=403, approved=False
        Q_013 Enforcement: No Promo/広告 label → approved=False
        
        Args:
            text: Content text to check
            claims: List of claims made in content
            language: Content language (ja, en, etc.)
            country: Target country for policy lookup
            user_age: User's age (if None, defaults to blocking)
            
        Returns:
            ComplianceResult with approval status and violations
        """
        violations = []
        policy_checks = {}
        
        # Get country policy
        policy = self.policy_pack.get(country, {})
        
        # Q_012: Age verification
        age_check_passed = self._check_age(user_age, violations, policy)
        policy_checks['age_verification'] = age_check_passed
        
        # If age check fails, return 403 immediately
        if not age_check_passed:
            result = ComplianceResult(
                approved=False,
                status_code=403,
                violations=violations,
                policy_checks=policy_checks,
                notes='Age verification failed'
            )
            
            # Record metrics
            compliance_checks_total.labels(
                country=country,
                language=language,
                result='rejected_age'
            ).inc()
            
            for violation_type in violations:
                compliance_violations_total.labels(
                    violation_type=violation_type,
                    country=country
                ).inc()
            
            return result
        
        # Q_013: Promo label enforcement (only for promotional content)
        if self._is_promotional(text, claims):
            promo_check_passed = self._check_promo_label(
                text, 
                violations, 
                policy,
                language
            )
            policy_checks['promo_label'] = promo_check_passed
        else:
            policy_checks['promo_label'] = True  # Not promotional, skip check
        
        # Other policy checks
        banned_claims_check = self._check_banned_claims(claims, violations, policy)
        policy_checks['banned_claims'] = banned_claims_check
        
        # Determine final approval
        approved = len(violations) == 0
        status_code = 200 if approved else 403
        
        result = ComplianceResult(
            approved=approved,
            status_code=status_code,
            violations=violations,
            policy_checks=policy_checks
        )
        
        # Record metrics
        compliance_checks_total.labels(
            country=country,
            language=language,
            result='approved' if approved else 'rejected_policy'
        ).inc()
        
        for violation_type in violations:
            compliance_violations_total.labels(
                violation_type=violation_type,
                country=country
            ).inc()
        
        return result
    
    def _check_age(
        self,
        user_age: Optional[int],
        violations: List[str],
        policy: Dict
    ) -> bool:
        """Q_012: Check age requirement
        
        Args:
            user_age: User's age (None = fail safe)
            violations: List to append violations to
            policy: Country policy config
            
        Returns:
            True if age check passes, False otherwise
        """
        min_age = policy.get('min_age', self.MIN_AGE)
        
        # If age not provided, fail safe (block)
        if user_age is None:
            violations.append('age_verification_required')
            return False
        
        # Q_012: age < 18 → reject with 403
        if user_age < min_age:
            violations.append(f'age_requirement: user must be >={min_age}, got {user_age}')
            return False
        
        return True
    
    def _check_promo_label(
        self,
        text: str,
        violations: List[str],
        policy: Dict,
        language: str
    ) -> bool:
        """Q_013: Check Promo/広告 label requirement
        
        Args:
            text: Content text
            violations: List to append violations to
            policy: Country policy config
            language: Content language
            
        Returns:
            True if promo label check passes, False otherwise
        """
        # Check if promo label required for this country
        if not policy.get('promo_label_required', False):
            return True  # Not required, skip check
        
        # Q_013: Check for Promo/広告 label at start of text
        has_label = any(
            re.match(pattern, text.strip())
            for pattern in self.PROMO_LABEL_PATTERNS
        )
        
        if not has_label:
            violations.append(
                f'promo_label_missing: Promotional content must start with (Promo/広告) or (Promo) or (広告)'
            )
            return False
        
        # Additional check: Label must be at the very beginning
        if not text.strip().startswith('('):
            violations.append(
                f'promo_label_position: Label must be at the start of text'
            )
            return False
        
        return True
    
    def _is_promotional(self, text: str, claims: List[str]) -> bool:
        """Detect if content is promotional
        
        Args:
            text: Content text
            claims: Claims made
            
        Returns:
            True if content appears promotional
        """
        # Simple heuristic: Look for promotional keywords
        promo_keywords = [
            'buy', 'purchase', 'offer', 'sale', 'discount', 'limited',
            '購入', '買う', 'セール', '割引', '特価', 'キャンペーン',
            'お買い得', '今すぐ'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in promo_keywords)
    
    def _check_banned_claims(
        self,
        claims: List[str],
        violations: List[str],
        policy: Dict
    ) -> bool:
        """Check for banned claims (medical, absolute, etc.)
        
        Args:
            claims: List of claims
            violations: List to append violations to
            policy: Country policy config
            
        Returns:
            True if no banned claims found
        """
        banned_patterns = policy.get('banned_claims', [
            r'\b(cure|治療|治す)\b',          # Medical cure claims
            r'\b(guaranteed|保証|100%)\b',    # Absolute guarantees
            r'\b(best|最高|No\.1)\b',        # Absolute superlatives
        ])
        
        for claim in claims:
            for pattern in banned_patterns:
                if re.search(pattern, claim, re.IGNORECASE):
                    violations.append(f'banned_claim: "{claim}" matches banned pattern {pattern}')
                    return False
        
        return True
