"""
MMM Model Monitoring — Drift Detection & Auto-Retrain Triggers

Metrics Tracked:
- MAPE (Mean Absolute Percentage Error)
- R² (coefficient of determination)
- Holdout accuracy
- Statistical drift (KS test)

Auto-retrain triggers:
- MAPE degradation > 5% from baseline
- R² drop > 0.1 from baseline
- KS test p-value < 0.05 (distribution shift)
"""

from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
from scipy import stats
import numpy as np


class MMMMonitoring:
    """
    Monitor MMM model performance and detect drift
    
    Examples:
        >>> monitor = MMMMonitoring(baseline_mape=0.12, baseline_r2=0.89)
        >>> drift = monitor.detect_model_drift(current_mape=0.18, current_r2=0.85)
        >>> assert drift.drift_detected == True
        >>> assert drift.reason == "mape_degradation"
    """
    
    def __init__(
        self,
        baseline_mape: float = 0.12,
        baseline_r2: float = 0.89,
        mape_threshold: float = 0.05,
        r2_threshold: float = 0.1,
        ks_threshold: float = 0.05
    ):
        self.baseline_mape = baseline_mape
        self.baseline_r2 = baseline_r2
        self.mape_threshold = mape_threshold
        self.r2_threshold = r2_threshold
        self.ks_threshold = ks_threshold
    
    def detect_model_drift(
        self,
        current_mape: float,
        current_r2: float,
        historical_predictions: Optional[np.ndarray] = None,
        current_predictions: Optional[np.ndarray] = None
    ) -> 'DriftResult':
        """
        Detect if model has drifted and needs retraining
        
        Args:
            current_mape: Current MAPE metric
            current_r2: Current R² metric
            historical_predictions: Historical prediction distribution
            current_predictions: Current prediction distribution
        
        Returns:
            DriftResult with drift_detected flag and reason
        """
        drift_detected = False
        reasons = []
        
        # MAPE degradation check
        mape_change = current_mape - self.baseline_mape
        if mape_change > self.mape_threshold:
            drift_detected = True
            reasons.append(f"mape_degradation: {mape_change:.3f} > {self.mape_threshold}")
        
        # R² drop check
        r2_change = self.baseline_r2 - current_r2
        if r2_change > self.r2_threshold:
            drift_detected = True
            reasons.append(f"r2_drop: {r2_change:.3f} > {self.r2_threshold}")
        
        # Distribution shift check (KS test)
        if historical_predictions is not None and current_predictions is not None:
            ks_statistic, p_value = stats.ks_2samp(historical_predictions, current_predictions)
            if p_value < self.ks_threshold:
                drift_detected = True
                reasons.append(f"distribution_shift: p={p_value:.4f} < {self.ks_threshold}")
        
        return DriftResult(
            drift_detected=drift_detected,
            reasons=reasons,
            current_mape=current_mape,
            current_r2=current_r2,
            mape_change=mape_change,
            r2_change=r2_change,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
    
    def should_retrain(self, drift_result: 'DriftResult') -> bool:
        """
        Determine if auto-retrain should be triggered
        
        Args:
            drift_result: Result from drift detection
        
        Returns:
            True if model should be retrained
        """
        # Retrain if drift detected
        if drift_result.drift_detected:
            return True
        
        # Also retrain if MAPE > 20% regardless of drift
        if drift_result.current_mape > 0.20:
            return True
        
        return False
    
    def log_metrics(
        self,
        metrics: Dict,
        log_path: str = "SSOT/METRICS/mmm_metrics.jsonl"
    ):
        """
        Append metrics to JSONL log
        
        Args:
            metrics: Dictionary of metrics to log
            log_path: Path to JSONL log file
        """
        log_file = Path(log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **metrics
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')


class DriftResult:
    """
    Result from drift detection
    
    Attributes:
        drift_detected: Whether drift was detected
        reasons: List of reasons for drift
        current_mape: Current MAPE value
        current_r2: Current R² value
        mape_change: Change from baseline MAPE
        r2_change: Change from baseline R²
        timestamp: ISO 8601 UTC timestamp
    """
    
    def __init__(
        self,
        drift_detected: bool,
        reasons: list,
        current_mape: float,
        current_r2: float,
        mape_change: float,
        r2_change: float,
        timestamp: str
    ):
        self.drift_detected = drift_detected
        self.reasons = reasons
        self.current_mape = current_mape
        self.current_r2 = current_r2
        self.mape_change = mape_change
        self.r2_change = r2_change
        self.timestamp = timestamp
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "drift_detected": self.drift_detected,
            "reasons": self.reasons,
            "current_mape": self.current_mape,
            "current_r2": self.current_r2,
            "mape_change": self.mape_change,
            "r2_change": self.r2_change,
            "timestamp": self.timestamp
        }
    
    @property
    def reason(self) -> Optional[str]:
        """Get primary reason (first in list)"""
        return self.reasons[0] if self.reasons else None


def compute_model_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray
) -> Dict[str, float]:
    """
    Compute standard MMM metrics
    
    Args:
        predictions: Model predictions
        actuals: Actual values
    
    Returns:
        Dictionary with mape, r_squared, mae, rmse
    """
    # MAPE
    mape = np.mean(np.abs((actuals - predictions) / actuals))
    
    # R²
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # MAE
    mae = np.mean(np.abs(actuals - predictions))
    
    # RMSE
    rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
    
    return {
        "mape": float(mape),
        "r_squared": float(r_squared),
        "mae": float(mae),
        "rmse": float(rmse)
    }


# Auto-retrain trigger threshold
AUTO_RETRAIN_MAPE_THRESHOLD = 0.20
AUTO_RETRAIN_DRIFT_THRESHOLD = 0.05
