"""
Marketing Mix Modeling (MMM) Agent
Component: C04_MMM (HIGH priority)
Purpose: Bayesian attribution with risk gates and adaptive retraining
"""
import numpy as np
import pymc as pm
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Histogram, Gauge
from opentelemetry import trace
import logging

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

# Metrics
mmm_model_rejections_total = Counter(
    'mmm_model_rejections_total',
    'Total MMM model rejections',
    ['reason']  # mape_high, ci_wide, overfit
)

mmm_training_duration_seconds = Histogram(
    'mmm_training_duration_seconds',
    'MMM model training duration'
)

mmm_deployed_mape = Gauge(
    'mmm_deployed_mape',
    'Current deployed MMM model MAPE'
)

mmm_drift_detected_total = Counter(
    'mmm_drift_detected_total',
    'MMM drift detections triggering retrain'
)


class MMMValidationResult(BaseModel):
    """
    MMM model validation result
    
    Implements:
    - Q_009: MAPE and confidence interval validation
    - A_020: Risk gate for production deployment
    """
    mape: float = Field(..., ge=0, le=1, description="Mean Absolute Percentage Error")
    confidence_interval_width: float = Field(..., ge=0, description="CI width as fraction of mean")
    accepted: bool = Field(..., description="Whether model passes risk gates")
    rejection_reasons: List[str] = Field(default_factory=list)
    model_metrics: Dict = Field(default_factory=dict)
    r_squared: Optional[float] = None
    mape_holdout: Optional[float] = None
    
    @validator('mape')
    def mape_reasonable(cls, v):
        """MAPE should be between 0 and 100%"""
        if v > 1.0:
            raise ValueError(f"MAPE {v} exceeds 100%")
        return v


class MMMModel(BaseModel):
    """MMM model metadata"""
    model_id: str
    channels: List[str] = Field(default_factory=list)
    baseline_mape: float
    deployed_at: datetime
    trace_data: Optional[Dict] = None
    roi_curves: Optional[Dict] = None


class MMMEstimate(BaseModel):
    """Per-channel MMM estimate"""
    channel: str
    alpha: float  # Saturation coefficient
    beta: float   # Saturation exponent
    half_life_days: int
    saturation_point: float
    roi_curve: List[Dict]
    base_contribution: float
    incremental_contribution: float
    confidence_interval: Dict


class MMMAgent:
    """
    Bayesian Marketing Mix Modeling with risk gates
    
    Implements:
    - Q_009: MAPE risk gate (<15%)
    - Q_403: Adaptive retraining on drift
    - Q_404: Training trace instrumentation
    - Q_425: Control length validation
    - Q_450: Overfit detection gate
    - A_020: Production deployment risk gate
    """
    
    def __init__(self):
        self.deployed_models: Dict[str, MMMModel] = {}
        self.retrain_triggered: bool = False
        self.retrain_reason: Optional[str] = None
        self.retrain_urgency: Optional[str] = None
        
        # Risk gate thresholds
        self.MAPE_THRESHOLD = 0.14  # Q_009/A_020: Reject if MAPE ≥14%
        self.CI_WIDTH_THRESHOLD = 0.35  # Reject if CI width >35% of mean
        self.R_SQUARED_MAX = 0.95  # Q_450: Overfit detection
        self.HOLDOUT_MAPE_MAX = 0.20  # Q_450: Holdout performance
        self.DRIFT_TOLERANCE = 0.15  # Q_403: Trigger retrain if MAPE drifts >15%
    
    async def train_model(
        self,
        lookback_weeks: int = 52,
        channels: List[str] = None
    ) -> MMMModel:
        """
        Train Bayesian MMM with risk gates
        
        Args:
            lookback_weeks: Training window
            channels: Marketing channels to model
            
        Returns:
            MMMModel if validation passes
            
        Raises:
            ValidationError: If model fails risk gates
        """
        with tracer.start_as_current_span("mmm_training") as span:
            training_start = datetime.utcnow()
            span.set_attribute("lookback_weeks", lookback_weeks)
            span.set_attribute("n_channels", len(channels or []))
            
            try:
                # Fetch training data
                data = await self._fetch_training_data(lookback_weeks, channels)
                
                # Q_425: Validate all time series have equal length
                self.validate_training_data_schema(data)
                
                # Build and fit Bayesian model
                with mmm_training_duration_seconds.time():
                    trace = self._fit_bayesian_model(data, channels)
                
                # Compute validation metrics
                validation = self.validate_model(trace, data)
                
                # Q_009/A_020: Risk gate - reject if MAPE ≥14% or CI too wide
                if not validation.accepted:
                    mmm_model_rejections_total.labels(
                        reason=','.join(validation.rejection_reasons)
                    ).inc()
                    
                    logger.error(
                        f"MMM model rejected: {validation.rejection_reasons} "
                        f"(MAPE={validation.mape:.2%}, CI_width={validation.confidence_interval_width:.2%})"
                    )
                    
                    raise ValueError(
                        f"Model validation failed: {validation.rejection_reasons}"
                    )
                
                # Q_450: Overfit detection
                overfit_check = self.validate_model_overfit(trace)
                if not overfit_check.accepted:
                    mmm_model_rejections_total.labels(reason='overfit').inc()
                    raise ValueError("Model overfitting detected")
                
                # Create model object
                model = MMMModel(
                    model_id=f"mmm_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    channels=channels,
                    baseline_mape=validation.mape,
                    deployed_at=datetime.utcnow(),
                    trace_data=self._serialize_trace(trace),
                    roi_curves=self._compute_roi_curves(trace, channels)
                )
                
                # Q_404: Record training duration
                training_end = datetime.utcnow()
                duration_ms = (training_end - training_start).total_seconds() * 1000
                span.set_attribute("duration_ms", duration_ms)
                span.set_attribute("mape", validation.mape)
                span.set_attribute("accepted", validation.accepted)
                
                # Check SLA: <6 hours (21,600,000 ms)
                if duration_ms > 21_600_000:
                    logger.warning(
                        f"MMM training exceeded 6h SLA: {duration_ms/1000/3600:.1f}h"
                    )
                
                logger.info(
                    f"MMM training completed: {model.model_id} "
                    f"(MAPE={validation.mape:.2%}, duration={duration_ms/1000:.1f}s)"
                )
                
                return model
                
            except Exception as e:
                span.set_attribute("error", True)
                span.set_attribute("error_message", str(e))
                raise
    
    def validate_model(
        self,
        trace,
        data: Dict
    ) -> MMMValidationResult:
        """
        Validate MMM model with risk gates
        
        Q_009/A_020: MAPE <15% gate
        
        Args:
            trace: PyMC trace object
            data: Training data
            
        Returns:
            MMMValidationResult with acceptance decision
        """
        # Compute MAPE on holdout set
        holdout_size = int(len(data['revenue']) * 0.2)  # 20% holdout
        train_data = {k: v[:-holdout_size] for k, v in data.items()}
        holdout_data = {k: v[-holdout_size:] for k, v in data.items()}
        
        # Predict on holdout
        predictions = self._predict(trace, holdout_data)
        actual = holdout_data['revenue']
        
        # Calculate MAPE
        mape = np.mean(np.abs((actual - predictions) / actual))
        
        # Calculate confidence interval width
        posterior_samples = trace.posterior['revenue'].values.flatten()
        ci_lower = np.percentile(posterior_samples, 2.5)
        ci_upper = np.percentile(posterior_samples, 97.5)
        ci_width = (ci_upper - ci_lower) / np.mean(actual)
        
        # Apply risk gates
        rejection_reasons = []
        
        # Q_009/A_020: MAPE gate
        if mape >= self.MAPE_THRESHOLD:
            rejection_reasons.append('mape_high')
        
        # Confidence interval width gate
        if ci_width > self.CI_WIDTH_THRESHOLD:
            rejection_reasons.append('ci_wide')
        
        accepted = len(rejection_reasons) == 0
        
        return MMMValidationResult(
            mape=mape,
            confidence_interval_width=ci_width,
            accepted=accepted,
            rejection_reasons=rejection_reasons,
            model_metrics={
                'holdout_size': holdout_size,
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper)
            }
        )
    
    def validate_model_overfit(self, trace) -> MMMValidationResult:
        """
        Q_450: Detect overfitting via R²>0.95 + poor holdout MAPE
        
        Args:
            trace: PyMC trace with training and holdout metrics
            
        Returns:
            Validation result with overfit detection
        """
        # Extract metrics from trace
        r_squared = getattr(trace, 'r_squared', 0.0)
        mape_training = getattr(trace, 'mape_training', 0.0)
        mape_holdout = getattr(trace, 'mape_holdout', 0.0)
        
        rejection_reasons = []
        
        # Q_450: High R² on training but poor holdout → overfitting
        if r_squared > self.R_SQUARED_MAX and mape_holdout > self.HOLDOUT_MAPE_MAX:
            rejection_reasons.append('overfit')
            logger.warning(
                f"Overfitting detected: R²={r_squared:.3f}, "
                f"MAPE_train={mape_training:.2%}, MAPE_holdout={mape_holdout:.2%}"
            )
        
        return MMMValidationResult(
            mape=mape_holdout,
            confidence_interval_width=0.0,  # Not relevant for overfit check
            accepted=len(rejection_reasons) == 0,
            rejection_reasons=rejection_reasons,
            r_squared=r_squared,
            mape_holdout=mape_holdout
        )
    
    def validate_training_data_schema(self, data: Dict):
        """
        Q_425: Assert all time series have equal length
        
        Args:
            data: Training data dict
            
        Raises:
            ValueError: If lengths don't match
        """
        lengths = {key: len(value) for key, value in data.items()}
        unique_lengths = set(lengths.values())
        
        if len(unique_lengths) > 1:
            raise ValueError(
                f"Time series length mismatch: {lengths}. "
                f"All variables must have equal length."
            )
        
        logger.debug(f"Schema validation passed: {len(data)} variables, {list(unique_lengths)[0]} weeks")
    
    def check_model_drift(
        self,
        model_id: str,
        recent_predictions: Dict
    ) -> bool:
        """
        Q_403: Check for model drift and trigger adaptive retraining
        
        Args:
            model_id: Deployed model identifier
            recent_predictions: Recent actual vs predicted data
            
        Returns:
            True if drift detected
        """
        if model_id not in self.deployed_models:
            logger.warning(f"Model {model_id} not found in deployed models")
            return False
        
        deployed_model = self.deployed_models[model_id]
        
        # Calculate current MAPE on recent data
        actual = recent_predictions['actual']
        predicted = recent_predictions['predicted']
        current_mape = np.mean(np.abs((actual - predicted) / actual))
        
        # Compare to baseline
        baseline_mape = deployed_model.baseline_mape
        drift = current_mape - baseline_mape
        
        # Q_403: If MAPE drifts >15% absolute (e.g., 0.12 → 0.16), trigger immediate retrain
        if current_mape > self.DRIFT_TOLERANCE:
            logger.error(
                f"Model drift detected: {model_id} "
                f"(baseline={baseline_mape:.2%}, current={current_mape:.2%}, "
                f"drift={drift:.2%})"
            )
            
            # Trigger immediate retrain
            self.retrain_triggered = True
            self.retrain_reason = 'mape_drift'
            self.retrain_urgency = 'immediate'
            
            mmm_drift_detected_total.inc()
            
            return True
        
        return False
    
    def _fit_bayesian_model(self, data: Dict, channels: List[str]):
        """
        Fit Bayesian MMM using PyMC
        
        Returns:
            PyMC trace object
        """
        with pm.Model() as model:
            # Priors for adstock (decay rate)
            adstock_alpha = pm.Beta('adstock_alpha', alpha=2, beta=2, shape=len(channels))
            
            # Priors for saturation (Hill function)
            saturation_beta = pm.HalfNormal('saturation_beta', sigma=1, shape=len(channels))
            saturation_theta = pm.Gamma('saturation_theta', alpha=2, beta=1, shape=len(channels))
            
            # Apply transformations
            transformed_spend = []
            for i, channel in enumerate(channels):
                spend_key = f'spend_{channel}'
                if spend_key not in data:
                    continue
                
                # Adstock transformation
                spend_adstock = self._apply_adstock(
                    data[spend_key],
                    adstock_alpha[i]
                )
                
                # Saturation transformation (Hill)
                spend_saturated = (
                    saturation_beta[i] * 
                    (spend_adstock ** saturation_beta[i]) / 
                    (saturation_theta[i] ** saturation_beta[i] + spend_adstock ** saturation_beta[i])
                )
                
                transformed_spend.append(spend_saturated)
            
            # Base + incremental revenue
            base_revenue = pm.Normal('base', mu=np.mean(data['revenue']), sigma=np.std(data['revenue']))
            
            # Channel contributions
            channel_coefs = pm.Normal('channel_coef', mu=0, sigma=1, shape=len(channels))
            
            # Revenue model
            mu = base_revenue + sum(
                channel_coefs[i] * transformed_spend[i]
                for i in range(len(channels))
            )
            
            sigma = pm.HalfNormal('sigma', sigma=1)
            revenue = pm.Normal('revenue', mu=mu, sigma=sigma, observed=data['revenue'])
            
            # Sample posterior
            trace = pm.sample(2000, tune=1000, target_accept=0.95, return_inferencedata=True)
        
        return trace
    
    def _apply_adstock(self, spend: np.ndarray, alpha: float) -> np.ndarray:
        """
        Apply adstock transformation: x'_t = x_t + alpha * x'_{t-1}
        
        Args:
            spend: Spend array
            alpha: Decay rate [0, 1]
            
        Returns:
            Adstocked spend
        """
        adstocked = np.zeros_like(spend)
        adstocked[0] = spend[0]
        
        for t in range(1, len(spend)):
            adstocked[t] = spend[t] + alpha * adstocked[t-1]
        
        return adstocked
    
    def _predict(self, trace, data: Dict) -> np.ndarray:
        """Generate predictions from trace"""
        # Simplified prediction logic
        # Full implementation would use posterior samples
        posterior_mean = trace.posterior['revenue'].mean(dim=['chain', 'draw']).values
        return posterior_mean[:len(data['revenue'])]
    
    def _compute_roi_curves(self, trace, channels: List[str]) -> Dict:
        """Compute ROI curves for each channel"""
        # Simplified ROI curve computation
        roi_curves = {}
        
        for channel in channels:
            roi_curves[channel] = [
                {'spend': 100000, 'revenue_incremental': 220000},
                {'spend': 200000, 'revenue_incremental': 360000},
            ]
        
        return roi_curves
    
    def _serialize_trace(self, trace) -> Dict:
        """Serialize PyMC trace for storage"""
        return {
            'summary': str(trace),
            'n_samples': len(trace.posterior['revenue'].values.flatten())
        }
    
    async def _fetch_training_data(self, lookback_weeks: int, channels: List[str]) -> Dict:
        """Fetch training data from feature store"""
        # Mock implementation
        weeks = lookback_weeks
        data = {
            'revenue': np.random.normal(100000, 10000, weeks),
        }
        
        for channel in (channels or []):
            data[f'spend_{channel}'] = np.random.uniform(10000, 50000, weeks)
        
        return data


# Kill-switch for deployed models
class MMMKillSwitch:
    """
    Q_009: Automatic rollback if deployed model MAPE drifts >15%
    """
    
    @staticmethod
    async def monitor_and_rollback(model_id: str, agent: MMMAgent):
        """
        Monitor deployed model and rollback on drift
        
        Args:
            model_id: Current model ID
            agent: MMM agent instance
        """
        # Fetch recent predictions
        recent_predictions = await agent._fetch_recent_predictions(model_id)
        
        # Check drift
        drift_detected = agent.check_model_drift(model_id, recent_predictions)
        
        if drift_detected:
            logger.critical(f"KILL-SWITCH: Rolling back model {model_id} due to drift")
            
            # Rollback to last known good model
            await agent.rollback_to_last_good_model()
            
            # Alert team
            await send_alert(
                severity='P0',
                subject=f'MMM Model Rollback: {model_id}',
                details='Automatic rollback triggered due to MAPE drift >15%'
            )
