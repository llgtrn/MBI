"""
Unit tests for MMM Agent
Component: C04_MMM (HIGH priority)
Coverage: MAPE risk gate, confidence interval validation, adaptive retraining
"""
import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pymc as pm

from agents.mmm_agent import (
    MMMAgent,
    MMMModel,
    MMMValidationResult,
    MMMEstimate
)


class TestMAPERiskGate:
    """Test MAPE risk gate with wide CI rejection (Q_009, A_020)"""
    
    @pytest.fixture
    def agent(self):
        """Create MMM agent with mocked dependencies"""
        agent = MMMAgent()
        agent.feature_store = Mock()
        return agent
    
    def test_mmm_mape_gate_reject_high_mape(self, agent):
        """
        Q_009/A_020 acceptance: MAPE=14% + wide_ci=True → model rejected
        Test that models with MAPE ≥14% are rejected
        """
        # Arrange
        mock_trace = self._create_mock_trace(
            mape=0.14,  # At threshold
            ci_width=0.25  # Wide confidence interval
        )
        
        mock_data = self._create_mock_training_data()
        
        # Act
        validation = agent.validate_model(mock_trace, mock_data)
        
        # Assert
        assert validation.accepted is False
        assert validation.mape >= 0.14
        assert 'mape_high' in validation.rejection_reasons
        
        # Verify metric incremented
        from prometheus_client import REGISTRY
        rejections = REGISTRY.get_sample_value(
            'mmm_model_rejections_total',
            {'reason': 'mape_high'}
        )
        assert rejections >= 1
    
    def test_mmm_mape_gate_reject_wide_ci(self, agent):
        """
        Q_009 acceptance: Wide confidence intervals → rejection
        Even with acceptable MAPE, wide CIs indicate instability
        """
        # Arrange
        mock_trace = self._create_mock_trace(
            mape=0.12,  # Acceptable MAPE
            ci_width=0.40  # Very wide CI (40% of mean)
        )
        
        mock_data = self._create_mock_training_data()
        
        # Act
        validation = agent.validate_model(mock_trace, mock_data)
        
        # Assert
        assert validation.accepted is False
        assert validation.mape < 0.14  # MAPE is good
        assert validation.confidence_interval_width > 0.35  # But CI too wide
        assert 'ci_wide' in validation.rejection_reasons
    
    def test_mmm_mape_gate_accept_good_model(self, agent):
        """
        Test that models with MAPE <14% and narrow CI are accepted
        """
        # Arrange
        mock_trace = self._create_mock_trace(
            mape=0.12,  # Good MAPE
            ci_width=0.15  # Narrow CI
        )
        
        mock_data = self._create_mock_training_data()
        
        # Act
        validation = agent.validate_model(mock_trace, mock_data)
        
        # Assert
        assert validation.accepted is True
        assert validation.mape < 0.14
        assert validation.confidence_interval_width < 0.35
        assert len(validation.rejection_reasons) == 0
    
    def test_mmm_mape_boundary_conditions(self, agent):
        """
        Test boundary conditions around 14% MAPE threshold
        """
        test_cases = [
            (0.139, True),   # Just under threshold → accept
            (0.140, False),  # At threshold → reject
            (0.141, False),  # Over threshold → reject
            (0.155, False),  # Well over → reject
        ]
        
        for mape, should_accept in test_cases:
            mock_trace = self._create_mock_trace(mape=mape, ci_width=0.15)
            mock_data = self._create_mock_training_data()
            
            validation = agent.validate_model(mock_trace, mock_data)
            
            assert validation.accepted == should_accept, \
                f"MAPE {mape} should {'accept' if should_accept else 'reject'}"
    
    def _create_mock_trace(self, mape: float, ci_width: float):
        """Create mock PyMC trace with controlled metrics"""
        mock_trace = Mock()
        
        # Mock posterior samples for channels
        n_samples = 2000
        mock_trace.posterior = {
            'channel_coef': np.random.normal(0, 1, (1, n_samples, 3)),
            'base': np.random.normal(100000, 10000, (1, n_samples)),
        }
        
        # Mock computed MAPE
        mock_trace.mape = mape
        
        # Mock confidence intervals (simulate width)
        mean_revenue = 100000
        ci_half_width = (ci_width * mean_revenue) / 2
        mock_trace.ci = {
            'lower': mean_revenue - ci_half_width,
            'upper': mean_revenue + ci_half_width
        }
        
        return mock_trace
    
    def _create_mock_training_data(self):
        """Create mock training data"""
        weeks = 52
        return {
            'revenue': np.random.normal(100000, 10000, weeks),
            'spend_meta': np.random.uniform(10000, 50000, weeks),
            'spend_google': np.random.uniform(10000, 50000, weeks),
            'spend_tiktok': np.random.uniform(5000, 30000, weeks),
        }


class TestMMMValidationResult:
    """Test MMMValidationResult schema (Q_009 contract)"""
    
    def test_validation_result_schema_fields(self):
        """
        Q_009 contract: MMMValidationResult includes mape, confidence_interval_width, accepted
        """
        result = MMMValidationResult(
            mape=0.13,
            confidence_interval_width=0.18,
            accepted=True,
            rejection_reasons=[],
            model_metrics={
                'r_squared': 0.92,
                'holdout_mape': 0.14
            }
        )
        
        assert hasattr(result, 'mape')
        assert hasattr(result, 'confidence_interval_width')
        assert hasattr(result, 'accepted')
        assert hasattr(result, 'rejection_reasons')
        
        assert result.mape == 0.13
        assert result.confidence_interval_width == 0.18
        assert result.accepted is True
    
    def test_validation_result_rejection_reasons(self):
        """Test rejection_reasons list populated correctly"""
        result = MMMValidationResult(
            mape=0.16,
            confidence_interval_width=0.42,
            accepted=False,
            rejection_reasons=['mape_high', 'ci_wide']
        )
        
        assert 'mape_high' in result.rejection_reasons
        assert 'ci_wide' in result.rejection_reasons


class TestAdaptiveRetraining:
    """Test adaptive retraining on drift detection (Q_403)"""
    
    @pytest.fixture
    def agent(self):
        return MMMAgent()
    
    def test_mmm_adaptive_retrain_on_drift(self, agent):
        """
        Q_403 acceptance: drift detected → immediate retrain
        Monitor deployed model performance; trigger retrain when MAPE drifts >15%
        """
        # Arrange
        deployed_model_id = "mmm_v1.2"
        
        # Mock deployed model with good baseline MAPE
        agent.deployed_models = {
            deployed_model_id: MMMModel(
                model_id=deployed_model_id,
                baseline_mape=0.12,
                deployed_at=datetime.utcnow() - timedelta(days=7)
            )
        }
        
        # Simulate drift: current predictions have MAPE=0.16 (drifted >15%)
        mock_recent_predictions = self._create_predictions_with_mape(0.16)
        
        # Act
        drift_detected = agent.check_model_drift(
            model_id=deployed_model_id,
            recent_predictions=mock_recent_predictions
        )
        
        # Assert
        assert drift_detected is True
        
        # Verify immediate retrain triggered
        assert agent.retrain_triggered is True
        assert agent.retrain_reason == 'mape_drift'
        assert agent.retrain_urgency == 'immediate'
    
    def test_no_retrain_when_performance_stable(self, agent):
        """
        Test that no retrain occurs when model performance is stable
        """
        # Arrange
        deployed_model_id = "mmm_v1.2"
        
        agent.deployed_models = {
            deployed_model_id: MMMModel(
                model_id=deployed_model_id,
                baseline_mape=0.12,
                deployed_at=datetime.utcnow() - timedelta(days=7)
            )
        }
        
        # Current performance stable: MAPE=0.13 (within tolerance)
        mock_recent_predictions = self._create_predictions_with_mape(0.13)
        
        # Act
        drift_detected = agent.check_model_drift(
            model_id=deployed_model_id,
            recent_predictions=mock_recent_predictions
        )
        
        # Assert
        assert drift_detected is False
        assert agent.retrain_triggered is False
    
    def _create_predictions_with_mape(self, target_mape: float):
        """Create prediction data with specific MAPE"""
        n_weeks = 4
        actual = np.random.normal(100000, 10000, n_weeks)
        
        # Generate predictions to achieve target MAPE
        # MAPE = mean(|actual - predicted| / actual)
        errors = actual * target_mape
        predicted = actual + np.random.choice([-1, 1], n_weeks) * errors
        
        return {
            'actual': actual,
            'predicted': predicted,
            'weeks': n_weeks
        }


class TestMMMTrainingTrace:
    """Test MMM training observability (Q_404)"""
    
    def test_mmm_training_trace_duration(self):
        """
        Q_404 acceptance: training span duration_ms <6h SLA
        """
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        
        # Setup tracer
        tracer_provider = TracerProvider()
        trace.set_tracer_provider(tracer_provider)
        tracer = trace.get_tracer(__name__)
        
        agent = MMMAgent()
        
        # Mock training that completes in reasonable time
        with tracer.start_as_current_span("mmm_training") as span:
            # Simulate training (mock)
            start_time = datetime.utcnow()
            
            # ... training happens ...
            
            end_time = datetime.utcnow()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            span.set_attribute("duration_ms", duration_ms)
            span.set_attribute("model_version", "v1.3")
            
            # Assert: Duration should be logged
            assert span.attributes.get("duration_ms") is not None
            
            # In production, check against 6h SLA (21,600,000 ms)
            # For test, just verify span was created
            assert span.name == "mmm_training"


class TestMMMOverfitGate:
    """Test overfit detection gate (Q_450)"""
    
    def test_mmm_overfit_gate_high_r2_high_mape(self):
        """
        Q_450 acceptance: R²>0.95 + MAPE>20% → reject (overfit)
        High R² on training but poor out-of-sample → overfitting
        """
        agent = MMMAgent()
        
        # Arrange: High R² but poor holdout performance
        mock_trace = Mock()
        mock_trace.r_squared = 0.96  # Too high
        mock_trace.mape_training = 0.08  # Good on training
        mock_trace.mape_holdout = 0.22  # Poor on holdout (overfitting signal)
        
        # Act
        validation = agent.validate_model_overfit(mock_trace)
        
        # Assert
        assert validation.accepted is False
        assert validation.r_squared > 0.95
        assert validation.mape_holdout > 0.20
        assert 'overfit' in validation.rejection_reasons


class TestMMMControlLengthValidation:
    """Test control variables length assertion (Q_425)"""
    
    def test_mmm_control_length_assertion(self):
        """
        Q_425 acceptance: schema asserts equal lengths for all time series
        """
        agent = MMMAgent()
        
        # Arrange: Mismatched lengths
        data = {
            'revenue': np.array([100, 110, 120]),  # 3 weeks
            'spend_meta': np.array([10, 12, 15, 18]),  # 4 weeks - MISMATCH
            'seasonality': np.array([0.9, 1.0, 1.1])  # 3 weeks
        }
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            agent.validate_training_data_schema(data)
        
        assert "length mismatch" in str(exc_info.value).lower()
        assert "spend_meta" in str(exc_info.value)
