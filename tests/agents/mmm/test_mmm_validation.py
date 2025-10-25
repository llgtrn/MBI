"""
MMM Validation Tests - Model Quality Assurance

Tests for holdout validation, MAPE tracking, and model rejection criteria.
Related: Q_003, A_003
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date, timedelta
from unittest.mock import Mock, patch

from src.agents.mmm.mmm_agent import MMMAgent
from src.agents.mmm.model_validator import ModelValidator, ValidationMetrics


class TestMMMValidation:
    """Test MMM model validation framework"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample training data"""
        dates = pd.date_range(start='2024-01-01', periods=52, freq='W')
        
        data = pd.DataFrame({
            'date': dates,
            'revenue': np.random.normal(100000, 20000, 52),
            'spend_meta': np.random.normal(30000, 5000, 52),
            'spend_google': np.random.normal(25000, 4000, 52),
            'spend_tiktok': np.random.normal(15000, 3000, 52),
            'seasonality': np.sin(np.arange(52) * 2 * np.pi / 52),
            'price': np.random.normal(100, 5, 52)
        })
        
        return data
    
    @pytest.fixture
    def mmm_agent(self):
        return MMMAgent()
    
    @pytest.fixture
    def model_validator(self):
        return ModelValidator(max_mape=0.15)
    
    def test_holdout_set_creation(self, mmm_agent, sample_data):
        """ACCEPTANCE: 20% holdout set created for validation"""
        train_data, holdout_data = mmm_agent._split_train_holdout(
            sample_data,
            holdout_pct=0.20
        )
        
        # Verify split ratio
        total_rows = len(sample_data)
        holdout_rows = len(holdout_data)
        train_rows = len(train_data)
        
        assert train_rows + holdout_rows == total_rows
        assert abs(holdout_rows / total_rows - 0.20) < 0.05
        
        # Verify temporal ordering (holdout is most recent)
        assert holdout_data['date'].min() > train_data['date'].max()
    
    def test_mape_calculation_on_holdout(self, model_validator):
        """ACCEPTANCE: MAPE computed correctly on validation set"""
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 320, 380, 510])
        
        mape = model_validator.calculate_mape(y_true, y_pred)
        
        # Manual calculation
        expected_mape = np.mean(np.abs((y_true - y_pred) / y_true))
        
        assert abs(mape - expected_mape) < 0.001
        assert 0 <= mape <= 1
    
    def test_model_rejection_if_mape_exceeds_threshold(self, model_validator):
        """ACCEPTANCE: Reject models with MAPE > 0.15"""
        # Good model (MAPE < 0.15)
        y_true_good = np.array([100, 200, 300, 400, 500])
        y_pred_good = np.array([105, 195, 310, 395, 505])
        mape_good = model_validator.calculate_mape(y_true_good, y_pred_good)
        
        validation_good = model_validator.validate_model(
            y_true=y_true_good,
            y_pred=y_pred_good
        )
        
        assert mape_good < 0.15
        assert validation_good.approved is True
        
        # Bad model (MAPE > 0.15)
        y_true_bad = np.array([100, 200, 300, 400, 500])
        y_pred_bad = np.array([150, 150, 400, 300, 600])
        mape_bad = model_validator.calculate_mape(y_true_bad, y_pred_bad)
        
        validation_bad = model_validator.validate_model(
            y_true=y_true_bad,
            y_pred=y_pred_bad
        )
        
        assert mape_bad > 0.15
        assert validation_bad.approved is False
        assert 'MAPE exceeds threshold' in validation_bad.rejection_reason
    
    def test_model_registry_stores_performance_metrics(self, mmm_agent, sample_data):
        """ACCEPTANCE: Model registry tracks MAPE, R², and holdout performance"""
        # Train model
        model_result = mmm_agent.train_model(
            data=sample_data,
            channels=['meta', 'google', 'tiktok']
        )
        
        # Verify model registry entry
        assert model_result.model_id is not None
        assert model_result.metrics is not None
        
        metrics = model_result.metrics
        assert 'mape_holdout' in metrics
        assert 'r_squared' in metrics
        assert 'rmse' in metrics
        
        # MAPE should be computed
        assert 0 <= metrics['mape_holdout'] <= 1
    
    def test_cross_validation_k_fold(self, mmm_agent, sample_data):
        """ACCEPTANCE: Optional k-fold CV for robust validation"""
        cv_results = mmm_agent.cross_validate(
            data=sample_data,
            channels=['meta', 'google', 'tiktok'],
            k_folds=5
        )
        
        # Verify k-fold results
        assert len(cv_results.fold_mapes) == 5
        assert all(0 <= mape <= 1 for mape in cv_results.fold_mapes)
        
        # Mean MAPE
        mean_mape = np.mean(cv_results.fold_mapes)
        std_mape = np.std(cv_results.fold_mapes)
        
        assert cv_results.mean_mape == mean_mape
        assert cv_results.std_mape == std_mape
    
    def test_overfitting_detection(self, model_validator):
        """ACCEPTANCE: Detect overfitting via train vs. validation MAPE gap"""
        train_mape = 0.05  # Very low on training
        validation_mape = 0.25  # High on validation → overfitting
        
        is_overfit = model_validator.detect_overfitting(
            train_mape=train_mape,
            validation_mape=validation_mape,
            max_gap=0.10
        )
        
        assert is_overfit is True
        
        # Not overfit
        is_overfit_ok = model_validator.detect_overfitting(
            train_mape=0.08,
            validation_mape=0.12,
            max_gap=0.10
        )
        
        assert is_overfit_ok is False
    
    def test_validation_metrics_schema(self):
        """ACCEPTANCE: ValidationMetrics schema defined with required fields"""
        metrics = ValidationMetrics(
            mape=0.12,
            rmse=5000.0,
            r_squared=0.85,
            mape_train=0.10,
            mape_validation=0.12,
            approved=True,
            rejection_reason=None
        )
        
        assert metrics.mape == 0.12
        assert metrics.approved is True
        assert metrics.rejection_reason is None
    
    def test_model_version_tracking(self, mmm_agent):
        """ACCEPTANCE: Each model has version and training timestamp"""
        model_result = mmm_agent.train_model(
            data=pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=52, freq='W'),
                'revenue': np.random.normal(100000, 10000, 52),
                'spend_meta': np.random.normal(30000, 5000, 52),
                'spend_google': np.random.normal(25000, 4000, 52)
            }),
            channels=['meta', 'google']
        )
        
        assert model_result.model_version is not None
        assert model_result.trained_at is not None
        assert model_result.channels == ['meta', 'google']


class TestMMMModelValidator:
    """Test ModelValidator class"""
    
    def test_validator_initialization(self):
        """ACCEPTANCE: Validator configured with MAPE threshold"""
        validator = ModelValidator(max_mape=0.15)
        
        assert validator.max_mape == 0.15
        assert validator.max_train_validation_gap == 0.10
    
    def test_validation_failure_logs_reason(self):
        """ACCEPTANCE: Rejection reason logged for debugging"""
        validator = ModelValidator(max_mape=0.15)
        
        y_true = np.array([100, 200, 300])
        y_pred = np.array([200, 100, 400])  # Poor predictions
        
        result = validator.validate_model(y_true, y_pred)
        
        assert result.approved is False
        assert result.rejection_reason is not None
        assert 'MAPE' in result.rejection_reason or 'threshold' in result.rejection_reason
