"""
MMM Agent - Marketing Mix Modeling with Validation

Bayesian MMM with:
- 20% holdout validation
- MAPE < 0.15 quality gate
- Model registry with performance tracking
- Overfitting detection

Related: Q_003, A_003, C02_MMMAgent
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel

from src.agents.mmm.model_validator import ModelValidator, ValidationMetrics


class MMMModel(BaseModel):
    """MMM model result with validation metrics"""
    model_id: str
    model_version: str
    channels: List[str]
    trained_at: datetime
    metrics: Dict[str, float]
    validation: ValidationMetrics
    approved_for_deployment: bool


class CVResults(BaseModel):
    """Cross-validation results"""
    fold_mapes: List[float]
    mean_mape: float
    std_mape: float
    k_folds: int


class MMMAgent:
    """
    Marketing Mix Modeling Agent with validation framework.
    
    Key Features:
    - Train/holdout split (80/20)
    - MAPE validation on holdout
    - Model rejection if MAPE > 0.15
    - Optional k-fold cross-validation
    """
    
    def __init__(self, validator: Optional[ModelValidator] = None):
        """
        Initialize MMM agent.
        
        Args:
            validator: ModelValidator instance (default: MAPE < 0.15)
        """
        self.validator = validator or ModelValidator(max_mape=0.15)
        self._model_registry: Dict[str, MMMModel] = {}
    
    def _split_train_holdout(
        self,
        data: pd.DataFrame,
        holdout_pct: float = 0.20
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and holdout sets (temporal split).
        
        Args:
            data: Time series data sorted by date
            holdout_pct: Percentage for holdout (default 20%)
        
        Returns:
            (train_data, holdout_data)
        """
        # Ensure temporal ordering
        data = data.sort_values('date').reset_index(drop=True)
        
        # Split point
        n = len(data)
        split_idx = int(n * (1 - holdout_pct))
        
        train_data = data.iloc[:split_idx].copy()
        holdout_data = data.iloc[split_idx:].copy()
        
        return train_data, holdout_data
    
    def train_model(
        self,
        data: pd.DataFrame,
        channels: List[str],
        holdout_pct: float = 0.20
    ) -> MMMModel:
        """
        Train MMM model with holdout validation.
        
        Args:
            data: Training data with columns: date, revenue, spend_<channel>, controls
            channels: List of channel names to model
            holdout_pct: Percentage for holdout validation
        
        Returns:
            MMMModel with validation results
        """
        # Split train/holdout
        train_data, holdout_data = self._split_train_holdout(data, holdout_pct)
        
        # Train simplified model (placeholder for Bayesian MMM)
        # In production: use PyMC with adstock + saturation
        revenue_col = 'revenue'
        
        # Simple linear model for demonstration
        X_train = train_data[[f'spend_{ch}' for ch in channels]].values
        y_train = train_data[revenue_col].values
        
        # Fit via least squares
        coefficients = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
        
        # Predictions on train
        y_train_pred = X_train @ coefficients
        
        # Predictions on holdout
        X_holdout = holdout_data[[f'spend_{ch}' for ch in channels]].values
        y_holdout = holdout_data[revenue_col].values
        y_holdout_pred = X_holdout @ coefficients
        
        # Validate model
        validation = self.validator.validate_model(
            y_true=y_holdout,
            y_pred=y_holdout_pred,
            y_train_true=y_train,
            y_train_pred=y_train_pred
        )
        
        # Create model result
        model_id = f"mmm_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        model_version = "v1.0_simplified"
        
        model = MMMModel(
            model_id=model_id,
            model_version=model_version,
            channels=channels,
            trained_at=datetime.utcnow(),
            metrics={
                'mape_holdout': validation.mape,
                'mape_train': validation.mape_train or 0.0,
                'r_squared': validation.r_squared,
                'rmse': validation.rmse
            },
            validation=validation,
            approved_for_deployment=validation.approved
        )
        
        # Store in registry
        self._model_registry[model_id] = model
        
        return model
    
    def cross_validate(
        self,
        data: pd.DataFrame,
        channels: List[str],
        k_folds: int = 5
    ) -> CVResults:
        """
        K-fold cross-validation for robust performance estimation.
        
        Args:
            data: Training data
            channels: Channel names
            k_folds: Number of folds (default 5)
        
        Returns:
            CVResults with fold-wise MAPE scores
        """
        data = data.sort_values('date').reset_index(drop=True)
        n = len(data)
        fold_size = n // k_folds
        
        fold_mapes = []
        
        for i in range(k_folds):
            # Define validation fold
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i < k_folds - 1 else n
            
            # Split data
            val_data = data.iloc[val_start:val_end]
            train_data = pd.concat([
                data.iloc[:val_start],
                data.iloc[val_end:]
            ])
            
            # Train on fold
            X_train = train_data[[f'spend_{ch}' for ch in channels]].values
            y_train = train_data['revenue'].values
            
            coefficients = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
            
            # Validate on fold
            X_val = val_data[[f'spend_{ch}' for ch in channels]].values
            y_val = val_data['revenue'].values
            y_val_pred = X_val @ coefficients
            
            fold_mape = self.validator.calculate_mape(y_val, y_val_pred)
            fold_mapes.append(fold_mape)
        
        return CVResults(
            fold_mapes=fold_mapes,
            mean_mape=float(np.mean(fold_mapes)),
            std_mape=float(np.std(fold_mapes)),
            k_folds=k_folds
        )
    
    def get_model(self, model_id: str) -> Optional[MMMModel]:
        """Retrieve model from registry"""
        return self._model_registry.get(model_id)
    
    def list_approved_models(self) -> List[MMMModel]:
        """List all models approved for deployment"""
        return [
            model for model in self._model_registry.values()
            if model.approved_for_deployment
        ]
