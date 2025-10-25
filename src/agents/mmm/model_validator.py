"""
Model Validator for MMM - Quality Assurance

Validates MMM models before deployment:
- MAPE < 0.15 threshold
- Overfitting detection
- Performance metrics tracking

Related: Q_003, A_003, C02_MMMAgent
"""

import numpy as np
from pydantic import BaseModel, Field
from typing import Optional


class ValidationMetrics(BaseModel):
    """Model validation metrics schema"""
    
    mape: float = Field(
        description="Mean Absolute Percentage Error on validation set",
        ge=0.0,
        le=1.0
    )
    
    rmse: float = Field(
        description="Root Mean Squared Error",
        ge=0.0
    )
    
    r_squared: float = Field(
        description="R² coefficient of determination",
        ge=0.0,
        le=1.0
    )
    
    mape_train: Optional[float] = Field(
        default=None,
        description="MAPE on training set (for overfitting detection)"
    )
    
    mape_validation: Optional[float] = Field(
        default=None,
        description="MAPE on validation/holdout set"
    )
    
    approved: bool = Field(
        description="Whether model passes validation criteria"
    )
    
    rejection_reason: Optional[str] = Field(
        default=None,
        description="Reason for rejection if not approved"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "mape": 0.12,
                "rmse": 5000.0,
                "r_squared": 0.85,
                "mape_train": 0.10,
                "mape_validation": 0.12,
                "approved": True,
                "rejection_reason": None
            }
        }


class ModelValidator:
    """
    Validates MMM models before deployment.
    
    Rejection Criteria:
    - MAPE > max_mape (default 0.15)
    - Train-validation MAPE gap > max_train_validation_gap (overfitting)
    - Negative R² (model worse than mean baseline)
    """
    
    def __init__(
        self,
        max_mape: float = 0.15,
        max_train_validation_gap: float = 0.10,
        min_r_squared: float = 0.5
    ):
        """
        Initialize validator with quality thresholds.
        
        Args:
            max_mape: Maximum acceptable MAPE (default 0.15 = 15%)
            max_train_validation_gap: Max difference between train and validation MAPE
            min_r_squared: Minimum acceptable R²
        """
        self.max_mape = max_mape
        self.max_train_validation_gap = max_train_validation_gap
        self.min_r_squared = min_r_squared
    
    def calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error.
        
        MAPE = mean(|y_true - y_pred| / |y_true|)
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
        
        Returns:
            MAPE score (0 to 1, lower is better)
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Avoid division by zero
        mask = y_true != 0
        
        if not mask.any():
            return 1.0  # Worst case
        
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
        
        return float(mape)
    
    def calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Root Mean Squared Error.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
        
        Returns:
            RMSE (same units as y_true)
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        
        return float(rmse)
    
    def calculate_r_squared(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate R² (coefficient of determination).
        
        R² = 1 - (SS_res / SS_tot)
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
        
        Returns:
            R² score (1.0 is perfect, 0 is baseline, negative is worse than baseline)
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        r_squared = 1 - (ss_res / ss_tot)
        
        return float(r_squared)
    
    def detect_overfitting(
        self,
        train_mape: float,
        validation_mape: float,
        max_gap: Optional[float] = None
    ) -> bool:
        """
        Detect overfitting by comparing train vs. validation MAPE.
        
        Args:
            train_mape: MAPE on training set
            validation_mape: MAPE on validation set
            max_gap: Maximum acceptable gap (default: self.max_train_validation_gap)
        
        Returns:
            True if overfitting detected
        """
        if max_gap is None:
            max_gap = self.max_train_validation_gap
        
        gap = validation_mape - train_mape
        
        return gap > max_gap
    
    def validate_model(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train_true: Optional[np.ndarray] = None,
        y_train_pred: Optional[np.ndarray] = None
    ) -> ValidationMetrics:
        """
        Validate model predictions against ground truth.
        
        Args:
            y_true: Validation set ground truth
            y_pred: Validation set predictions
            y_train_true: Optional training set ground truth (for overfitting check)
            y_train_pred: Optional training set predictions
        
        Returns:
            ValidationMetrics with approval decision
        """
        # Calculate metrics
        mape = self.calculate_mape(y_true, y_pred)
        rmse = self.calculate_rmse(y_true, y_pred)
        r_squared = self.calculate_r_squared(y_true, y_pred)
        
        # Train metrics if provided
        mape_train = None
        if y_train_true is not None and y_train_pred is not None:
            mape_train = self.calculate_mape(y_train_true, y_train_pred)
        
        # Validation criteria
        approved = True
        rejection_reason = None
        
        # Check MAPE threshold
        if mape > self.max_mape:
            approved = False
            rejection_reason = f"MAPE ({mape:.3f}) exceeds threshold ({self.max_mape})"
        
        # Check R² minimum
        elif r_squared < self.min_r_squared:
            approved = False
            rejection_reason = f"R² ({r_squared:.3f}) below minimum ({self.min_r_squared})"
        
        # Check overfitting
        elif mape_train is not None and self.detect_overfitting(mape_train, mape):
            approved = False
            gap = mape - mape_train
            rejection_reason = f"Overfitting detected: train MAPE {mape_train:.3f}, validation MAPE {mape:.3f} (gap {gap:.3f})"
        
        return ValidationMetrics(
            mape=mape,
            rmse=rmse,
            r_squared=r_squared,
            mape_train=mape_train,
            mape_validation=mape,
            approved=approved,
            rejection_reason=rejection_reason
        )
