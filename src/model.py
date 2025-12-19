"""
XGBoost model wrapper for NBA injury prediction
Simple wrapper around XGBoost for consistent interface
"""

import xgboost as xgb
import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Dict, Any


class InjuryPredictionModel:
    """
    XGBoost wrapper for injury prediction
    Provides clean interface for training, prediction, and model persistence
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize model with XGBoost parameters
        
        Args:
            params: XGBoost parameters dict (if None, uses defaults from R tuning)
        """
        if params is None:
            # Default hyperparameters from R tuning process
            params = {
                'eta': 0.3,
                'max_depth': 7,
                'min_child_weight': 15,
                'gamma': 0,
                'subsample': 0.9,
                'colsample_bytree': 0.7,
                'objective': 'binary:logistic',
                'eval_metric': ['auc', 'error'],
                'seed': 111111
            }
        self.params = params
        self.model = None
    
    def get_model(self):
        """Get the underlying XGBoost model"""
        return self.model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict injury probabilities
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of probabilities [0, 1]
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        dmatrix = xgb.DMatrix(X)
        return self.model.predict(dmatrix)
    
    def predict_class(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict injury classes (0 or 1)
        
        Args:
            X: Feature matrix
            threshold: Classification threshold
            
        Returns:
            Array of binary predictions
        """
        probas = self.predict(X)
        return (probas >= threshold).astype(int)
    
    def save(self, path: str):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load trained model"""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {path}")
        return self
