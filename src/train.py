"""
Training pipeline with cross-validation for NBA injury prediction
Replicates R XGBoost training logic with tuned hyperparameters
"""

import xgboost as xgb
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
from .model import InjuryPredictionModel


class InjuryPredictor:
    """
    Training pipeline for injury prediction model
    Uses hyperparameters from R tuning process
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize predictor with configuration
        
        Args:
            config_path: Path to YAML config file (optional)
        """
        self.config = None
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
        # Get params from config or use defaults from R tuning
        if self.config and 'model' in self.config:
            self.params = self.config['model']
        else:
            self.params = {
                'eta': 0.3,                    # Learning rate
                'max_depth': 7,                # Tree depth
                'min_child_weight': 15,        # Min samples per leaf
                'gamma': 0,                    # Min loss reduction for split
                'subsample': 0.9,              # Row sampling
                'colsample_bytree': 0.7,       # Feature sampling
                'objective': 'binary:logistic',
                'eval_metric': ['auc', 'error'],
                'seed': 111111                 # For reproducibility
            }
        
        self.model_wrapper = None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              num_boost_round: int = 70) -> xgb.Booster:
        """
        Train XGBoost model with tuned hyperparameters from R analysis
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            num_boost_round: Number of boosting rounds (from R: 70)
            
        Returns:
            Trained XGBoost Booster model
        """
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        evals = []
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals = [(dtrain, 'train'), (dval, 'val')]
        else:
            evals = [(dtrain, 'train')]
        
        # Train model
        model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=20,
            verbose_eval=10
        )
        
        # Wrap in model class
        self.model_wrapper = InjuryPredictionModel(self.params)
        self.model_wrapper.model = model
        
        return model
    
    def train_with_cv(self, X_train: np.ndarray, y_train: np.ndarray, 
                      n_folds: int = 5, num_boost_round: int = 70) -> Tuple[xgb.Booster, Any]:
        """
        Train with cross-validation (matching R's xgb.cv)
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_folds: Number of CV folds
            num_boost_round: Max boosting rounds
            
        Returns:
            Tuple of (trained model, cv_results)
        """
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        print("Running cross-validation...")
        cv_results = xgb.cv(
            self.params,
            dtrain,
            num_boost_round=num_boost_round,
            nfold=n_folds,
            metrics=['auc', 'error'],
            early_stopping_rounds=20,
            seed=111111,
            verbose_eval=10
        )
        
        best_iteration = len(cv_results)
        print(f"\nBest iteration: {best_iteration}")
        print(f"Best AUC: {cv_results['test-auc-mean'].max():.4f}")
        print(f"Best Error: {cv_results['test-error-mean'].min():.4f}")
        
        # Train final model on all data with best iteration
        print(f"\nTraining final model with {best_iteration} rounds...")
        model = xgb.train(self.params, dtrain, num_boost_round=best_iteration)
        
        # Wrap in model class
        self.model_wrapper = InjuryPredictionModel(self.params)
        self.model_wrapper.model = model
        
        return model, cv_results
    
    def save_model(self, path: str = "models/xgboost_injury.pkl"):
        """Save trained model"""
        if self.model_wrapper is None:
            raise ValueError("No model to save. Train the model first.")
        self.model_wrapper.save(path)
    
    def load_model(self, path: str = "models/xgboost_injury.pkl"):
        """Load trained model"""
        self.model_wrapper = InjuryPredictionModel(self.params)
        self.model_wrapper.load(path)
        return self.model_wrapper.model
    
    def get_model(self):
        """Get the trained model"""
        if self.model_wrapper is None:
            raise ValueError("No model available. Train or load a model first.")
        return self.model_wrapper.model
