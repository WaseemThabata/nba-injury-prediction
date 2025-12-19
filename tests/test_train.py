"""
Unit tests for training module
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from train import InjuryPredictor


class TestInjuryPredictor:
    """Test cases for InjuryPredictor class"""
    
    def test_init_without_config(self):
        """Test predictor initialization without config file"""
        predictor = InjuryPredictor()
        
        assert predictor.params is not None
        assert predictor.params['eta'] == 0.3
        assert predictor.params['max_depth'] == 7
        assert predictor.model_wrapper is None
    
    def test_init_with_nonexistent_config(self):
        """Test predictor initialization with non-existent config"""
        predictor = InjuryPredictor('nonexistent.yaml')
        
        # Should fall back to default params
        assert predictor.params is not None
    
    def test_train(self, sample_features):
        """Test basic training"""
        X, y = sample_features
        
        predictor = InjuryPredictor()
        model = predictor.train(X, y, num_boost_round=5)
        
        assert model is not None
        assert predictor.model_wrapper is not None
    
    def test_train_with_validation(self, sample_features):
        """Test training with validation set"""
        X, y = sample_features
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]
        
        predictor = InjuryPredictor()
        model = predictor.train(X_train, y_train, X_val, y_val, num_boost_round=5)
        
        assert model is not None
    
    def test_train_with_cv(self, sample_features):
        """Test training with cross-validation"""
        X, y = sample_features
        
        predictor = InjuryPredictor()
        model, cv_results = predictor.train_with_cv(X, y, n_folds=3, num_boost_round=5)
        
        assert model is not None
        assert cv_results is not None
        assert len(cv_results) > 0
    
    def test_save_load_model(self, sample_features):
        """Test model save and load"""
        X, y = sample_features
        
        predictor = InjuryPredictor()
        predictor.train(X, y, num_boost_round=5)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            predictor.save_model(tmp_path)
            
            # Load in new predictor
            predictor2 = InjuryPredictor()
            predictor2.load_model(tmp_path)
            
            assert predictor2.model_wrapper is not None
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_get_model_without_training(self):
        """Test get_model raises error when not trained"""
        predictor = InjuryPredictor()
        
        with pytest.raises(ValueError, match="No model available"):
            predictor.get_model()
    
    def test_save_model_without_training(self):
        """Test save_model raises error when not trained"""
        predictor = InjuryPredictor()
        
        with pytest.raises(ValueError, match="No model to save"):
            predictor.save_model('test.pkl')
