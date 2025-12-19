"""
Unit tests for model module
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from model import InjuryPredictionModel


class TestInjuryPredictionModel:
    """Test cases for InjuryPredictionModel class"""
    
    def test_init_default_params(self):
        """Test model initialization with default parameters"""
        model = InjuryPredictionModel()
        
        assert model.params is not None
        assert model.params['eta'] == 0.3
        assert model.params['max_depth'] == 7
        assert model.params['objective'] == 'binary:logistic'
        assert model.model is None
    
    def test_init_custom_params(self):
        """Test model initialization with custom parameters"""
        custom_params = {
            'eta': 0.1,
            'max_depth': 5,
            'objective': 'binary:logistic'
        }
        
        model = InjuryPredictionModel(custom_params)
        
        assert model.params['eta'] == 0.1
        assert model.params['max_depth'] == 5
    
    def test_predict_without_training(self):
        """Test that predict raises error when model not trained"""
        model = InjuryPredictionModel()
        X = np.random.randn(10, 5)
        
        with pytest.raises(ValueError, match="Model not trained"):
            model.predict(X)
    
    def test_predict_class_without_training(self):
        """Test that predict_class raises error when model not trained"""
        model = InjuryPredictionModel()
        X = np.random.randn(10, 5)
        
        with pytest.raises(ValueError, match="Model not trained"):
            model.predict_class(X)
    
    def test_save_without_training(self):
        """Test that save raises error when model not trained"""
        model = InjuryPredictionModel()
        
        with tempfile.NamedTemporaryFile(suffix='.pkl') as tmp:
            with pytest.raises(ValueError, match="No model to save"):
                model.save(tmp.name)
    
    def test_save_and_load(self, sample_features):
        """Test model save and load functionality"""
        import xgboost as xgb
        
        X, y = sample_features
        
        # Train a simple model
        model = InjuryPredictionModel()
        dtrain = xgb.DMatrix(X, label=y)
        model.model = xgb.train(model.params, dtrain, num_boost_round=5)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            model.save(tmp_path)
            
            # Load model
            model2 = InjuryPredictionModel()
            model2.load(tmp_path)
            
            # Check predictions match
            pred1 = model.predict(X)
            pred2 = model2.predict(X)
            
            np.testing.assert_array_almost_equal(pred1, pred2)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_predict_class_threshold(self, sample_features):
        """Test predict_class with different thresholds"""
        import xgboost as xgb
        
        X, y = sample_features
        
        # Train a simple model
        model = InjuryPredictionModel()
        dtrain = xgb.DMatrix(X, label=y)
        model.model = xgb.train(model.params, dtrain, num_boost_round=5)
        
        # Test different thresholds
        preds_05 = model.predict_class(X, threshold=0.5)
        preds_02 = model.predict_class(X, threshold=0.2)
        
        # Lower threshold should result in more positive predictions
        assert preds_02.sum() >= preds_05.sum()
        
        # Check binary output
        assert set(np.unique(preds_05)).issubset({0, 1})
        assert set(np.unique(preds_02)).issubset({0, 1})
