"""
Pytest configuration and fixtures for NBA injury prediction tests
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


@pytest.fixture
def sample_injury_data():
    """Generate sample injury dataset for testing"""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'name': [f'Player_{i}' for i in range(n_samples)],
        'season': np.random.choice(range(2015, 2021), n_samples),
        'Date': pd.date_range('2015-01-01', periods=n_samples, freq='D'),
        'injury': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'age': np.random.randint(20, 38, n_samples),
        'gp': np.random.randint(10, 82, n_samples),
        'salary': np.random.uniform(1e6, 4e7, n_samples),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_features():
    """Generate sample feature matrix for testing"""
    np.random.seed(42)
    n_samples = 100
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    
    return X, y


@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        'model': {
            'eta': 0.3,
            'max_depth': 7,
            'min_child_weight': 15,
            'gamma': 0,
            'subsample': 0.9,
            'colsample_bytree': 0.7,
            'num_boost_round': 10,  # Reduced for faster tests
            'objective': 'binary:logistic',
            'eval_metric': ['auc', 'error'],
            'seed': 42
        },
        'data': {
            'train_years': [2015, 2016, 2017],
            'test_years': [2018, 2019]
        },
        'evaluation': {
            'threshold': 0.2,
            'cv_folds': 3  # Reduced for faster tests
        }
    }


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory structure"""
    data_dir = tmp_path / "data"
    (data_dir / "raw").mkdir(parents=True)
    (data_dir / "processed").mkdir(parents=True)
    return data_dir


@pytest.fixture
def temp_models_dir(tmp_path):
    """Create temporary models directory"""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)
    return models_dir


@pytest.fixture
def temp_results_dir(tmp_path):
    """Create temporary results directory"""
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)
    return results_dir
