"""
Unit tests for preprocessing module
"""

import pytest
import numpy as np
import pandas as pd
from preprocessing import NBAPreprocessor


class TestNBAPreprocessor:
    """Test cases for NBAPreprocessor class"""
    
    def test_init(self):
        """Test preprocessor initialization"""
        train_years = [2015, 2016, 2017]
        test_years = [2018, 2019]
        
        preprocessor = NBAPreprocessor(train_years, test_years)
        
        assert preprocessor.train_years == train_years
        assert preprocessor.test_years == test_years
        assert preprocessor.feature_cols is None
    
    def test_engineer_features(self, sample_injury_data):
        """Test feature engineering"""
        preprocessor = NBAPreprocessor([2015], [2019])
        df = preprocessor.engineer_features(sample_injury_data)
        
        # Check that new features are created
        assert 'salary_gp_ratio' in df.columns
        assert 'pay_each_game' in df.columns
        assert 'total_earned' in df.columns
        assert 'team_salary_lost' in df.columns
        
        # Check no NaN values in new features
        assert not df['salary_gp_ratio'].isna().any()
        assert not df['pay_each_game'].isna().any()
    
    def test_engineer_features_zero_games(self):
        """Test feature engineering handles zero games played"""
        data = pd.DataFrame({
            'name': ['Player1'],
            'season': [2019],
            'salary': [1000000],
            'gp': [0]
        })
        
        preprocessor = NBAPreprocessor([2015], [2019])
        df = preprocessor.engineer_features(data)
        
        # Should not have division by zero errors
        assert not df['salary_gp_ratio'].isna().any()
    
    def test_create_train_test_split(self, sample_injury_data):
        """Test train/test split"""
        train_years = [2015, 2016, 2017]
        test_years = [2018, 2019]
        
        preprocessor = NBAPreprocessor(train_years, test_years)
        train_df, test_df = preprocessor.create_train_test_split(sample_injury_data)
        
        # Check split is correct
        assert all(train_df['season'].isin(train_years))
        assert all(test_df['season'].isin(test_years))
        
        # Check no overlap
        assert len(set(train_df.index) & set(test_df.index)) == 0
    
    def test_prepare_features(self, sample_injury_data):
        """Test feature preparation"""
        preprocessor = NBAPreprocessor([2015], [2019])
        df = preprocessor.engineer_features(sample_injury_data)
        
        X, y = preprocessor.prepare_features(df)
        
        # Check shapes
        assert X.shape[0] == len(df)
        assert y.shape[0] == len(df)
        
        # Check no NaN in features
        assert not np.isnan(X).any()
        
        # Check labels are binary
        assert set(np.unique(y)).issubset({0, 1})
    
    def test_prepare_features_with_custom_columns(self, sample_injury_data):
        """Test feature preparation with custom feature columns"""
        feature_cols = ['age', 'gp', 'salary']
        preprocessor = NBAPreprocessor([2015], [2019], feature_cols=feature_cols)
        df = preprocessor.engineer_features(sample_injury_data)
        
        X, y = preprocessor.prepare_features(df)
        
        # Check correct number of features
        assert X.shape[1] == len(feature_cols)
    
    def test_prepare_features_handles_missing_values(self):
        """Test that missing values are filled"""
        data = pd.DataFrame({
            'name': ['Player1', 'Player2'],
            'season': [2019, 2019],
            'injury': [0, 1],
            'age': [25, np.nan],
            'gp': [50, 60],
            'salary': [1000000, 2000000]
        })
        
        preprocessor = NBAPreprocessor([2019], [2020], feature_cols=['age', 'gp'])
        X, y = preprocessor.prepare_features(data)
        
        # Check no NaN values
        assert not np.isnan(X).any()
