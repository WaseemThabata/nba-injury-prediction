"""
Feature engineering and preprocessing for NBA injury prediction
Converts R preprocessing logic to Python
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional


class NBAPreprocessor:
    """Feature engineering and train/test splitting for injury prediction"""
    
    def __init__(self, train_years: List[int], test_years: List[int], 
                 feature_cols: Optional[List[str]] = None):
        self.train_years = train_years
        self.test_years = test_years
        self.feature_cols = feature_cols
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features for injury prediction
        Replicates R code:
        - salary_gp_ratio = round(salary/gp, 2)
        - pay_each_game = salary / 82
        - total_earned = pay_each_game * gp
        - team_salary_lost = salary - salary_gp_ratio
        """
        df = df.copy()
        
        # Salary-based features (from R analysis)
        if 'salary' in df.columns and 'gp' in df.columns:
            # Avoid division by zero
            df['gp_safe'] = df['gp'].replace(0, 1)
            df['salary_gp_ratio'] = (df['salary'] / df['gp_safe']).round(2)
            df['pay_each_game'] = df['salary'] / 82
            df['total_earned'] = df['pay_each_game'] * df['gp']
            df['team_salary_lost'] = df['salary'] - df['salary_gp_ratio']
            df.drop('gp_safe', axis=1, inplace=True)
        
        # Additional injury history features (placeholders if data available)
        # These would need historical injury tracking in the dataset
        # df['days_since_last_injury'] = ...
        # df['career_injury_count'] = ...
        # df['rolling_30day_workload'] = ...
        
        return df
    
    def create_train_test_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data by season (time-based split)
        Matches R approach of using specific year ranges
        """
        train = df[df['season'].isin(self.train_years)].copy()
        test = df[df['season'].isin(self.test_years)].copy()
        
        print(f"Train: {len(train)} samples ({train['season'].min()}-{train['season'].max()})")
        print(f"Test: {len(test)} samples ({test['season'].min()}-{test['season'].max()})")
        
        return train, test
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract feature matrix and labels
        From R code: train_data[,18:43] for features
        """
        if self.feature_cols is None:
            # Auto-detect numeric columns (excluding target and metadata)
            exclude_cols = ['injury', 'name', 'date', 'Date', 'season', 'season_year']
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            self.feature_cols = [col for col in numeric_cols if col not in exclude_cols]
            print(f"Auto-detected {len(self.feature_cols)} feature columns")
        
        # Handle missing values
        X = df[self.feature_cols].fillna(0).values
        
        # Extract labels
        if 'injury' in df.columns:
            y = df['injury'].values
        else:
            print("Warning: 'injury' column not found, using zeros")
            y = np.zeros(len(df))
        
        return X, y
