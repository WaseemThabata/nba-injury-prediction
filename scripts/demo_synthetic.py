"""
Demo script using synthetic data to show pipeline functionality
Use this to test the pipeline without real data files

Usage:
    python scripts/demo_synthetic.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from preprocessing import NBAPreprocessor
from train import InjuryPredictor
from evaluate import ModelEvaluator
from explainability import SHAPAnalyzer


def generate_synthetic_data(n_samples=1000, n_features=26, random_state=111111):
    """
    Generate synthetic NBA-like data for demonstration
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        random_state: Random seed
        
    Returns:
        DataFrame with synthetic data
    """
    np.random.seed(random_state)
    
    # Generate features
    data = {
        'season': np.random.choice(range(2010, 2021), n_samples),
        'name': [f'Player_{i}' for i in range(n_samples)],
        'age': np.random.randint(19, 40, n_samples),
        'gp': np.random.randint(1, 82, n_samples),
        'minutes': np.random.uniform(5, 40, n_samples),
        'usage_rate': np.random.uniform(10, 35, n_samples),
        'salary': np.random.uniform(500000, 40000000, n_samples),
    }
    
    # Add more features
    for i in range(n_features - 7):
        data[f'feature_{i}'] = np.random.randn(n_samples)
    
    df = pd.DataFrame(data)
    
    # Engineer features
    df['salary_gp_ratio'] = df['salary'] / df['gp'].replace(0, 1)
    df['pay_each_game'] = df['salary'] / 82
    df['total_earned'] = df['pay_each_game'] * df['gp']
    df['team_salary_lost'] = df['salary'] - df['salary_gp_ratio']
    
    # Generate injury labels (with some correlation to age and minutes)
    injury_prob = 0.15 + 0.01 * (df['age'] - 27) + 0.005 * (df['minutes'] - 20)
    injury_prob = np.clip(injury_prob, 0, 1)
    df['injury'] = (np.random.random(n_samples) < injury_prob).astype(int)
    
    return df


def main():
    """Run demo with synthetic data"""
    
    print("=" * 60)
    print("NBA Injury Prediction Demo (Synthetic Data)")
    print("=" * 60)
    print()
    
    # Generate synthetic data
    print("Step 1: Generating synthetic data...")
    print("-" * 60)
    df = generate_synthetic_data(n_samples=2000, n_features=26)
    print(f"✓ Generated {len(df)} synthetic player records")
    print(f"✓ Injury rate: {df['injury'].mean():.1%}")
    print()
    
    # Split data
    print("Step 2: Splitting train/test data...")
    print("-" * 60)
    
    train_years = list(range(2010, 2019))
    test_years = [2019, 2020]
    
    preprocessor = NBAPreprocessor(
        train_years=train_years,
        test_years=test_years
    )
    
    train_df, test_df = preprocessor.create_train_test_split(df)
    X_train, y_train = preprocessor.prepare_features(train_df)
    X_test, y_test = preprocessor.prepare_features(test_df)
    
    print(f"✓ Train: {len(X_train)} samples, {X_train.shape[1]} features")
    print(f"✓ Test: {len(X_test)} samples")
    print()
    
    # Train model
    print("Step 3: Training XGBoost model...")
    print("-" * 60)
    
    predictor = InjuryPredictor()
    model, cv_results = predictor.train_with_cv(X_train, y_train, n_folds=3)
    
    # Save model
    Path('models').mkdir(exist_ok=True)
    predictor.save_model('models/xgboost_injury_demo.pkl')
    print()
    
    # Evaluate
    print("Step 4: Evaluating model...")
    print("-" * 60)
    
    evaluator = ModelEvaluator(model, threshold=0.2)
    metrics = evaluator.evaluate(X_test, y_test, feature_names=preprocessor.feature_cols)
    print()
    
    # SHAP analysis
    print("Step 5: SHAP explainability analysis...")
    print("-" * 60)
    
    # Use subset for faster SHAP computation
    X_train_sample = X_train[:500] if len(X_train) > 500 else X_train
    X_test_sample = X_test[:200] if len(X_test) > 200 else X_test
    
    shap_analyzer = SHAPAnalyzer(model, X_train_sample, 
                                 feature_names=preprocessor.feature_cols)
    shap_analyzer.analyze(X_test_sample, save_path='results/shap_summary_demo.png')
    print()
    
    # Summary
    print("=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print(f"\n✓ Model saved to: models/xgboost_injury_demo.pkl")
    print(f"✓ Results saved to: results/")
    print(f"\nPerformance on Synthetic Data:")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print()
    print("NOTE: This used synthetic data for demonstration.")
    print("Run 'python scripts/run_pipeline.py' with real data for actual results.")
    print()


if __name__ == "__main__":
    main()
