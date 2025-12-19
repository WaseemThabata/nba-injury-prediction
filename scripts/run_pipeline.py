"""
End-to-end NBA injury prediction pipeline (Python version of R analysis)
Achieves balanced accuracy of ~60% matching R results

Usage:
    python scripts/run_pipeline.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import NBADataLoader
from preprocessing import NBAPreprocessor
from train import InjuryPredictor
from evaluate import ModelEvaluator
from explainability import SHAPAnalyzer
import yaml
import numpy as np


def main():
    """Run complete injury prediction pipeline"""
    
    # Load config
    config_path = 'config/config.yaml'
    if not Path(config_path).exists():
        print(f"Warning: {config_path} not found, using default parameters")
        config = None
    else:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    print("=" * 60)
    print("NBA Injury Prediction Pipeline")
    print("=" * 60)
    print()
    
    # 1. Load data
    print("Step 1: Loading data...")
    print("-" * 60)
    
    data_dir = config['paths']['raw_data'] if config else 'data/raw'
    loader = NBADataLoader(data_dir)
    df = loader.merge_datasets()
    
    if df.empty:
        print("\n" + "!" * 60)
        print("ERROR: No data loaded. Please ensure data files exist:")
        print("  - data/raw/injuries_2010_2020.xlsx")
        print("  - data/raw/all_seasons.xlsx")
        print("  - data/raw/NBA_Salaries.xlsx")
        print("!" * 60)
        print("\nPipeline stopped. Add data files and try again.")
        return
    
    print(f"✓ Loaded {len(df)} player-game records")
    print()
    
    # 2. Preprocess
    print("Step 2: Feature engineering...")
    print("-" * 60)
    
    train_years = config['data']['train_years'] if config else list(range(2010, 2019))
    test_years = config['data']['test_years'] if config else [2019, 2020]
    
    preprocessor = NBAPreprocessor(
        train_years=train_years,
        test_years=test_years
    )
    
    df = preprocessor.engineer_features(df)
    train_df, test_df = preprocessor.create_train_test_split(df)
    
    X_train, y_train = preprocessor.prepare_features(train_df)
    X_test, y_test = preprocessor.prepare_features(test_df)
    
    print(f"✓ Train: {len(X_train)} samples, {X_train.shape[1]} features")
    print(f"✓ Test: {len(X_test)} samples")
    print()
    
    # 3. Train model
    print("Step 3: Training XGBoost model...")
    print("-" * 60)
    print("Using hyperparameters from R tuning process:")
    
    predictor = InjuryPredictor(config_path if Path(config_path).exists() else None)
    print(f"  eta={predictor.params['eta']}, max_depth={predictor.params['max_depth']}")
    print(f"  min_child_weight={predictor.params['min_child_weight']}, gamma={predictor.params['gamma']}")
    print(f"  subsample={predictor.params['subsample']}, colsample_bytree={predictor.params['colsample_bytree']}")
    print()
    
    model, cv_results = predictor.train_with_cv(X_train, y_train, n_folds=5)
    predictor.save_model()
    print()
    
    # 4. Evaluate
    print("Step 4: Evaluating model...")
    print("-" * 60)
    
    threshold = config['evaluation']['threshold'] if config else 0.2
    evaluator = ModelEvaluator(model, threshold=threshold)
    metrics = evaluator.evaluate(X_test, y_test, feature_names=preprocessor.feature_cols)
    print()
    
    # 5. SHAP analysis
    print("Step 5: SHAP explainability analysis...")
    print("-" * 60)
    
    shap_analyzer = SHAPAnalyzer(model, X_train, 
                                 feature_names=preprocessor.feature_cols)
    shap_analyzer.analyze(X_test)
    print()
    
    # Summary
    print("=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"\n✓ Model saved to: models/xgboost_injury.pkl")
    print(f"✓ Results saved to: results/")
    print(f"\nPerformance Metrics:")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"\n(R baseline: ~0.6009 balanced accuracy)")
    print()


if __name__ == "__main__":
    main()
