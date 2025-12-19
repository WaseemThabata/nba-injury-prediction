"""
Generate results and visualizations from trained model

Usage:
    python scripts/generate_results.py
"""

import sys
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from evaluate import ModelEvaluator
from explainability import SHAPAnalyzer


def main():
    """Generate results from saved model"""
    
    model_path = 'models/xgboost_injury.pkl'
    
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print("Please run training first: python scripts/run_pipeline.py")
        return
    
    print("Loading trained model...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print("Model loaded successfully!")
    print("\nTo generate new results, load test data and call:")
    print("  evaluator = ModelEvaluator(model, threshold=0.2)")
    print("  evaluator.evaluate(X_test, y_test)")
    print("\n  shap_analyzer = SHAPAnalyzer(model, X_train)")
    print("  shap_analyzer.analyze(X_test)")
    
    # Check if results already exist
    results_dir = Path('results')
    if results_dir.exists():
        print("\nExisting results found:")
        for file in results_dir.glob('*'):
            print(f"  - {file}")


if __name__ == "__main__":
    main()
