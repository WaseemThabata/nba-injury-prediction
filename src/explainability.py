"""
SHAP explainability analysis for NBA injury prediction
Replicates R SHAP analysis in Python
"""

import shap
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
from pathlib import Path
from typing import Optional, List


class SHAPAnalyzer:
    """
    SHAP (SHapley Additive exPlanations) analysis for model interpretability
    Provides feature importance and interaction analysis
    """
    
    def __init__(self, model: xgb.Booster, X_train: np.ndarray, 
                 feature_names: Optional[List[str]] = None):
        """
        Initialize SHAP analyzer
        
        Args:
            model: Trained XGBoost Booster
            X_train: Training data for SHAP explainer
            feature_names: List of feature names
        """
        self.model = model
        self.explainer = shap.TreeExplainer(model)
        self.X_train = X_train
        self.feature_names = feature_names
    
    def analyze(self, X_test: np.ndarray, save_path: str = "results/shap_summary.png") -> np.ndarray:
        """
        Generate SHAP analysis (equivalent to R SHAP code)
        
        Args:
            X_test: Test data for SHAP value calculation
            save_path: Path to save summary plot
            
        Returns:
            SHAP values array
        """
        # Calculate SHAP values
        print("Calculating SHAP values...")
        shap_values = self.explainer.shap_values(X_test)
        
        # Ensure results directory exists
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        
        # Summary plot (beeswarm) - shows feature importance and effects
        print("Generating SHAP summary plot...")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, 
                         feature_names=self.feature_names,
                         show=False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved SHAP summary plot to {save_path}")
        plt.close()
        
        # Feature importance bar plot
        print("Generating SHAP importance bar plot...")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, 
                         feature_names=self.feature_names,
                         plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig('results/shap_importance.png', dpi=300, bbox_inches='tight')
        print("Saved SHAP importance to results/shap_importance.png")
        plt.close()
        
        return shap_values
    
    def plot_force_plot(self, X_test: np.ndarray, index: int = 0, 
                       save_path: str = "results/shap_force_plot.html"):
        """
        Generate force plot for individual prediction
        
        Args:
            X_test: Test data
            index: Index of sample to explain
            save_path: Path to save HTML plot
        """
        shap_values = self.explainer.shap_values(X_test)
        
        # Force plot for single prediction
        shap.force_plot(
            self.explainer.expected_value,
            shap_values[index],
            X_test[index],
            feature_names=self.feature_names,
            matplotlib=False
        )
        
        # Save as HTML
        shap.save_html(save_path, 
                      shap.force_plot(
                          self.explainer.expected_value,
                          shap_values[index],
                          X_test[index],
                          feature_names=self.feature_names
                      ))
        print(f"Saved force plot to {save_path}")
    
    def plot_dependence(self, X_test: np.ndarray, feature_idx: int,
                       save_path: str = "results/shap_dependence.png"):
        """
        Generate dependence plot for a specific feature
        
        Args:
            X_test: Test data
            feature_idx: Index of feature to analyze
            save_path: Path to save plot
        """
        shap_values = self.explainer.shap_values(X_test)
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_idx,
            shap_values,
            X_test,
            feature_names=self.feature_names,
            show=False
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved dependence plot to {save_path}")
        plt.close()
