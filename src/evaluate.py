"""
Model evaluation and metrics for NBA injury prediction
Replicates R evaluation logic with confusion matrix and balanced accuracy
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    balanced_accuracy_score,
    roc_auc_score,
    classification_report,
    accuracy_score
)
import xgboost as xgb
import json
from pathlib import Path


class ModelEvaluator:
    """
    Evaluate injury prediction model
    Matches R output: confusion matrix, balanced accuracy, AUC
    """
    
    def __init__(self, model: xgb.Booster, threshold: float = 0.2):
        """
        Initialize evaluator
        
        Args:
            model: Trained XGBoost Booster
            threshold: Classification threshold (from R: 0.2)
        """
        self.model = model
        self.threshold = threshold
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, 
                 feature_names: list = None) -> dict:
        """
        Evaluate model and generate metrics (matching R output)
        From R: boost_pred_label[boost_preds_1 >= 0.2] <- 1
        
        Args:
            X_test: Test features
            y_test: Test labels
            feature_names: Feature names for importance plot
            
        Returns:
            Dictionary of metrics
        """
        dtest = xgb.DMatrix(X_test)
        
        # Get predictions
        y_pred_proba = self.model.predict(dtest)
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        
        # Calculate metrics
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        acc = accuracy_score(y_test, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'balanced_accuracy': float(balanced_acc),
            'accuracy': float(acc),
            'auc': float(auc),
            'threshold': self.threshold,
            'confusion_matrix': {
                'TN': int(cm[0, 0]),
                'FP': int(cm[0, 1]),
                'FN': int(cm[1, 0]),
                'TP': int(cm[1, 1])
            }
        }
        
        print("\n=== Evaluation Results ===")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        print(f"Accuracy: {acc:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"Threshold: {self.threshold}")
        print(f"\nConfusion Matrix:")
        print(cm)
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Injury', 'Injury']))
        
        # Save metrics
        Path('results').mkdir(exist_ok=True)
        with open('results/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        print("\nMetrics saved to results/metrics.json")
        
        # Plot confusion matrix
        self._plot_confusion_matrix(cm)
        
        # Plot feature importance
        if feature_names:
            self._plot_feature_importance(feature_names)
        else:
            self._plot_feature_importance()
        
        return metrics
    
    def _plot_confusion_matrix(self, cm: np.ndarray):
        """Save confusion matrix plot"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Injury', 'Injury'],
                   yticklabels=['No Injury', 'Injury'])
        plt.title('Confusion Matrix - NBA Injury Prediction')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('results/confusion_matrix.png', dpi=300)
        print("Saved confusion matrix to results/confusion_matrix.png")
        plt.close()
    
    def _plot_feature_importance(self, feature_names: list = None):
        """Plot and save feature importance"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if feature_names:
            # Get feature importance
            importance = self.model.get_score(importance_type='gain')
            
            # Convert to sorted list
            features = []
            scores = []
            for feat_idx, score in importance.items():
                feat_num = int(feat_idx[1:]) if feat_idx.startswith('f') else int(feat_idx)
                if feat_num < len(feature_names):
                    features.append(feature_names[feat_num])
                    scores.append(score)
            
            # Sort by score and take top 10
            sorted_idx = np.argsort(scores)[-10:]
            features_sorted = [features[i] for i in sorted_idx]
            scores_sorted = [scores[i] for i in sorted_idx]
            
            # Plot
            ax.barh(features_sorted, scores_sorted)
            ax.set_xlabel('Gain')
            ax.set_title('Top 10 Feature Importance')
        else:
            # Use default XGBoost plotting
            xgb.plot_importance(self.model, ax=ax, max_num_features=10,
                               importance_type='gain', title='Top 10 Feature Importance')
        
        plt.tight_layout()
        plt.savefig('results/feature_importance.png', dpi=300)
        print("Saved feature importance to results/feature_importance.png")
        plt.close()
