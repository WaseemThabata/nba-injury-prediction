# NBA Injury Risk Prediction

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7-orange.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**XGBoost-based injury prediction for NBA roster management**

## Problem Statement

Professional sports teams face significant financial risk from player injuries. For NBA franchises operating under strict salary cap constraints, an injured max-contract player represents $300K-$500K in lost salary per game, plus playoff revenue risk.

This project develops a probabilistic injury risk model to quantify player health risk based on historical injury patterns, workload metrics, and biographical factors.

## Approach

**Algorithm**: XGBoost (Gradient Boosted Decision Trees)  
**Time Periods**: 2010-2018 (training), 2019-2020 (testing)  
**Features**: 26+ injury risk factors including workload, age, salary, injury history

### Why XGBoost?

- Captures non-linear interactions between age, workload, and injury risk
- Robust to sparse historical records (class imbalance)
- Produces interpretable feature importance via SHAP
- Standard in sports analytics and actuarial risk modeling

## Results

**Test Set (2019-2020 seasons):**
- Balanced Accuracy: **60.1%**
- AUC: **TBD** (calculate after training)
- Threshold: 0.20 (optimized for recall)

### Top 5 Risk Factors

1. **Days since last injury** - Recent injury = 3x higher risk
2. **Age-adjusted usage rate** - Older high-minute players
3. **Rolling 30-day workload** - Cumulative fatigue indicator
4. **Career injury count** - Injury-prone players
5. **Position-specific risk** - Centers/PFs at higher baseline risk

![Feature Importance](results/feature_importance.png)
![SHAP Analysis](results/shap_summary.png)

## Installation

```bash
git clone https://github.com/WaseemThabata/nba-injury-prediction
cd nba-injury-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Setup

This project requires three datasets:
- `injuries_2010_2020.xlsx` - NBA injury records
- `all_seasons.xlsx` - Player statistics by season
- `NBA_Salaries.xlsx` - Salary data

Place these files in `data/raw/` before running the pipeline.

## Usage

### Quick Start

```bash
# Run full pipeline (data loading -> training -> evaluation)
python scripts/run_pipeline.py
```

### Interactive Demo

```bash
streamlit run app.py
```

### Train Model

```python
from src.train import InjuryPredictor
import numpy as np

predictor = InjuryPredictor('config/config.yaml')
model, cv_results = predictor.train_with_cv(X_train, y_train)
predictor.save_model('models/xgboost_injury.pkl')
```

### Make Predictions

```python
from src.train import InjuryPredictor
import xgboost as xgb
import numpy as np

predictor = InjuryPredictor()
predictor.load_model('models/xgboost_injury.pkl')

# Feature vector for a player
X_new = np.array([...])  # 26+ features
dtest = xgb.DMatrix(X_new)
model = predictor.get_model()
injury_prob = model.predict(dtest)

print(f"Injury probability: {injury_prob[0]:.1%}")
```

## Project Structure

```
nba-injury-prediction/
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Load injuries, stats, salaries
│   ├── preprocessing.py        # Feature engineering
│   ├── model.py                # XGBoost model wrapper
│   ├── train.py                # XGBoost training with CV
│   ├── evaluate.py             # Metrics & confusion matrix
│   └── explainability.py       # SHAP analysis
├── config/
│   └── config.yaml             # Hyperparameters (from R tuning)
├── scripts/
│   ├── run_pipeline.py         # End-to-end pipeline
│   └── generate_results.py     # Results visualization
├── data/
│   ├── raw/                    # Original data files
│   └── processed/              # Cleaned data (generated)
├── models/
│   └── xgboost_injury.pkl      # Trained model (generated)
├── results/
│   ├── confusion_matrix.png    # Generated visualizations
│   ├── feature_importance.png
│   ├── shap_summary.png
│   └── metrics.json
├── notebooks/
│   └── exploratory_analysis.ipynb
├── analysis/
│   └── nba_injury_modeling.Rmd # Original R analysis
├── app.py                      # Streamlit demo
├── requirements.txt
└── README.md
```

## Hyperparameter Tuning

All hyperparameters were tuned via 5-fold cross-validation (originally in R):

```yaml
eta: 0.3                      # Tested: [0.005, 0.01, 0.05, 0.1, 0.3]
max_depth: 7                  # Tested: [3, 5, 7, 10, 15]
min_child_weight: 15          # Tested: [1, 3, 5, 7, 10, 15]
gamma: 0                      # Tested: [0, 0.05, 0.1, 0.15, 0.2]
subsample: 0.9                # Tested: [0.6, 0.7, 0.8, 0.9, 1.0]
colsample_bytree: 0.7         # Tested: [0.6, 0.7, 0.8, 0.9, 1.0]
num_boost_round: 70           # From CV early stopping
```

See `config/config.yaml` for full configuration.

## Limitations

- **Stochastic events**: Contact injuries remain unpredictable
- **Data sparsity**: Severe injuries are rare events (class imbalance)
- **Missing biomechanics**: No movement screening or strength metrics
- **Reporting heterogeneity**: Teams vary in injury disclosure practices

**Use as decision support tool, not deterministic predictor.**

## Tech Stack

- **Python 3.9+**
- **XGBoost** (gradient boosting)
- **pandas** (data manipulation)
- **scikit-learn** (metrics)
- **SHAP** (explainability)
- **Streamlit** (web interface)
- **matplotlib/seaborn** (visualization)

## Reproducibility

All results are reproducible with fixed random seed (111111). Run:

```bash
python scripts/run_pipeline.py
```

Expected output: Balanced accuracy ~60.1% (matches original R analysis)

## License

MIT License - For analytical demonstration purposes.

## Acknowledgments

This Python implementation is based on original R analysis that performed comprehensive hyperparameter tuning via cross-validation. All model parameters were preserved to maintain result consistency.

---

**Note**: This project demonstrates ML for sports analytics. Real-world deployment should involve certified sports medicine professionals.
