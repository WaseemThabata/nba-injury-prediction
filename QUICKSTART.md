# Quick Start Guide

Get started with NBA Injury Prediction in 5 minutes!

## Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- 3 data files (see `data/README.md`)

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/WaseemThabata/nba-injury-prediction.git
cd nba-injury-prediction
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- XGBoost (machine learning)
- pandas (data processing)
- scikit-learn (metrics)
- SHAP (explainability)
- Streamlit (web interface)
- And more...

## Data Setup

Place these 3 files in `data/raw/`:
1. `injuries_2010_2020.xlsx`
2. `all_seasons.xlsx`
3. `NBA_Salaries.xlsx`

See `data/README.md` for details.

## Running the Pipeline

### Option 1: Complete Pipeline

Train the model end-to-end:

```bash
python scripts/run_pipeline.py
```

This will:
1. Load and merge datasets
2. Engineer features
3. Train XGBoost with cross-validation
4. Evaluate on test set
5. Generate SHAP analysis
6. Save model and results

**Expected output**: Balanced accuracy ~60.1%

### Option 2: Interactive Demo

Launch the Streamlit web interface:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### Option 3: Jupyter Notebook

Explore interactively:

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

## What Gets Generated

After running the pipeline, you'll see:

```
models/
└── xgboost_injury.pkl          # Trained model

results/
├── metrics.json                # Performance metrics
├── confusion_matrix.png        # Confusion matrix plot
├── feature_importance.png      # Top features
├── shap_summary.png           # SHAP analysis
└── shap_importance.png        # SHAP bar chart
```

## Making Predictions

```python
from src.train import InjuryPredictor
import xgboost as xgb
import numpy as np

# Load model
predictor = InjuryPredictor()
model = predictor.load_model('models/xgboost_injury.pkl')

# Create feature vector (26+ features)
player_features = np.array([[
    27,      # age
    65,      # games played
    28.5,    # minutes per game
    22.0,    # usage rate
    10000000,# salary
    # ... more features
]])

# Predict
dtest = xgb.DMatrix(player_features)
injury_prob = model.predict(dtest)
print(f"Injury probability: {injury_prob[0]:.1%}")
```

## Configuration

Edit `config/config.yaml` to customize:
- Hyperparameters (from R tuning)
- Train/test split years
- Evaluation threshold
- Paths

## Troubleshooting

### Import Errors

```bash
# Make sure you're in the virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Data Not Found

```
ERROR: No data loaded. Please ensure data files exist...
```

**Solution**: Place data files in `data/raw/` directory (see `data/README.md`)

### Module Not Found

```python
ModuleNotFoundError: No module named 'src'
```

**Solution**: Run scripts from project root directory, not from subdirectories

```bash
# Correct
cd nba-injury-prediction
python scripts/run_pipeline.py

# Wrong
cd scripts
python run_pipeline.py
```

## Next Steps

1. ✅ Run the pipeline
2. ✅ Review results in `results/`
3. ✅ Try the Streamlit demo
4. ✅ Read model documentation in README.md
5. ✅ Explore the Jupyter notebook
6. ✅ Customize hyperparameters in `config/config.yaml`

## Getting Help

- Check README.md for detailed documentation
- Review `data/README.md` for data requirements
- Examine example notebook in `notebooks/`
- Open an issue on GitHub

---

**Happy Modeling! 🏀**
