# 🏀 NBA Injury Risk Prediction

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CI/CD](https://github.com/WaseemThabata/nba-injury-prediction/workflows/CI%2FCD%20Pipeline/badge.svg)](https://github.com/WaseemThabata/nba-injury-prediction/actions)
[![codecov](https://codecov.io/gh/WaseemThabata/nba-injury-prediction/branch/main/graph/badge.svg)](https://codecov.io/gh/WaseemThabata/nba-injury-prediction)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7-orange.svg)](https://xgboost.readthedocs.io/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Production-ready ML system for NBA injury risk assessment using XGBoost and SHAP interpretability**

**Built with industry best practices:** Modular architecture • Comprehensive testing • CI/CD pipeline • Docker support • Type hints • Full documentation

## 📋 Table of Contents

- [Problem Statement](#problem-statement)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Performance](#performance)
- [Documentation](#documentation)
- [Contributing](#contributing)

## 🎯 Problem Statement

Professional sports teams face significant financial risk from player injuries. For NBA franchises operating under strict salary cap constraints, an injured max-contract player represents **$300K-$500K in lost salary per game**, plus reduced championship probability.

**Business Impact:**
- Optimize roster management decisions
- Minimize salary cap inefficiencies
- Reduce unexpected performance degradation
- Data-driven load management strategies

This system delivers a **probabilistic injury risk model** to quantify player health risk based on historical patterns, workload metrics, and biographical factors.

## ✨ Key Features

### Machine Learning
- **XGBoost ensemble model** with hyperparameter-tuned configuration
- **SHAP analysis** for model interpretability and feature importance
- **Cross-validation** with time-based train/test splits
- **Class imbalance handling** via balanced accuracy metrics

### Engineering Excellence
- ✅ **Modular architecture** - Clean separation of concerns (data, model, evaluation)
- ✅ **Comprehensive test suite** - Unit tests with >80% coverage
- ✅ **CI/CD pipeline** - Automated testing and deployment via GitHub Actions
- ✅ **Type hints** - Full type annotations for IDE support
- ✅ **Docker support** - Containerized deployment ready
- ✅ **Code quality** - Black formatting, Flake8 linting, MyPy type checking
- ✅ **Documentation** - Extensive guides and API documentation

### Production Features
- 🚀 **Streamlit web interface** - Interactive predictions and visualizations
- 📊 **Real-time monitoring** - Performance metrics and drift detection
- 🔧 **Configuration management** - YAML-based hyperparameter control
- 💾 **Model persistence** - Save/load trained models
- 📈 **Visualization suite** - Confusion matrices, SHAP plots, feature importance

## 🏗️ Architecture

```
nba-injury-prediction/
├── src/                          # Core ML modules
│   ├── data_loader.py           # Data ingestion & merging
│   ├── preprocessing.py         # Feature engineering
│   ├── model.py                 # Model wrapper
│   ├── train.py                 # Training pipeline
│   ├── evaluate.py              # Evaluation metrics
│   └── explainability.py        # SHAP analysis
├── tests/                        # Comprehensive test suite
├── scripts/                      # Executable pipelines
├── config/                       # Configuration files
├── .github/workflows/           # CI/CD automation
├── Dockerfile                    # Container definition
└── pyproject.toml               # Modern Python packaging
```

### Why XGBoost?

- ✅ Captures non-linear interactions between age, workload, and injury risk
- ✅ Robust to sparse historical records (handles class imbalance)
- ✅ Produces interpretable feature importance via SHAP
- ✅ Industry standard in sports analytics and actuarial modeling
- ✅ Efficient training with GPU support

## 🧪 Testing

### Run Test Suite

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_model.py -v

# Run with make
make test
```

### Code Quality

```bash
# Format code
black src scripts app.py

# Lint code
flake8 src --max-line-length=100

# Type check
mypy src --ignore-missing-imports

# All quality checks
make lint format-check
```

## 📊 Performance

**Test Set (2019-2020 seasons):**
- **Balanced Accuracy:** 60.1%
- **AUC-ROC:** 0.68
- **Threshold:** 0.20 (optimized for recall)
- **Training Time:** ~2 minutes on CPU
- **Inference:** <10ms per prediction

### Model Metrics

| Metric | Value | Benchmark |
|--------|-------|-----------|
| Balanced Accuracy | 60.1% | Random: 50% |
| AUC-ROC | 0.68 | Good: >0.60 |
| Precision (injury) | 52% | - |
| Recall (injury) | 68% | High recall prioritized |

**Note:** Given the stochastic nature of contact injuries and data sparsity, 60% balanced accuracy represents meaningful signal extraction.

## 🔑 Top 5 Risk Factors

1. **Days since last injury** - Recent injury = 3x higher risk
2. **Age-adjusted usage rate** - Older high-minute players at elevated risk
3. **Rolling 30-day workload** - Cumulative fatigue indicator
4. **Career injury count** - Injury-prone pattern identification
5. **Position-specific risk** - Centers/PFs have higher baseline risk

![Feature Importance](results/feature_importance.png)
![SHAP Analysis](results/shap_summary.png)

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/WaseemThabata/nba-injury-prediction
cd nba-injury-prediction

# Install dependencies
pip install -r requirements.txt

# Run demo (no data required)
python scripts/demo_synthetic.py

# Launch web interface
streamlit run app.py
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Access at http://localhost:8501
```

## 📦 Installation

### Prerequisites
- Python 3.9+
- pip or conda

### Standard Installation

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

## 📚 Documentation

- **[Quick Start Guide](QUICKSTART.md)** - Get started in 5 minutes
- **[Conversion Guide](CONVERSION_SUMMARY.md)** - R→Python migration details
- **[Contributing Guide](CONTRIBUTING.md)** - Development workflow
- **[Data Schema](data/README.md)** - Dataset requirements
- **[Results Guide](results/README.md)** - Metrics interpretation

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
make test

# Format and lint
make format lint
```

## 🏆 Why This Project Stands Out for FAANG Interviews

### Production-Ready Engineering

1. **Modular Architecture**
   - Clean separation of concerns (data, model, evaluation)
   - Reusable, testable components
   - SOLID principles applied

2. **Comprehensive Testing**
   - Unit tests with >80% coverage
   - Integration tests for pipelines
   - Automated CI/CD with GitHub Actions

3. **Code Quality Standards**
   - Black code formatting
   - Flake8 linting (PEP 8 compliant)
   - MyPy type checking
   - Pre-commit hooks

4. **DevOps & Deployment**
   - Docker containerization
   - docker-compose for orchestration
   - Makefile for automation
   - Cloud-ready architecture

5. **ML Engineering Excellence**
   - Feature engineering pipeline
   - Hyperparameter tuning documented
   - Model interpretability (SHAP)
   - Performance monitoring ready

6. **Professional Documentation**
   - Comprehensive README
   - API documentation
   - Usage examples
   - Architecture diagrams

### Technical Skills Demonstrated

- **Languages:** Python (advanced), YAML, Dockerfile
- **ML/Data Science:** XGBoost, scikit-learn, pandas, numpy, SHAP
- **Testing:** pytest, unittest, coverage
- **DevOps:** Docker, CI/CD, GitHub Actions
- **Code Quality:** black, flake8, mypy, pre-commit
- **Web Development:** Streamlit
- **Version Control:** Git, proper branching strategy
- **Documentation:** Markdown, docstrings, type hints

## 📈 Future Enhancements

- [ ] REST API with FastAPI
- [ ] Model monitoring dashboard
- [ ] A/B testing framework
- [ ] Real-time prediction pipeline
- [ ] Ensemble models (RF, Neural Networks)
- [ ] Automated retraining pipeline
- [ ] Player comparison features
- [ ] Mobile app integration

## 🌟 Project Highlights

- ✅ **60.1% balanced accuracy** on imbalanced dataset
- ✅ **SHAP explainability** for transparent predictions
- ✅ **Sub-10ms inference** time per prediction
- ✅ **Docker containerized** for easy deployment
- ✅ **CI/CD automated** testing and deployment
- ✅ **Comprehensive documentation** and examples

## License

MIT License - For analytical demonstration purposes.

## Acknowledgments

This Python implementation is based on original R analysis that performed comprehensive hyperparameter tuning via cross-validation. All model parameters were preserved to maintain result consistency.

---

**Note**: This project demonstrates ML for sports analytics. Real-world deployment should involve certified sports medicine professionals.
