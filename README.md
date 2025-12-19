# NBA Injury Risk Modeling

**XGBoost-based injury prediction for NBA roster management**

## Problem Statement

Professional sports teams face significant financial risk from player injuries. For NBA franchises operating under strict salary cap constraints, an injured max-contract player represents lost on-court production, constrained roster flexibility, and reduced championship probability.

This project develops a probabilistic injury risk model to quantify player health risk based on historical injury patterns, workload metrics, and biographical factors.

## Approach

**Algorithm**: XGBoost (Gradient Boosted Decision Trees)  
**Time Periods Analyzed**: 3 distinct modeling periods  
**Training Data**: Historical NBA injury records (2010-2023), player statistics, and contract information

### Why XGBoost?

- Captures non-linear interactions between age, workload, and injury risk
- Robust to sparse historical records
- Produces interpretable feature importance
- Standard in sports analytics and actuarial risk modeling

## Data Sources

- NBA injury records (2010-2023)
- Player biographical data (age, position, draft year)
- Season statistics (games played, minutes, usage rate)
- Contract information (salary, years remaining)

## Results

### Model Performance

**Balanced Accuracy**: 0.6009

This represents the model's ability to correctly classify injury risk across balanced classes. Given the inherent randomness of contact injuries and data sparsity, this performance indicates meaningful signal extraction from available features.

### Top 5 Predictors

1. **Days since last injury**: Recent injury history strongly predicts future risk
2. **Age-adjusted usage rate**: Older players with high minutes face elevated risk
3. **Rolling 30-day workload**: Cumulative fatigue indicator
4. **Historical injury count**: Prior injury frequency
5. **Position-specific risk score**: Centers and power forwards show higher baseline risk

## Evidence

**Full Analysis Report**: [docs/nba_injury_report.pdf](docs/nba_injury_report.pdf)  

The PDF report contains:
- Detailed methodology
- Model evaluation across 3 time periods
- Feature importance visualizations
- Business implications

**Note**: The PDF document "Minimizing-NBA-Injury-Risk-for-Monetary-Reward-1.pdf" should be uploaded to the `docs/` folder.

## Reproducibility

### Requirements

- R 4.3+
- Core packages: `xgboost`, `tidymodels`, `tidyverse`
- Evaluation: `pROC`, `caret`

### Setup

```r
# Install renv for package management
install.packages("renv")
renv::restore()  # Installs exact package versions
```

### Running the Analysis

```r
# Navigate to analysis folder
setwd("analysis")

# Run the R Markdown file
rmarkdown::render("nba_injury_modeling.Rmd")
```

**Note**: The R Markdown file "NBA_Injury_Project_Final-1.Rmd" should be uploaded to the `analysis/` folder as `nba_injury_modeling.Rmd`.

## Project Structure

```
nba-injury-prediction/
├── README.md
├── analysis/
│   └── nba_injury_modeling.Rmd    # Main analysis (TODO: upload)
├── docs/
│   └── nba_injury_report.pdf       # Full report (TODO: upload)
├── data/
│   └── (raw injury and player data)
└── results/
    └── (model outputs and figures)
```

## Limitations

- **Stochastic events**: Contact injuries and acute trauma remain unpredictable
- **Data sparsity**: Severe injuries are rare events (class imbalance)
- **Missing biomechanics**: No access to movement screening or strength metrics
- **Reporting heterogeneity**: Teams vary in injury disclosure practices

This model should be used as a decision support tool, not a deterministic predictor.

## Tech Stack

- **Language**: R
- **ML**: XGBoost, tidymodels
- **Evaluation**: pROC, caret
- **Reproducibility**: renv

## License

MIT License - For analytical demonstration purposes.

---

**Note**: This project demonstrates machine learning for sports analytics. Any real-world deployment should involve collaboration with certified sports medicine professionals.
