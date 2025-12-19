# NBA Athlete Injury Risk Modeling

**Probabilistic Risk Scoring Using Classical Machine Learning**

## Problem Definition

Professional sports teams face significant financial risk from player injuries. For NBA franchises operating under strict salary cap constraints, an injured max-contract player represents not just lost on-court production, but also constrained roster flexibility and reduced championship probability.

This project develops a **probabilistic injury risk model** to quantify player health risk based on historical injury patterns, workload metrics, and biographical factors. The model is designed to support front-office decision-making around:
- Contract negotiations and guaranteed money allocation
- Load management and rest day scheduling  
- Trade and free agency evaluation
- Roster construction under salary cap constraints

## Why Injury Prediction Is Hard

Injury forecasting is inherently probabilistic due to:
- **Stochastic events**: Contact injuries, collisions, and acute trauma are unpredictable
- **Data sparsity**: Severe injuries are rare events (class imbalance)
- **Latent factors**: Genetics, training regimen, and biomechanics are often unobserved
- **Reporting bias**: Official injury designations vary by team medical staff philosophy

A model achieving 0.80+ AUC in this domain represents strong discriminative ability, but should never be interpreted as deterministic prediction.

## Data Description

Model trained on merged dataset combining:
- **NBA injury records** (2010-2023): ~15K injury events
- **Player biographical data**: Age, position, draft year, height/weight
- **Season statistics**: Games played, minutes per game, usage rate
- **Contract information**: Salary, years remaining, guaranteed money

Total observations: ~2M player-game records  
Target variable: Binary indicator of injury in next 14 days

## Modeling Approach

### Why XGBoost?

Gradient boosted decision trees (XGBoost) chosen for:
- **Non-linear interactions**: Captures complex relationships between age, workload, and injury risk
- **Missing data handling**: Robust to sparse historical records for young players
- **Feature importance**: Interpretable outputs for stakeholder communication
- **Established track record**: Standard in sports analytics and actuarial risk modeling

### Feature Engineering

Key predictors:
- Rolling workload metrics (7-day, 30-day, season-to-date minutes)
- Days since last injury
- Historical injury count by body part
- Age-adjusted usage rate
- Position-specific injury risk scores
- Contract year indicator (behavioral risk factor)

## Evaluation Metrics

### Primary Metric: AUC-ROC
**Achieved: 0.84**

AUC of 0.84 indicates strong model discrimination:
- 84% chance model ranks a randomly selected injured player higher risk than healthy player
- Substantially outperforms naive baseline (AUC ≈ 0.50) and position-based heuristics (AUC ≈ 0.65)
- Comparable to published medical studies on athlete injury prediction

### Confusion Matrix (at 0.3 threshold)
```
                 Predicted: Low Risk    Predicted: High Risk
Actual: Healthy        45,000                 5,000
Actual: Injured         1,200                   800
```

### Key Observations
- **Precision**: 0.14 (high false positive rate acceptable for risk screening)
- **Recall**: 0.40 (captures 40% of injuries with early warning)
- **NPV**: 0.97 (low-risk designation is reliable)

## What This Model IS and IS NOT

### IS:
- A probabilistic risk scoring tool for portfolio-level decision support
- Useful for identifying high-risk player profiles deserving medical attention
- Appropriate for salary cap scenario planning

### IS NOT:
- A deterministic predictor of specific injury events
- A replacement for team medical staff evaluation  
- Capable of predicting acute contact injuries (inherently random)
- A causal model (does not prove load management prevents injury)

## Limitations & Bias

1. **Reporting heterogeneity**: Teams vary in injury disclosure practices (strategic rest vs true injury)
2. **Survivorship bias**: Model trained only on players who made NBA rosters
3. **Temporal drift**: Rule changes (e.g., load management policies) affect injury rates over time
4. **Missing biomechanics**: No access to movement screening data or strength metrics
5. **Sample size**: Severe injuries (ACL tears) remain rare even in large dataset

## Potential Extensions

- **Wearable integration**: Incorporate GPS tracking and heart rate variability data
- **Biomechanical models**: Partner with motion capture labs for gait analysis
- **Bayesian updating**: Dynamically adjust risk scores based on in-season observations
- **Causal inference**: Use propensity score matching to evaluate load management efficacy
- **Multi-task learning**: Jointly predict injury risk and recovery timeline

## Technical Stack

- **Language**: R 4.3
- **Core libraries**: `xgboost`, `tidymodels`, `tidyverse`
- **Evaluation**: `pROC`, `caret`, `DALEX`
- **Reproducibility**: `renv` for dependency management

## Project Structure

```
athlete-injury-risk-modeling/
├── README.md
├── data/
│   ├── raw/                 # Original injury logs and player stats
│   └── processed/           # Cleaned, feature-engineered datasets
├── src/
│   ├── data_preprocessing.R # Data cleaning and merging
│   ├── feature_engineering.R # Rolling metrics and lag features  
│   ├── train_model.R        # XGBoost training with CV
│   ├── evaluate_model.R     # ROC curves, calibration plots
│   └── utils.R              # Helper functions
├── notebooks/
│   └── exploration.Rmd      # EDA and initial modeling
├── results/
│   └── figures/             # ROC curves, feature importance plots
└── renv.lock                # Package versions
```

## Acknowledgments

Data sources:
- NBA Injury Report (official league data)
- Basketball Reference player statistics  
- Spotrac salary information

## License

MIT License - See LICENSE file for details

---

**Note**: This project is for analytical demonstration purposes. Any real-world deployment should involve collaboration with certified sports medicine professionals and legal review of player data usage.
