# API Documentation

## Core Modules

### data_loader.py

#### NBADataLoader

Load and merge NBA datasets.

**Methods:**
- `load_injuries()` - Load injury records
- `load_player_stats()` - Load player statistics
- `load_salaries()` - Load salary data
- `merge_datasets()` - Merge all datasets

### preprocessing.py

#### NBAPreprocessor

Feature engineering and data preprocessing.

**Methods:**
- `engineer_features(df)` - Create derived features
- `create_train_test_split(df)` - Time-based splitting
- `prepare_features(df)` - Extract feature matrix and labels

### model.py

#### InjuryPredictionModel

XGBoost model wrapper.

**Methods:**
- `predict(X)` - Get injury probabilities
- `predict_class(X, threshold)` - Get binary predictions
- `save(path)` - Save model to disk
- `load(path)` - Load model from disk

### train.py

#### InjuryPredictor

Training pipeline with cross-validation.

**Methods:**
- `train(X_train, y_train)` - Train model
- `train_with_cv(X_train, y_train, n_folds)` - Train with CV
- `save_model(path)` - Save trained model
- `load_model(path)` - Load trained model

### evaluate.py

#### ModelEvaluator

Model evaluation and metrics.

**Methods:**
- `evaluate(X_test, y_test)` - Generate evaluation metrics

### explainability.py

#### SHAPAnalyzer

SHAP-based model interpretation.

**Methods:**
- `analyze(X_test)` - Generate SHAP analysis
