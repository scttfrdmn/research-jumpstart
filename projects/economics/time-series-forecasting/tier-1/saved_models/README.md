# Saved Models Directory

This directory stores trained forecasting models for persistence between Studio Lab sessions.

## Directory Structure

```
saved_models/
├── arima/              # ARIMA model objects
├── var/                # VAR model objects
├── lstm/               # LSTM neural network weights
├── prophet/            # Prophet model objects
├── xgboost/            # XGBoost model objects
└── ensemble/           # Ensemble model combinations
```

## Model Formats

### ARIMA Models
- Format: Pickle (.pkl)
- Library: statsmodels
- Usage: `joblib.load('arima/model_US_GDP.pkl')`

### VAR Models
- Format: Pickle (.pkl)
- Library: statsmodels
- Includes: Fitted coefficients, lag order, variable names

### LSTM Models
- Format: HDF5 (.h5) or SavedModel
- Library: TensorFlow/Keras
- Usage: `keras.models.load_model('lstm/gdp_forecast.h5')`
- Also saves: Scaler objects, sequence lengths

### Prophet Models
- Format: Pickle (.pkl)
- Library: Prophet
- Includes: Fitted trend, seasonality components

### XGBoost Models
- Format: JSON or pickle
- Library: XGBoost
- Usage: `xgboost.Booster().load_model('xgboost/model.json')`

### Ensemble Models
- Format: Pickle (.pkl)
- Contains: Dictionary of models + weights
- Usage: `joblib.load('ensemble/ensemble_v1.pkl')`

## Naming Convention

Models are named using the pattern:
```
{model_type}_{country}_{indicator}_{version}.{ext}
```

Examples:
- `arima_US_GDP_v1.pkl`
- `lstm_multi_country_GDP_v2.h5`
- `var_G7_panel_v1.pkl`
- `ensemble_20countries_v3.pkl`

## Model Metadata

Each model has an associated metadata JSON file:
```json
{
  "model_type": "LSTM",
  "country": "US",
  "indicator": "GDP_Growth",
  "training_date": "2024-01-15",
  "training_samples": 120,
  "test_rmse": 0.45,
  "hyperparameters": {
    "lookback": 12,
    "units": [64, 32],
    "dropout": 0.2
  }
}
```

## Model Checkpointing

During long training runs, models are checkpointed:
- Automatic checkpoints every epoch for LSTM
- Best model saved based on validation loss
- Resume training from checkpoint if interrupted

Example:
```python
# Save checkpoint
model.save('lstm/gdp_checkpoint_epoch50.h5')

# Resume training
model = keras.models.load_model('lstm/gdp_checkpoint_epoch50.h5')
```

## Storage Management

- Models are gitignored (too large for git)
- Keep only best versions to save space
- Old models archived or deleted after 90 days
- Typical model sizes:
  - ARIMA: <1 MB
  - VAR: 1-5 MB
  - LSTM: 10-50 MB
  - XGBoost: 5-20 MB
  - Ensemble: varies

## Loading Models

Use the utility functions:

```python
from src.models import load_model

# Load any model type
model = load_model('saved_models/ensemble/ensemble_v1.pkl')

# Make predictions
forecast = model.predict(new_data, horizon=12)
```

## Version Control

- v1: Initial model
- v2: Hyperparameter tuning
- v3: More training data
- v4: Architecture changes
- etc.

Keep a CHANGELOG in this directory to track model improvements.
