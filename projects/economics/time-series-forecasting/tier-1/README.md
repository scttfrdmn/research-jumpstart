# Multi-Country Ensemble Economic Modeling

**Duration:** 4-8 hours
**Platform:** SageMaker Studio Lab ONLY
**Cost:** $0 (free Studio Lab account, no AWS charges)
**Data:** ~10GB multi-country panel data

## Research Goal

Perform comprehensive economic forecasting using multi-country panel data and ensemble modeling techniques. Train multiple forecasting models (ARIMA, VAR, LSTM, Prophet, XGBoost) on macroeconomic indicators from 20+ countries, analyze cross-country spillover effects, and generate probabilistic forecasts with uncertainty quantification.

**This project requires Studio Lab** - it won't work on Colab Free due to:
- **10GB dataset** (Colab has no persistent storage)
- **5-6 hour continuous training** (Colab disconnects after 90 min idle)
- **Model checkpointing essential** (must resume from checkpoints)
- **Environment persistence** (complex dependency stack)

## What This Enables

Real research that isn't possible on Colab:

### 🔬 Dataset Persistence
- Download 10GB of multi-country economic data **once**
- Access instantly in all future sessions
- No 15-20 minute re-downloads every session
- Cache intermediate processing results
- Store trained models for reuse

### ⚡ Long-Running Training
- Train 5-6 different model types (30-60 min each)
- Total compute: 5-6 hours continuous
- Automatic checkpointing every epoch
- Resume from checkpoint if needed
- Parallel model training workflows

### 🧪 Reproducible Environments
- Conda environment with 25+ packages
- Persists between sessions
- No reinstalling dependencies
- Team members use identical setup
- Version-controlled requirements

### 📊 Iterative Analysis
- Save ensemble analysis results
- Build on previous runs
- Refine models incrementally
- Collaborative notebook development
- Persistent visualizations and reports

## What You'll Build

Multi-notebook research workflow:

1. **Data Acquisition** (60 min)
   - Download data from FRED, World Bank, OECD APIs
   - 20+ countries, 10+ indicators per country (~10GB)
   - Cache in persistent storage
   - Preprocess and align panel data
   - Handle missing values and outliers

2. **Exploratory Analysis** (45 min)
   - Cross-country correlations
   - Granger causality tests
   - Structural break detection
   - Business cycle synchronization
   - Regional spillover effects

3. **Ensemble Model Training** (5-6 hours)
   - ARIMA: Classical time series forecasting
   - VAR: Vector autoregression for multivariate modeling
   - LSTM: Deep learning for complex patterns
   - Prophet: Facebook's additive model
   - XGBoost: Gradient boosting for feature-rich forecasting
   - Ensemble weighting and combination

4. **Cross-Country Spillover Analysis** (60 min)
   - VAR-based impulse response functions
   - Forecast error variance decomposition
   - Network analysis of economic linkages
   - Policy shock simulations

5. **Uncertainty Quantification** (30 min)
   - Ensemble probabilistic forecasts
   - Prediction intervals
   - Scenario analysis
   - Risk assessment

6. **Results and Reporting** (30 min)
   - Model performance comparison
   - Publication-ready visualizations
   - Interactive dashboards
   - Export forecasts for downstream use

## Datasets

**Multi-Country Economic Panel**
- **Countries:** 20+ (US, EU-15, China, Japan, Canada, Australia, etc.)
- **Sources:** FRED, World Bank, OECD, IMF
- **Indicators per country:**
  - GDP growth (quarterly)
  - CPI inflation (monthly)
  - Unemployment rate (monthly)
  - Interest rates (monthly)
  - Exchange rates (daily → monthly)
  - Trade balance (monthly)
  - Industrial production (monthly)
  - Consumer confidence (monthly)
  - Stock market indices (daily → monthly)
  - Government debt (quarterly)
- **Period:** 1990-2024
- **Total size:** ~10GB CSV/Parquet files
- **Storage:** Cached in Studio Lab's 15GB persistent storage

## Setup

### 1. Create Studio Lab Account
1. Go to [SageMaker Studio Lab](https://studiolab.sagemaker.aws)
2. Request free account (approval in 1-2 days)
3. No credit card or AWS account needed

### 2. Import Project
```bash
# In Studio Lab terminal
git clone https://github.com/YOUR_USERNAME/research-jumpstart.git
cd research-jumpstart/projects/economics/time-series-forecasting/tier-1
```

### 3. Install Dependencies
```bash
# Create conda environment (persists between sessions!)
conda env create -f environment.yml
conda activate economics-studio-lab

# Or use pip
pip install -r requirements.txt
```

### 4. Run Notebooks in Order
1. `01_data_acquisition.ipynb` - Download and cache multi-country datasets
2. `02_exploratory_analysis.ipynb` - Cross-country correlations and causality
3. `03_ensemble_modeling.ipynb` - Train multiple forecasting models
4. `04_spillover_analysis.ipynb` - Analyze cross-country effects
5. `05_forecasting_dashboard.ipynb` - Generate forecasts and visualizations

## Key Features

### Persistence Example
```python
# Save trained models (persists between sessions!)
import joblib
joblib.dump(ensemble_models, 'saved_models/ensemble_v1.pkl')

# Load in next session
ensemble_models = joblib.load('saved_models/ensemble_v1.pkl')
```

### Longer Computations
```python
# Train multiple models sequentially (5-6 hours)
# Studio Lab: 12-hour GPU sessions, 4-hour CPU sessions
models = {
    'ARIMA': train_arima_models(countries),
    'VAR': train_var_models(countries),
    'LSTM': train_lstm_models(countries),
    'Prophet': train_prophet_models(countries),
    'XGBoost': train_xgboost_models(countries)
}
```

### Shared Utilities
```python
# Import from project modules (better code organization)
from src.data_utils import load_fred_data, load_world_bank_data
from src.models import EnsembleForecaster, VARModel
from src.analysis import granger_causality_test, spillover_analysis
from src.visualization import plot_forecast_comparison, create_dashboard
```

## Project Structure

```
tier-1/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment
│
├── notebooks/
│   ├── 01_data_acquisition.ipynb      # Download and cache data
│   ├── 02_exploratory_analysis.ipynb  # EDA and causality testing
│   ├── 03_ensemble_modeling.ipynb     # Train multiple models
│   ├── 04_spillover_analysis.ipynb    # Cross-country effects
│   └── 05_forecasting_dashboard.ipynb # Interactive forecasts
│
├── src/
│   ├── __init__.py
│   ├── data_utils.py                  # Data loading from APIs
│   ├── preprocessing.py               # Data cleaning and alignment
│   ├── models.py                      # Forecasting model classes
│   ├── analysis.py                    # Statistical analysis
│   └── visualization.py               # Plotting functions
│
├── data/                              # Persistent data storage (gitignored)
│   ├── raw/                          # Downloaded datasets
│   ├── processed/                    # Cleaned panel data
│   └── README.md                     # Data documentation
│
└── saved_models/                      # Model checkpoints (gitignored)
    └── README.md                      # Model documentation
```

## Why This Needs Studio Lab

| Requirement | Colab Free | Studio Lab |
|-------------|------------|------------|
| **10GB dataset** | ❌ No storage | ✅ 15GB persistent |
| **5-6 hour training** | ❌ 90 min limit | ✅ 12 hour sessions |
| **Checkpointing** | ❌ Lost on disconnect | ✅ Persists forever |
| **Environment setup** | ❌ Reinstall each time | ✅ Conda persists |
| **Resume analysis** | ❌ Start from scratch | ✅ Pick up where you left off |
| **Team sharing** | ❌ Copy/paste notebooks | ✅ Git integration |

**Bottom line:** This research workflow is not viable on Colab Free.

## Time Estimate

**First Run:**
- Setup: 15 minutes (one-time)
- Data download: 60 minutes (one-time, ~10GB)
- Environment setup: 10 minutes (one-time)
- Exploratory analysis: 45 minutes
- Model training: 5-6 hours
- Spillover analysis: 60 minutes
- Dashboard creation: 30 minutes
- **Total: 8-10 hours**

**Subsequent Runs:**
- Data: Instant (cached)
- Environment: Instant (persisted)
- Training: 5-6 hours (or resume from checkpoint)
- **Total: 5-7 hours**

You can pause and resume at any time!

## Model Details

### ARIMA (AutoRegressive Integrated Moving Average)
- Classical statistical forecasting
- Automatic order selection (auto_arima)
- Handles non-stationarity through differencing
- Best for univariate, stationary-transformed series

### VAR (Vector AutoRegression)
- Multivariate time series modeling
- Captures interdependencies between variables
- Impulse response functions for shock analysis
- Granger causality testing

### LSTM (Long Short-Term Memory)
- Deep learning for sequential data
- Captures long-term dependencies
- Handles non-linear patterns
- Best for complex, high-dimensional data

### Prophet
- Additive model by Facebook
- Handles missing data and outliers
- Built-in holiday effects
- Uncertainty intervals via simulation

### XGBoost
- Gradient boosting with engineered features
- Lagged variables, moving averages, trends
- Fast training and prediction
- Feature importance analysis

### Ensemble
- Weighted average of all models
- Weights optimized on validation set
- Reduces individual model errors
- More robust predictions

## Next Steps

After mastering Studio Lab:

- **Tier 2:** Introduction to AWS services (S3, Lambda, Athena, SageMaker) - $5-15
- **Tier 3:** Production infrastructure with CloudFormation - $50-500/month

## Resources

- [SageMaker Studio Lab Documentation](https://studiolab.sagemaker.aws)
- [FRED API Documentation](https://fred.stlouisfed.org/docs/api/fred/)
- [World Bank API](https://datahelpdesk.worldbank.org/knowledgebase/topics/125589)
- [OECD Data API](https://data.oecd.org/api/)
- [Forecasting: Principles and Practice](https://otexts.com/fpp3/)

## Troubleshooting

### Environment Issues
```bash
# Reset environment
conda env remove -n economics-studio-lab
conda env create -f environment.yml
```

### Storage Full
```bash
# Check usage
du -sh data/ saved_models/

# Clean old files
rm -rf data/raw/old_*.csv
rm -rf saved_models/backup_*.pkl
```

### Session Timeout
Data persists! Just restart and continue where you left off.

### API Rate Limits
The notebooks include rate limiting and caching to avoid API throttling.

---

**Built for SageMaker Studio Lab with [Claude Code](https://claude.com/claude-code)**
