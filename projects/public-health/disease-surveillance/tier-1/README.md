# Multi-Disease Ensemble Surveillance System

**Duration:** 4-8 hours
**Platform:** SageMaker Studio Lab ONLY
**Cost:** $0 (free Studio Lab account, no AWS charges)
**Data:** ~10GB multi-disease, multi-region surveillance data

## Research Goal

Build an ensemble forecasting system for multiple diseases (influenza, COVID-19, RSV, etc.) across multiple regions. Train deep learning models (LSTM ensembles) for each disease-region pair, perform spatiotemporal analysis, and generate probabilistic forecasts with uncertainty quantification.

**This project requires Studio Lab** - it won't work on Colab Free due to:
- **10GB dataset** (Colab has no persistent storage)
- **5-6 hour continuous training** (Colab disconnects after 90 min idle)
- **Model checkpointing essential** (must resume from checkpoints)
- **Environment persistence** (complex dependency stack)

## What This Enables

Real research that isn't possible on Colab:

### Dataset Persistence
- Download 10GB of multi-disease surveillance data **once**
- Access instantly in all future sessions
- No 15-20 minute re-downloads every session
- Cache intermediate processing results

### Long-Running Training
- Train 5-6 disease-specific LSTM ensembles (40-60 min each)
- Total compute: 4-6 hours continuous
- Automatic checkpointing every epoch
- Resume from checkpoint if needed

### Reproducible Environments
- Conda environment with 25+ packages
- Persists between sessions
- No reinstalling dependencies
- Team members use identical setup

### Iterative Analysis
- Save ensemble analysis results
- Build on previous runs
- Refine models incrementally
- Collaborative notebook development

## What You'll Build

Multi-notebook research workflow:

1. **Data Acquisition** (60 min)
   - Download CDC, WHO, state-level surveillance data (~10GB total)
   - Multiple diseases: ILI, COVID-19, RSV, norovirus, pertussis
   - Multiple regions: US states, WHO regions, counties
   - Cache in persistent storage
   - Preprocess and align temporal grids
   - Generate training/validation splits

2. **Spatiotemporal Feature Engineering** (45 min)
   - Lag features (1-4 weeks)
   - Geographic neighbor features
   - Population mobility patterns
   - Seasonal decomposition
   - Holiday and event indicators

3. **Ensemble Model Training** (4-6 hours)
   - Train LSTM ensemble for each disease (5-6 diseases)
   - Multi-region forecasting models
   - Transfer learning across diseases
   - Checkpoint every epoch
   - Parallel training workflows

4. **Uncertainty Quantification** (60 min)
   - Generate ensemble probabilistic forecasts
   - Calculate inter-model variance
   - Regional outbreak risk assessment
   - Confidence intervals for 1-4 week forecasts

5. **Interactive Dashboard** (30 min)
   - Real-time forecast visualization
   - Outbreak alert system
   - Model performance comparison
   - Publication-ready figures

## Datasets

**Multi-Disease Surveillance Ensemble**
- **Diseases:** 5-6 (ILI, COVID-19, RSV, Norovirus, Pertussis, Measles)
- **Sources:** CDC FluView, CDC COVID Data Tracker, WHO, State Health Departments
- **Regions:** 50 US states + 10 countries
- **Variables:** Cases, deaths, hospitalizations, test positivity, vaccination rates
- **Period:** 2015-2024 (weekly aggregated)
- **Resolution:** Weekly by region
- **Total size:** ~10GB CSV files
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
cd research-jumpstart/projects/public-health/disease-surveillance/tier-1
```

### 3. Install Dependencies
```bash
# Create conda environment (persists between sessions!)
conda env create -f environment.yml
conda activate epi-studio-lab

# Or use pip
pip install -r requirements.txt
```

### 4. Run Notebooks in Order
1. `01_data_preparation.ipynb` - Download and cache datasets
2. `02_feature_engineering.ipynb` - Spatiotemporal features
3. `03_ensemble_training.ipynb` - Train LSTM models (4-6 hours)
4. `04_forecast_evaluation.ipynb` - Evaluate and compare forecasts
5. `05_interactive_dashboard.ipynb` - Visualization and alerts

## Key Features

### Persistence Example
```python
# Save model checkpoint (persists between sessions!)
model.save('saved_models/ili_forecast_lstm_v1.h5')

# Load in next session
from tensorflow import keras
model = keras.models.load_model('saved_models/ili_forecast_lstm_v1.h5')
```

### Longer Computations
```python
# Run intensive computations that would timeout on Colab
# Studio Lab: 12-hour GPU sessions, 4-hour CPU sessions
ensemble_results = train_multi_disease_ensemble(
    diseases=['ILI', 'COVID-19', 'RSV', 'Norovirus', 'Pertussis'],
    regions=us_states,
    epochs=50
)
```

### Shared Utilities
```python
# Import from project modules (better code organization)
from src.data_utils import load_surveillance_data, calculate_epidemic_threshold
from src.forecasting import train_lstm_ensemble, generate_probabilistic_forecast
from src.visualization import create_forecast_dashboard, plot_outbreak_heatmap
```

## Project Structure

```
tier-1/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment
│
├── notebooks/
│   ├── 01_data_preparation.ipynb      # Download and cache data
│   ├── 02_feature_engineering.ipynb   # Spatiotemporal features
│   ├── 03_ensemble_training.ipynb     # Train LSTM models
│   ├── 04_forecast_evaluation.ipynb   # Evaluate forecasts
│   └── 05_interactive_dashboard.ipynb # Dashboards and alerts
│
├── src/
│   ├── __init__.py
│   ├── data_utils.py                  # Data loading utilities
│   ├── feature_engineering.py         # Feature creation
│   ├── forecasting.py                 # LSTM training/prediction
│   ├── evaluation.py                  # Model evaluation metrics
│   └── visualization.py               # Plotting functions
│
├── data/                              # Persistent data storage (gitignored)
│   ├── raw/                          # Downloaded datasets
│   │   ├── cdc_ili/
│   │   ├── cdc_covid/
│   │   ├── cdc_rsv/
│   │   └── who_data/
│   ├── processed/                    # Cleaned data
│   └── README.md                     # Data documentation
│
└── saved_models/                      # Model checkpoints (gitignored)
    ├── ili_lstm_ensemble/
    ├── covid_lstm_ensemble/
    └── README.md                      # Model documentation
```

## Why This Needs Studio Lab

| Requirement | Colab Free | Studio Lab |
|-------------|------------|------------|
| **10GB dataset** | No storage | 15GB persistent |
| **5-6 hour training** | 90 min limit | 12 hour sessions |
| **Checkpointing** | Lost on disconnect | Persists forever |
| **Environment setup** | Reinstall each time | Conda persists |
| **Resume analysis** | Start from scratch | Pick up where you left off |
| **Team sharing** | Copy/paste notebooks | Git integration |

**Bottom line:** Multi-disease ensemble surveillance is not viable on Colab Free.

## Time Estimate

**First Run:**
- Setup: 15 minutes (one-time)
- Data download: 60 minutes (one-time, ~10GB)
- Environment setup: 10 minutes (one-time)
- Feature engineering: 45 minutes
- Model training: 4-6 hours
- Evaluation & visualization: 1-2 hours
- **Total: 7-9 hours**

**Subsequent Runs:**
- Data: Instant (cached)
- Environment: Instant (persisted)
- Training: 4-6 hours (or resume from checkpoint)
- **Total: 4-6 hours**

You can pause and resume at any time!

## Model Architecture

### LSTM Ensemble
```python
# Each disease gets an ensemble of 3-5 LSTM models
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(lookback_weeks, n_features)),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(forecast_horizon)  # 1-4 week forecasts
])
```

### Ensemble Aggregation
- Mean prediction across models
- Weighted by recent performance
- Uncertainty from model variance
- Probabilistic forecast intervals

## Evaluation Metrics

### Forecast Accuracy
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- R² Score

### Epidemiological Metrics
- Peak timing accuracy
- Peak intensity accuracy
- Epidemic onset detection
- Duration prediction

### Probabilistic Metrics
- Prediction interval coverage
- Continuous Ranked Probability Score (CRPS)
- Brier Score for binary outbreak prediction

## Next Steps

After mastering Studio Lab:

- **Tier 2:** Introduction to AWS services (S3, Lambda, SageMaker) - $5-15
- **Tier 3:** Production surveillance infrastructure with CloudFormation - $50-500/month

## Resources

- [CDC FluView API](https://www.cdc.gov/flu/weekly/overview.htm)
- [COVID-19 Data Tracker](https://covid.cdc.gov/covid-data-tracker/)
- [SageMaker Studio Lab Documentation](https://studiolab.sagemaker.aws/docs)
- [Epidemic Forecasting: A Primer](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007271)

## Troubleshooting

### Environment Issues
```bash
# Reset environment
conda env remove -n epi-studio-lab
conda env create -f environment.yml
```

### Storage Full
```bash
# Check usage
du -sh data/ saved_models/

# Clean old files
rm -rf data/raw/old_*.csv
rm -rf saved_models/checkpoints/old_*
```

### Session Timeout
Data persists! Just restart and continue where you left off.

### Training Interrupted
```python
# Resume from last checkpoint
model = load_model('saved_models/checkpoints/ili_lstm_epoch_25.h5')
# Continue training from epoch 26
```

---

**Built for SageMaker Studio Lab with [Claude Code](https://claude.com/claude-code)**
