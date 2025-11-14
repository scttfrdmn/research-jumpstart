# Multi-Grid Renewable Integration Ensemble

**Duration:** 4-8 hours
**Platform:** SageMaker Studio Lab ONLY
**Cost:** $0 (free Studio Lab account, no AWS charges)
**Data:** ~10GB multi-grid data (load, solar, wind, storage)

## Research Goal

Build ensemble forecasting models for renewable energy integration across multiple grid zones. Predict electricity demand, solar generation, wind power output, and battery storage needs with uncertainty quantification. Perform grid stability analysis to optimize renewable dispatch and prevent frequency deviations.

**This project requires Studio Lab** - it won't work on Colab Free due to:
- **10GB dataset** (Colab has no persistent storage)
- **5-6 hour continuous training** (Colab disconnects after 90 min idle)
- **Model checkpointing essential** (must resume from checkpoints)
- **Environment persistence** (complex dependency stack)

## What This Enables

Real research that isn't possible on Colab:

### Dataset Persistence
- Download 10GB of multi-grid data **once**
- Access instantly in all future sessions
- No 20-minute re-downloads every session
- Cache intermediate processing results

### Long-Running Training
- Train ensemble models (5-6 hours continuous)
- Automatic checkpointing every epoch
- Resume from checkpoint if needed
- Multiple model architectures in parallel

### Reproducible Environments
- Conda environment with 20+ packages
- Persists between sessions
- No reinstalling dependencies
- Team members use identical setup

### Iterative Analysis
- Save model predictions and results
- Build on previous experiments
- Refine models incrementally
- Collaborative notebook development

## What You'll Build

Multi-notebook research workflow:

1. **Data Acquisition** (45-60 min)
   - Download smart meter load data (5 grid zones, 2 years)
   - Download solar irradiance and PV generation data
   - Download wind speed and turbine generation data
   - Fetch weather forecasts (GFS, HRRR models)
   - Download battery storage state-of-charge data
   - Cache in persistent storage (~10GB total)

2. **Data Preprocessing** (30-45 min)
   - Time zone normalization across grid regions
   - Missing data imputation and outlier detection
   - Feature engineering (lag features, rolling statistics)
   - Calculate net load (demand - renewable generation)
   - Grid stability metrics (frequency, voltage)

3. **Ensemble Model Training** (5-6 hours)
   - Train LSTM for load forecasting
   - Train CNN-LSTM for solar generation
   - Train Transformer for wind power
   - Train XGBoost for battery dispatch
   - Train attention-based net load model
   - Ensemble combination with stacking
   - Checkpoint every epoch

4. **Grid Stability Analysis** (45-60 min)
   - Multi-zone load balance predictions
   - Renewable penetration scenarios (20%-80%)
   - Frequency regulation requirements
   - Ramping event detection
   - Battery scheduling optimization
   - Uncertainty quantification with prediction intervals

## Datasets

**Load Data (Smart Meters)**
- Resolution: 15-minute intervals
- Zones: 5 grid balancing areas
- Duration: 2 years historical
- Size: ~4GB
- Source: ISO/RTO public datasets
- Variables: Active power, reactive power

**Solar Generation Data**
- Resolution: 5-minute intervals
- Sites: 50+ PV installations (1MW-100MW)
- Duration: 2 years historical
- Size: ~2GB
- Source: NREL WIND Toolkit / PVDAQ
- Variables: GHI, DNI, DHI, power output

**Wind Power Data**
- Resolution: 10-minute intervals
- Sites: 30+ wind farms (10MW-500MW)
- Duration: 2 years historical
- Size: ~2GB
- Source: NREL WIND Toolkit
- Variables: Wind speed, direction, power output

**Weather Forecasts**
- Resolution: Hourly
- Models: GFS (global), HRRR (regional)
- Duration: 2 years historical + forecasts
- Size: ~1.5GB
- Source: NOAA
- Variables: Temp, wind, radiation, precipitation

**Battery Storage Data**
- Resolution: 1-minute intervals
- Sites: 10+ grid-scale batteries (10MWh-100MWh)
- Duration: 1 year historical
- Size: ~500MB
- Source: Utility SCADA systems (synthetic)
- Variables: SOC, charge/discharge power, efficiency

## Setup

### 1. Create Studio Lab Account
1. Go to [SageMaker Studio Lab](https://studiolab.sagemaker.aws)
2. Request free account (approval in 1-2 days)
3. No credit card or AWS account needed

### 2. Import Project
```bash
# In Studio Lab terminal
git clone https://github.com/YOUR_USERNAME/research-jumpstart.git
cd research-jumpstart/projects/energy-systems/grid-optimization/tier-1
```

### 3. Install Dependencies
```bash
# Create conda environment (persists between sessions!)
conda env create -f environment.yml
conda activate energy-studio-lab

# Or use pip
pip install -r requirements.txt
```

### 4. Run Notebooks in Order
1. `01_data_acquisition.ipynb` - Download and cache datasets
2. `02_preprocessing.ipynb` - Process multi-grid time series
3. `03_feature_engineering.ipynb` - Create temporal and weather features
4. `04_model_training.ipynb` - Train ensemble forecasting models
5. `05_grid_stability.ipynb` - Analyze renewable integration scenarios

## Key Features

### Persistence Example
```python
# Save model checkpoint (persists between sessions!)
model.save('saved_models/load_lstm_epoch_30.h5')

# Load in next session
model = keras.models.load_model('saved_models/load_lstm_epoch_30.h5')
```

### Multi-Grid Data Fusion
```python
from src.data_utils import load_grid_data, load_renewable_data

# Load cached data (instant after first download!)
load_data = load_grid_data(zones=['zone1', 'zone2'],
                           date_range='2022-01-01/2023-12-31')
solar_data = load_renewable_data(source='solar',
                                  date_range='2022-01-01/2023-12-31')
wind_data = load_renewable_data(source='wind',
                                 date_range='2022-01-01/2023-12-31')

# Calculate net load
net_load = load_data - (solar_data + wind_data)
```

### Grid Stability Analysis
```python
# Analyze renewable penetration impact
stability_metrics = analyze_grid_stability(
    load_forecast,
    renewable_forecast,
    penetration_levels=[0.2, 0.4, 0.6, 0.8]
)
# Returns: frequency deviation, ramping requirements, curtailment
```

## Project Structure

```
tier-1/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment
│
├── notebooks/
│   ├── 01_data_acquisition.ipynb      # Download grid & renewable data
│   ├── 02_preprocessing.ipynb         # Time series cleaning, alignment
│   ├── 03_feature_engineering.ipynb   # Temporal & weather features
│   ├── 04_model_training.ipynb        # Train ensemble models
│   └── 05_grid_stability.ipynb        # Renewable integration analysis
│
├── src/
│   ├── __init__.py
│   ├── data_utils.py                  # Data loading utilities
│   ├── preprocessing.py               # Time series processing
│   ├── features.py                    # Feature engineering
│   ├── models.py                      # Model architectures
│   └── visualization.py               # Plotting functions
│
├── data/                              # Persistent data storage (gitignored)
│   ├── raw/                          # Downloaded datasets
│   │   ├── load/
│   │   ├── solar/
│   │   ├── wind/
│   │   ├── weather/
│   │   └── storage/
│   ├── processed/                    # Processed data
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
- Data download: 45-60 minutes (one-time, ~10GB)
- Environment setup: 10 minutes (one-time)
- Preprocessing: 30-45 minutes
- Model training: 5-6 hours
- Analysis: 45-60 minutes
- **Total: 8-9 hours**

**Subsequent Runs:**
- Data: Instant (cached)
- Environment: Instant (persisted)
- Preprocessing: Skip (cached results)
- Training: 5-6 hours (or resume from checkpoint)
- **Total: 5-6 hours**

You can pause and resume at any time!

## Expected Results

### Model Performance
- **Load forecasting MAE:** 2-5% of peak demand
- **Solar forecast RMSE:** 10-15% of installed capacity
- **Wind forecast RMSE:** 15-20% of installed capacity
- **Net load MAE:** 3-7% of peak demand

### Grid Stability Insights
- Identify critical ramping events (>500MW/hour)
- Quantify frequency regulation needs with high renewables
- Optimize battery dispatch schedules
- Predict curtailment requirements
- Assess grid flexibility requirements

## Next Steps

After mastering Studio Lab:

- **Tier 2:** Introduction to AWS services (S3, Lambda, SageMaker) - $5-15
- **Tier 3:** Production infrastructure with CloudFormation - $50-500/month

## Resources

- [SageMaker Studio Lab Documentation](https://studiolab.sagemaker.aws)
- [NREL WIND Toolkit](https://www.nrel.gov/grid/wind-toolkit.html)
- [NREL PVDAQ](https://www.nrel.gov/grid/solar-power-data.html)
- [NOAA Weather Data](https://www.ncei.noaa.gov/)

## Troubleshooting

### Environment Issues
```bash
# Reset environment
conda env remove -n energy-studio-lab
conda env create -f environment.yml
```

### Storage Full
```bash
# Check usage
du -sh data/ saved_models/

# Clean old files
rm -rf data/raw/load/.old_*
rm -rf saved_models/*_epoch_[0-9].h5  # Keep only final models
```

### Session Timeout
Data persists! Just restart and continue where you left off.

### Data Download Errors
Check data source availability. Fallback to synthetic grid data for testing.

---

**Built for SageMaker Studio Lab with [Claude Code](https://claude.com/claude-code)**
