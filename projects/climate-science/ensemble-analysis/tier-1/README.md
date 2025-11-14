# Multi-Model CMIP6 Ensemble Analysis

**Duration:** 4-6 hours
**Platform:** SageMaker Studio Lab ONLY
**Cost:** $0 (free Studio Lab account, no AWS charges)
**Data:** ~10GB CMIP6 multi-model ensemble

## Research Goal

Perform uncertainty quantification on future climate projections using a 10-model CMIP6 ensemble. Train deep learning emulators for each model, generate probabilistic forecasts, and quantify inter-model variability for regional climate impacts.

**This project requires Studio Lab** - it won't work on Colab Free due to:
- **10GB dataset** (Colab has no persistent storage)
- **5-6 hour continuous training** (Colab disconnects after 90 min idle)
- **Model checkpointing essential** (must resume from checkpoints)
- **Environment persistence** (complex dependency stack)

## What This Enables

Real research that isn't possible on Colab:

### 🔬 Dataset Persistence
- Download 10GB of CMIP6 data **once**
- Access instantly in all future sessions
- No 20-minute re-downloads every session
- Cache intermediate processing results

### ⚡ Long-Running Training
- Train 10 model emulators (30-40 min each)
- Total compute: 5-6 hours continuous
- Automatic checkpointing every epoch
- Resume from checkpoint if needed

### 🧪 Reproducible Environments
- Conda environment with 20+ packages
- Persists between sessions
- No reinstalling dependencies
- Team members use identical setup

### 📊 Iterative Analysis
- Save ensemble analysis results
- Build on previous runs
- Refine models incrementally
- Collaborative notebook development

## What You'll Build

Multi-notebook research workflow:

1. **Data Acquisition** (45 min)
   - Download 10 CMIP6 models (~10GB total)
   - Cache in persistent storage
   - Preprocess and align grids
   - Generate training/validation splits

2. **Ensemble Emulator Training** (5-6 hours)
   - Train CNN emulator for each of 10 models
   - Transfer learning from base architecture
   - Checkpoint every epoch
   - Parallel training workflows

3. **Uncertainty Quantification** (45 min)
   - Generate ensemble probabilistic forecasts
   - Calculate inter-model variance
   - Regional impact analysis
   - Confidence intervals for 2050-2100

4. **Results Analysis** (30 min)
   - Compare model performance
   - Identify consensus projections
   - Quantify disagreement regions
   - Publication-ready figures

## Datasets

**CMIP6 Multi-Model Ensemble**
- **Models:** 10 (CESM2, UKESM, IPSL, MPI, CNRM, ACCESS, GFDL, MIROC, NorESM, CanESM)
- **Scenario:** SSP2-4.5 (moderate emissions)
- **Variables:** Temperature, precipitation, humidity
- **Period:** 1950-2100
- **Resolution:** 1° × 1° (100km grid)
- **Total size:** ~10GB netCDF files
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
cd research-jumpstart/projects/climate-science/ensemble-analysis/tier-1
```

### 3. Install Dependencies
```bash
# Create conda environment (persists between sessions!)
conda env create -f environment.yml
conda activate climate-studio-lab

# Or use pip
pip install -r requirements.txt
```

### 4. Run Notebooks in Order
1. `01_data_preparation.ipynb` - Download and cache datasets
2. `02_multi_variable_analysis.ipynb` - Analyze multiple climate variables
3. `03_forecasting_models.ipynb` - Build and save predictive models
4. `04_interactive_dashboard.ipynb` - Create interactive visualizations

## Key Features

### Persistence Example
```python
# Save model checkpoint (persists between sessions!)
model.save('saved_models/climate_forecast_v1.pkl')

# Load in next session
model = load_model('saved_models/climate_forecast_v1.pkl')
```

### Longer Computations
```python
# Run intensive computations that would timeout on Colab
# Studio Lab: 12-hour GPU sessions, 4-hour CPU sessions
results = run_monte_carlo_simulation(n_iterations=100000)
```

### Shared Utilities
```python
# Import from project modules (better code organization)
from src.data_utils import load_climate_data, calculate_anomalies
from src.visualization import create_interactive_dashboard
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
│   ├── 02_multi_variable_analysis.ipynb  # Multi-variable analysis
│   ├── 03_forecasting_models.ipynb    # Predictive modeling
│   └── 04_interactive_dashboard.ipynb # Interactive visualizations
│
├── src/
│   ├── __init__.py
│   ├── data_utils.py                  # Data loading utilities
│   ├── analysis.py                    # Analysis functions
│   └── visualization.py               # Plotting functions
│
├── data/                              # Persistent data storage (gitignored)
│   ├── raw/                          # Downloaded datasets
│   ├── processed/                    # Cleaned data
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
- Data download: 45 minutes (one-time, ~10GB)
- Environment setup: 10 minutes (one-time)
- Model training: 5-6 hours
- Analysis: 1-2 hours
- **Total: 7-9 hours**

**Subsequent Runs:**
- Data: Instant (cached)
- Environment: Instant (persisted)
- Training: 5-6 hours (or resume from checkpoint)
- **Total: 5-6 hours**

You can pause and resume at any time!

## Next Steps

After mastering Studio Lab:

- **Tier 2:** Introduction to AWS services (S3, Lambda, Athena) - $5-15
- **Tier 3:** Production infrastructure with CloudFormation - $50-500/month

## Resources

- [SageMaker Studio Lab Documentation](https://studiolab.sagemaker.aws/docs)
- [Getting Started Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lab.html)
- [Community Forum](https://github.com/aws/studio-lab-examples)

## Troubleshooting

### Environment Issues
```bash
# Reset environment
conda env remove -n climate-studio-lab
conda env create -f environment.yml
```

### Storage Full
```bash
# Check usage
du -sh data/ saved_models/

# Clean old files
rm -rf data/raw/old_*.nc
```

### Session Timeout
Data persists! Just restart and continue where you left off.

---

**🤖 Built for SageMaker Studio Lab with [Claude Code](https://claude.com/claude-code)**
