# Multi-Sensor Ocean Monitoring Ensemble

**Duration:** 4-8 hours
**Platform:** SageMaker Studio Lab ONLY
**Cost:** $0 (free Studio Lab account, no AWS charges)
**Data:** ~10GB multi-sensor ocean data

## Research Goal

Perform comprehensive ocean state monitoring using multi-sensor ensemble analysis. Integrate satellite ocean color data, Argo float profiles, and acoustic measurements to generate spatiotemporal ocean state predictions with uncertainty quantification.

**This project requires Studio Lab** - it won't work on Colab Free due to:
- **10GB dataset** (Colab has no persistent storage)
- **5-6 hour continuous training** (Colab disconnects after 90 min idle)
- **Model checkpointing essential** (must resume from checkpoints)
- **Environment persistence** (complex dependency stack)

## What This Enables

Real research that isn't possible on Colab:

### Ocean Data Persistence
- Download 10GB of multi-sensor ocean data **once**
- Access instantly in all future sessions
- No 20-minute re-downloads every session
- Cache intermediate processing results

### Long-Running Analysis
- Train 5-6 ensemble models (40-60 min each)
- Total compute: 5-6 hours continuous
- Automatic checkpointing every epoch
- Resume from checkpoint if needed

### Reproducible Environments
- Conda environment with 25+ packages
- Persists between sessions
- No reinstalling dependencies
- Team members use identical setup

### Iterative Ocean Analysis
- Save ensemble analysis results
- Build on previous runs
- Refine models incrementally
- Collaborative notebook development

## What You'll Build

Multi-notebook research workflow:

1. **Data Acquisition** (60 min)
   - Download satellite ocean color data (~4GB)
   - Download Argo float profiles (~3GB)
   - Download acoustic sensor data (~3GB)
   - Cache in persistent storage
   - Preprocess and align spatiotemporal grids

2. **Multi-Sensor Fusion** (90 min)
   - Align different sensor resolutions
   - Temporal interpolation and gap filling
   - Quality control and anomaly detection
   - Generate unified ocean state dataset

3. **Ensemble Model Training** (5-6 hours)
   - Train CNN for satellite ocean color
   - Train LSTM for Argo float time series
   - Train transformer for acoustic patterns
   - Ensemble learning with stacking
   - Checkpoint every epoch

4. **Spatiotemporal Analysis** (60 min)
   - Ocean state prediction with uncertainty
   - Anomaly detection (marine heatwaves, blooms)
   - Regional ocean health assessment
   - Temporal trend analysis

5. **Results Visualization** (30 min)
   - Interactive spatiotemporal maps
   - Ensemble prediction confidence
   - Multi-sensor comparison
   - Publication-ready figures

## Datasets

**Multi-Sensor Ocean Monitoring Suite**
- **Satellite Ocean Color:** MODIS-Aqua chlorophyll, SST (~4GB)
- **Argo Floats:** Temperature, salinity profiles (~3GB)
- **Acoustic Sensors:** Marine mammal, fish detection (~3GB)
- **Period:** 2015-2024
- **Region:** Pacific Ocean (customizable)
- **Total size:** ~10GB netCDF/CSV files
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
cd research-jumpstart/projects/marine-science/ocean-analysis/tier-1
```

### 3. Install Dependencies
```bash
# Create conda environment (persists between sessions!)
conda env create -f environment.yml
conda activate ocean-monitoring

# Or use pip
pip install -r requirements.txt
```

### 4. Run Notebooks in Order
1. `01_data_acquisition.ipynb` - Download and cache multi-sensor data
2. `02_sensor_fusion.ipynb` - Align and fuse sensor measurements
3. `03_ensemble_training.ipynb` - Train multi-model ensemble
4. `04_spatiotemporal_analysis.ipynb` - Ocean state prediction and trends
5. `05_visualization.ipynb` - Interactive ocean monitoring dashboard

## Key Features

### Persistence Example
```python
# Save ensemble checkpoint (persists between sessions!)
ensemble.save_checkpoint('models/ocean_ensemble_v1.pkl')

# Load in next session
ensemble = load_checkpoint('models/ocean_ensemble_v1.pkl')
```

### Longer Computations
```python
# Run intensive ocean modeling that would timeout on Colab
# Studio Lab: 12-hour GPU sessions, 4-hour CPU sessions
predictions = run_ensemble_forecast(sensors='all', duration=24)
```

### Shared Utilities
```python
# Import from project modules (better code organization)
from src.ocean_data import load_satellite_data, load_argo_floats
from src.fusion import align_spatiotemporal, fuse_sensors
from src.models import OceanCNN, ArgoLSTM, AcousticTransformer
```

## Project Structure

```
tier-1/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment
│
├── notebooks/
│   ├── 01_data_acquisition.ipynb      # Download multi-sensor data
│   ├── 02_sensor_fusion.ipynb         # Align and fuse sensors
│   ├── 03_ensemble_training.ipynb     # Train ensemble models
│   ├── 04_spatiotemporal_analysis.ipynb  # Ocean state analysis
│   └── 05_visualization.ipynb         # Interactive dashboard
│
├── src/
│   ├── __init__.py
│   ├── ocean_data.py                  # Data loading utilities
│   ├── fusion.py                      # Sensor fusion methods
│   ├── models.py                      # Neural network architectures
│   ├── ensemble.py                    # Ensemble learning
│   └── visualization.py               # Plotting functions
│
├── data/                              # Persistent data storage (gitignored)
│   ├── satellite/                     # MODIS ocean color
│   ├── argo/                          # Argo float profiles
│   ├── acoustic/                      # Acoustic sensor data
│   ├── processed/                     # Fused data
│   └── README.md                      # Data documentation
│
└── models/                            # Model checkpoints (gitignored)
    └── README.md                      # Model documentation
```

## Why This Needs Studio Lab

| Requirement | Colab Free | Studio Lab |
|-------------|------------|------------|
| **10GB multi-sensor data** | ❌ No storage | ✅ 15GB persistent |
| **5-6 hour training** | ❌ 90 min limit | ✅ 12 hour sessions |
| **Checkpointing** | ❌ Lost on disconnect | ✅ Persists forever |
| **Environment setup** | ❌ Reinstall each time | ✅ Conda persists |
| **Resume analysis** | ❌ Start from scratch | ✅ Pick up where you left off |
| **Team sharing** | ❌ Copy/paste notebooks | ✅ Git integration |

**Bottom line:** This ocean monitoring workflow is not viable on Colab Free.

## Time Estimate

**First Run:**
- Setup: 15 minutes (one-time)
- Data download: 60 minutes (one-time, ~10GB)
- Environment setup: 10 minutes (one-time)
- Sensor fusion: 90 minutes
- Model training: 5-6 hours
- Analysis: 1-2 hours
- **Total: 8-10 hours**

**Subsequent Runs:**
- Data: Instant (cached)
- Environment: Instant (persisted)
- Training: 5-6 hours (or resume from checkpoint)
- **Total: 5-6 hours**

You can pause and resume at any time!

## Next Steps

After mastering Studio Lab ocean monitoring:

- **Tier 2:** Introduction to AWS services (S3, Lambda, Athena) - $10-30
- **Tier 3:** Production infrastructure with CloudFormation - $50-500/month

## Resources

- [SageMaker Studio Lab Documentation](https://studiolab.sagemaker.aws/docs)
- [NOAA Ocean Data](https://www.ncei.noaa.gov/products/ocean)
- [Argo Float Data](https://argo.ucsd.edu/)
- [NASA Ocean Color](https://oceancolor.gsfc.nasa.gov/)
- [OBIS Marine Biodiversity](https://obis.org/)

## Troubleshooting

### Environment Issues
```bash
# Reset environment
conda env remove -n ocean-monitoring
conda env create -f environment.yml
```

### Storage Full
```bash
# Check usage
du -sh data/ models/

# Clean old files
rm -rf data/satellite/old_*.nc
```

### Session Timeout
Data persists! Just restart and continue where you left off.

---

**Built for SageMaker Studio Lab with [Claude Code](https://claude.com/claude-code)**
