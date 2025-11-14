# Multi-Sensor Precision Agriculture

**Duration:** 4-8 hours
**Platform:** SageMaker Studio Lab ONLY
**Cost:** $0 (free Studio Lab account, no AWS charges)
**Data:** ~10GB multi-sensor data (Sentinel-2, Landsat, MODIS, weather)

## Research Goal

Build ensemble yield prediction models using multi-sensor satellite data, weather information, and soil data. Perform temporal analysis across growing seasons to predict crop yields at field level with uncertainty quantification.

**This project requires Studio Lab** - it won't work on Colab Free due to:
- **10GB dataset** (Colab has no persistent storage)
- **5-6 hour continuous training** (Colab disconnects after 90 min idle)
- **Model checkpointing essential** (must resume from checkpoints)
- **Environment persistence** (complex dependency stack)

## What This Enables

Real research that isn't possible on Colab:

### Dataset Persistence
- Download 10GB of multi-sensor data **once**
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
   - Download Sentinel-2 imagery (10m resolution, 6 months)
   - Download Landsat-8 data (30m resolution, validation)
   - Download MODIS vegetation indices (250m, temporal context)
   - Fetch weather data (temperature, precipitation, GDD)
   - Cache in persistent storage (~10GB total)

2. **Data Preprocessing** (30-45 min)
   - Cloud masking and atmospheric correction
   - Multi-sensor co-registration
   - Calculate vegetation indices (NDVI, EVI, LAI)
   - Temporal aggregation and gap-filling
   - Generate training features

3. **Ensemble Model Training** (5-6 hours)
   - Train CNN for spatial patterns
   - Train LSTM for temporal dynamics
   - Train Random Forest for baseline
   - Train gradient boosting models
   - Ensemble combination with stacking
   - Checkpoint every epoch

4. **Yield Prediction & Analysis** (45-60 min)
   - Field-level yield predictions
   - Uncertainty quantification
   - Feature importance analysis
   - Temporal pattern visualization
   - Compare with ground truth

## Datasets

**Sentinel-2 (ESA)**
- Resolution: 10-20m
- Bands: RGB, NIR, Red Edge, SWIR
- Temporal: 5-day revisit
- Size: ~6GB (6-month time series)
- Source: AWS S3 Public Dataset

**Landsat-8 (USGS)**
- Resolution: 30m
- Bands: RGB, NIR, SWIR, Thermal
- Temporal: 16-day revisit
- Size: ~2GB (validation data)
- Source: AWS S3 Public Dataset

**MODIS (NASA)**
- Resolution: 250m
- Products: MOD13Q1 (NDVI/EVI)
- Temporal: 16-day composite
- Size: ~500MB
- Source: NASA EarthData

**Weather Data**
- Source: NOAA/PRISM
- Variables: Temp, Precip, GDD
- Temporal: Daily
- Size: ~100MB
- Format: NetCDF

**Soil Data**
- Source: SSURGO/gSSURGO
- Variables: Texture, organic matter, pH
- Resolution: Field-level polygons
- Size: ~50MB

## Setup

### 1. Create Studio Lab Account
1. Go to [SageMaker Studio Lab](https://studiolab.sagemaker.aws)
2. Request free account (approval in 1-2 days)
3. No credit card or AWS account needed

### 2. Import Project
```bash
# In Studio Lab terminal
git clone https://github.com/YOUR_USERNAME/research-jumpstart.git
cd research-jumpstart/projects/agriculture/precision-agriculture/tier-1
```

### 3. Install Dependencies
```bash
# Create conda environment (persists between sessions!)
conda env create -f environment.yml
conda activate agriculture-studio-lab

# Or use pip
pip install -r requirements.txt
```

### 4. Run Notebooks in Order
1. `01_data_acquisition.ipynb` - Download and cache datasets
2. `02_preprocessing.ipynb` - Process multi-sensor data
3. `03_feature_engineering.ipynb` - Create predictive features
4. `04_model_training.ipynb` - Train ensemble models
5. `05_yield_prediction.ipynb` - Generate predictions and analysis

## Key Features

### Persistence Example
```python
# Save model checkpoint (persists between sessions!)
model.save('saved_models/yield_cnn_epoch_20.h5')

# Load in next session
model = keras.models.load_model('saved_models/yield_cnn_epoch_20.h5')
```

### Multi-Sensor Data Fusion
```python
from src.data_utils import load_sentinel2, load_landsat8, load_modis

# Load cached data (instant after first download!)
s2_data = load_sentinel2(aoi='field_001', date_range='2023-04-01/2023-10-01')
l8_data = load_landsat8(aoi='field_001', date_range='2023-04-01/2023-10-01')
modis_data = load_modis(aoi='field_001', date_range='2023-04-01/2023-10-01')

# Fuse at 10m resolution
fused = fuse_sensors([s2_data, l8_data, modis_data], resolution=10)
```

### Temporal Analysis
```python
# Analyze crop phenology
phenology = extract_phenology_metrics(ndvi_timeseries)
# Returns: green-up, peak, senescence dates
```

## Project Structure

```
tier-1/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment
│
├── notebooks/
│   ├── 01_data_acquisition.ipynb      # Download satellite & weather data
│   ├── 02_preprocessing.ipynb         # Cloud masking, co-registration
│   ├── 03_feature_engineering.ipynb   # Vegetation indices, temporal features
│   ├── 04_model_training.ipynb        # Train ensemble models
│   └── 05_yield_prediction.ipynb      # Field-level predictions
│
├── src/
│   ├── __init__.py
│   ├── data_utils.py                  # Data loading utilities
│   ├── preprocessing.py               # Image processing functions
│   ├── features.py                    # Feature engineering
│   ├── models.py                      # Model architectures
│   └── visualization.py               # Plotting functions
│
├── data/                              # Persistent data storage (gitignored)
│   ├── raw/                          # Downloaded datasets
│   │   ├── sentinel2/
│   │   ├── landsat8/
│   │   ├── modis/
│   │   └── weather/
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
- **R² score:** 0.75-0.85 (field-level yield prediction)
- **RMSE:** 500-800 kg/ha (depends on crop type)
- **MAE:** 400-600 kg/ha

### Key Insights
- Identify critical growth stages for yield prediction
- Quantify impact of weather events on yield
- Optimize satellite revisit timing
- Compare sensor contributions

## Next Steps

After mastering Studio Lab:

- **Tier 2:** Introduction to AWS services (S3, Lambda, SageMaker) - $5-15
- **Tier 3:** Production infrastructure with CloudFormation - $50-500/month

## Resources

- [SageMaker Studio Lab Documentation](https://studiolab.sagemaker.aws/docs)
- [Sentinel-2 on AWS](https://registry.opendata.aws/sentinel-2/)
- [Landsat on AWS](https://registry.opendata.aws/landsat-8/)
- [MODIS Data](https://lpdaac.usgs.gov/products/mod13q1v006/)

## Troubleshooting

### Environment Issues
```bash
# Reset environment
conda env remove -n agriculture-studio-lab
conda env create -f environment.yml
```

### Storage Full
```bash
# Check usage
du -sh data/ saved_models/

# Clean old files
rm -rf data/raw/sentinel2/.old_*
rm -rf saved_models/*_epoch_[0-9].h5  # Keep only final models
```

### Session Timeout
Data persists! Just restart and continue where you left off.

### Data Download Errors
Check AWS S3 public dataset access. Fallback to local synthetic data for testing.

---

**Built for SageMaker Studio Lab with [Claude Code](https://claude.com/claude-code)**
