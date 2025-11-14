# Multi-City Mobility and Growth Ensemble

**Duration:** 4-8 hours
**Platform:** SageMaker Studio Lab ONLY
**Cost:** $0 (free Studio Lab account, no AWS charges)
**Data:** ~10GB multi-city dataset (imagery, mobility, demographics)

## Research Goal

Perform comprehensive urban dynamics analysis across multiple cities using ensemble models. Integrate satellite imagery, mobility patterns, and demographic data to predict urban growth, traffic patterns, and infrastructure needs. Compare urban development trajectories across diverse metropolitan areas.

**This project requires Studio Lab** - it won't work on Colab Free due to:
- **10GB multi-source dataset** (Colab has no persistent storage)
- **5-6 hour continuous training** (Colab disconnects after 90 min idle)
- **Model checkpointing essential** (must resume from checkpoints)
- **Environment persistence** (complex geospatial dependency stack)

## What This Enables

Real research that isn't possible on Colab:

### Dataset Persistence
- Download 10GB once (satellite imagery, mobility data, census)
- Access instantly in all future sessions
- No 20-minute re-downloads every session
- Cache intermediate processing results

### Long-Running Training
- Train 5-6 city-specific models (40-50 min each)
- Total compute: 5-6 hours continuous
- Automatic checkpointing every epoch
- Resume from checkpoint if needed

### Reproducible Environments
- Conda environment with geospatial stack (20+ packages)
- Persists between sessions
- No reinstalling GDAL/Fiona/Rasterio
- Team members use identical setup

### Iterative Analysis
- Save ensemble analysis results
- Build on previous runs
- Refine models incrementally
- Collaborative notebook development

## What You'll Build

Multi-notebook research workflow:

1. **Data Acquisition** (60 min)
   - Download imagery from 5-6 cities (~6GB total)
   - Mobility data from OpenStreetMap/transit APIs (~2GB)
   - Demographic data from Census Bureau (~2GB)
   - Cache in persistent storage
   - Preprocess and align spatial grids

2. **Urban Growth Modeling** (2-3 hours)
   - Train CNN for each city's expansion patterns
   - Transfer learning from base architecture
   - Checkpoint every epoch
   - Parallel training workflows

3. **Mobility Pattern Analysis** (1-2 hours)
   - Traffic flow prediction models
   - Public transit accessibility
   - Commute pattern analysis
   - Infrastructure capacity assessment

4. **Ensemble Analysis** (1-2 hours)
   - Cross-city comparative analysis
   - Identify common growth patterns
   - Quantify urban development disparities
   - Policy scenario modeling

5. **Results Visualization** (45 min)
   - Interactive dashboards
   - Spatial-temporal animations
   - Comparative city rankings
   - Publication-ready figures

## Datasets

**Multi-City Urban Analysis Suite**
- **Cities:** 5-6 diverse metros (e.g., Austin, Denver, Phoenix, Portland, Seattle, Charlotte)
- **Satellite imagery:** Landsat 8/9, 30m resolution (2000-2024)
- **Mobility data:** OpenStreetMap roads, GTFS transit, traffic counts
- **Demographics:** Census ACS 5-year estimates (population, income, employment)
- **Period:** 2000-2024 (24 years)
- **Total size:** ~10GB compressed
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
cd research-jumpstart/projects/urban-planning/city-analytics/tier-1
```

### 3. Install Dependencies
```bash
# Create conda environment (persists between sessions!)
conda env create -f environment.yml
conda activate urban-studio-lab

# Or use pip
pip install -r requirements.txt
```

### 4. Run Notebooks in Order
1. `01_data_acquisition.ipynb` - Download and cache all datasets
2. `02_urban_growth_models.ipynb` - Train city-specific growth models
3. `03_mobility_analysis.ipynb` - Analyze traffic and transit patterns
4. `04_ensemble_comparison.ipynb` - Cross-city comparative analysis
5. `05_interactive_dashboard.ipynb` - Create interactive visualizations

## Key Features

### Persistence Example
```python
# Save model checkpoint (persists between sessions!)
model.save('saved_models/austin_growth_model_v1.h5')

# Load in next session
model = load_model('saved_models/austin_growth_model_v1.h5')
```

### Longer Computations
```python
# Run intensive computations that would timeout on Colab
# Studio Lab: 12-hour GPU sessions, 4-hour CPU sessions
results = train_multi_city_ensemble(cities=6, epochs=100)
```

### Shared Utilities
```python
# Import from project modules (better code organization)
from src.data_utils import load_satellite_imagery, load_mobility_data
from src.urban_models import UrbanGrowthCNN, MobilityPredictor
from src.visualization import create_city_comparison_dashboard
```

## Project Structure

```
tier-1/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment
│
├── notebooks/
│   ├── 01_data_acquisition.ipynb      # Download and cache datasets
│   ├── 02_urban_growth_models.ipynb   # Train growth prediction models
│   ├── 03_mobility_analysis.ipynb     # Traffic and transit analysis
│   ├── 04_ensemble_comparison.ipynb   # Cross-city comparison
│   └── 05_interactive_dashboard.ipynb # Interactive visualizations
│
├── src/
│   ├── __init__.py
│   ├── data_utils.py                  # Data loading utilities
│   ├── urban_models.py                # ML model architectures
│   ├── mobility_analysis.py           # Mobility metrics
│   └── visualization.py               # Plotting functions
│
├── data/                              # Persistent data storage (gitignored)
│   ├── raw/                          # Downloaded datasets
│   │   ├── imagery/                  # Satellite imagery by city
│   │   ├── mobility/                 # OSM and transit data
│   │   └── demographics/             # Census data
│   ├── processed/                    # Cleaned data
│   └── README.md                     # Data documentation
│
└── saved_models/                      # Model checkpoints (gitignored)
    └── README.md                      # Model documentation
```

## Why This Needs Studio Lab

| Requirement | Colab Free | Studio Lab |
|-------------|------------|------------|
| **10GB dataset** | No storage | 15GB persistent |
| **5-6 hour training** | 90 min limit | 12 hour sessions |
| **Checkpointing** | Lost on disconnect | Persists forever |
| **Geospatial stack** | Reinstall each time | Conda persists |
| **Resume analysis** | Start from scratch | Pick up where you left off |
| **Team sharing** | Copy/paste notebooks | Git integration |

**Bottom line:** This research workflow is not viable on Colab Free.

## Time Estimate

**First Run:**
- Setup: 15 minutes (one-time)
- Data download: 60 minutes (one-time, ~10GB)
- Environment setup: 20 minutes (one-time, geospatial packages)
- Growth models: 2-3 hours
- Mobility analysis: 1-2 hours
- Ensemble analysis: 1-2 hours
- **Total: 6-9 hours**

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
conda env remove -n urban-studio-lab
conda env create -f environment.yml
```

### Storage Full
```bash
# Check usage
du -sh data/ saved_models/

# Clean old files
rm -rf data/raw/imagery/old_*.tif
```

### Session Timeout
Data persists! Just restart and continue where you left off.

---

**Built for SageMaker Studio Lab with [Claude Code](https://claude.com/claude-code)**
