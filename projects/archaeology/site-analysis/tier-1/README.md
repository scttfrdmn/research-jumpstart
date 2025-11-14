# Multi-Site Archaeological Ensemble Analysis

**Duration:** 4-8 hours
**Platform:** SageMaker Studio Lab ONLY
**Cost:** $0 (free Studio Lab account, no AWS charges)
**Data:** ~10GB multi-site archaeological data (imagery, LiDAR, geophysical)

## Research Goal

Perform cross-site comparative analysis using multi-modal archaeological data (artifact imagery, LiDAR terrain scans, geophysical surveys). Train ensemble deep learning models to identify patterns across excavation sites, quantify inter-site variability, and predict site characteristics from combined data sources.

**This project requires Studio Lab** - it won't work on Colab Free due to:
- **10GB multi-modal dataset** (Colab has no persistent storage)
- **5-6 hour continuous training** (Colab disconnects after 90 min idle)
- **Model checkpointing essential** (must resume from checkpoints)
- **Environment persistence** (complex dependency stack with archaeological tools)

## What This Enables

Real archaeological research that isn't possible on Colab:

### Dataset Persistence
- Download 10GB of multi-site data **once**
- Access instantly in all future sessions
- No 20-minute re-downloads every session
- Cache intermediate processing results (LiDAR preprocessing, feature extraction)

### Long-Running Training
- Train 5-6 ensemble models (30-40 min each)
- Total compute: 5-6 hours continuous
- Automatic checkpointing every epoch
- Resume from checkpoint if needed

### Reproducible Environments
- Conda environment with archaeological libraries
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
   - Download multi-site archaeological data (~10GB total)
   - Artifact imagery (5GB)
   - LiDAR terrain scans (3GB)
   - Geophysical survey data (2GB)
   - Cache in persistent storage
   - Preprocess and align coordinate systems

2. **Ensemble Model Training** (5-6 hours)
   - Train CNN for artifact classification (imagery)
   - Train terrain analysis model (LiDAR)
   - Train subsurface feature detection (geophysical)
   - Cross-modal fusion model
   - Transfer learning across sites
   - Checkpoint every epoch

3. **Cross-Site Comparative Analysis** (90 min)
   - Compare artifact assemblages across sites
   - Analyze settlement patterns from LiDAR
   - Identify subsurface features
   - Calculate inter-site similarity metrics
   - Cultural affiliation prediction

4. **Results Analysis** (45 min)
   - Visualize site relationships
   - Identify diagnostic artifacts
   - Map cultural boundaries
   - Publication-ready figures
   - Interactive 3D visualizations

## Datasets

**Multi-Site Archaeological Database**
- **Sites:** 6 excavation sites from different periods/cultures
- **Artifact imagery:** 5GB (10,000+ artifact photos)
- **LiDAR data:** 3GB (high-resolution terrain scans)
- **Geophysical surveys:** 2GB (ground-penetrating radar, magnetometry)
- **Metadata:** Site reports, artifact catalogs, radiocarbon dates
- **Total size:** ~10GB
- **Storage:** Cached in Studio Lab's 15GB persistent storage

### Site Coverage
- Site A: Neolithic settlement (5000-4000 BCE)
- Site B: Bronze Age fortification (2000-1500 BCE)
- Site C: Iron Age village (800-400 BCE)
- Site D: Roman villa (100-400 CE)
- Site E: Medieval monastery (1000-1400 CE)
- Site F: Post-medieval farmstead (1500-1800 CE)

## Setup

### 1. Create Studio Lab Account
1. Go to [SageMaker Studio Lab](https://studiolab.sagemaker.aws)
2. Request free account (approval in 1-2 days)
3. No credit card or AWS account needed

### 2. Import Project
```bash
# In Studio Lab terminal
git clone https://github.com/YOUR_USERNAME/research-jumpstart.git
cd research-jumpstart/projects/archaeology/site-analysis/tier-1
```

### 3. Install Dependencies
```bash
# Create conda environment (persists between sessions!)
conda env create -f environment.yml
conda activate archaeology-studio-lab

# Or use pip
pip install -r requirements.txt
```

### 4. Run Notebooks in Order
1. `01_data_preparation.ipynb` - Download and cache multi-site datasets
2. `02_artifact_imagery_analysis.ipynb` - Deep learning on artifact photos
3. `03_lidar_terrain_analysis.ipynb` - Settlement pattern detection
4. `04_geophysical_analysis.ipynb` - Subsurface feature identification
5. `05_ensemble_integration.ipynb` - Cross-modal fusion and comparative analysis
6. `06_interactive_visualization.ipynb` - 3D site visualizations and dashboards

## Key Features

### Persistence Example
```python
# Save model checkpoint (persists between sessions!)
model.save('saved_models/artifact_classifier_v1.h5')

# Save processed LiDAR data
lidar_features.to_netcdf('data/processed/site_a_lidar_features.nc')

# Load in next session
model = load_model('saved_models/artifact_classifier_v1.h5')
lidar_features = xr.open_dataset('data/processed/site_a_lidar_features.nc')
```

### Longer Computations
```python
# Run intensive computations that would timeout on Colab
# Studio Lab: 12-hour GPU sessions, 4-hour CPU sessions

# Process all sites (takes 3-4 hours)
for site in sites:
    artifact_features = extract_artifact_features(site.images)
    lidar_features = process_lidar_data(site.terrain)
    combined_features = integrate_modalities(artifact_features, lidar_features)
    results[site.name] = combined_features
```

### Shared Utilities
```python
# Import from project modules (better code organization)
from src.artifact_analysis import classify_artifacts, extract_features
from src.lidar_processing import process_terrain, detect_structures
from src.geophysical import process_gpr_data, detect_anomalies
from src.visualization import plot_site_comparison, create_3d_model
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
│   ├── 02_artifact_imagery_analysis.ipynb  # Artifact classification
│   ├── 03_lidar_terrain_analysis.ipynb     # Terrain and settlement analysis
│   ├── 04_geophysical_analysis.ipynb       # Subsurface features
│   ├── 05_ensemble_integration.ipynb       # Cross-modal integration
│   └── 06_interactive_visualization.ipynb  # 3D visualizations
│
├── src/
│   ├── __init__.py
│   ├── artifact_analysis.py          # Artifact classification utilities
│   ├── lidar_processing.py           # LiDAR data processing
│   ├── geophysical.py                # Geophysical survey analysis
│   ├── data_utils.py                 # Data loading utilities
│   └── visualization.py              # Plotting and 3D visualization
│
├── data/                              # Persistent data storage (gitignored)
│   ├── raw/                          # Downloaded datasets
│   │   ├── artifacts/                # Artifact imagery by site
│   │   ├── lidar/                    # LiDAR terrain data
│   │   └── geophysical/              # GPR and magnetometry
│   ├── processed/                    # Cleaned and aligned data
│   └── README.md                     # Data documentation
│
└── saved_models/                      # Model checkpoints (gitignored)
    ├── artifact_classifier.h5        # Trained artifact model
    ├── terrain_analyzer.h5           # Trained LiDAR model
    ├── geophysical_detector.h5       # Trained geophysical model
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
| **Archaeological tools** | Manual install | Pre-configured environment |

**Bottom line:** This multi-modal archaeological research workflow is not viable on Colab Free.

## Time Estimate

**First Run:**
- Setup: 15 minutes (one-time)
- Data download: 60 minutes (one-time, ~10GB)
- Environment setup: 15 minutes (one-time)
- Model training: 5-6 hours
- Analysis: 2-3 hours
- **Total: 8-10 hours**

**Subsequent Runs:**
- Data: Instant (cached)
- Environment: Instant (persisted)
- Training: 5-6 hours (or resume from checkpoint)
- **Total: 5-6 hours**

You can pause and resume at any time!

## Archaeological Research Applications

### Artifact Analysis
- Automated typology classification
- Chronological sequencing
- Cultural affiliation identification
- Diagnostic artifact detection

### Settlement Patterns
- LiDAR-based structure detection
- Site boundary mapping
- Landscape use analysis
- Inter-site visibility analysis

### Subsurface Investigation
- Ground-penetrating radar feature detection
- Magnetometry anomaly identification
- Excavation targeting
- Site preservation assessment

### Comparative Analysis
- Cross-site artifact similarity
- Cultural interaction patterns
- Trade network reconstruction
- Temporal change detection

## Next Steps

After mastering Studio Lab:

- **Tier 2:** Introduction to AWS services (S3, Lambda, SageMaker) - $10-25
  - Store 50GB+ archaeological datasets on S3
  - Distributed preprocessing with Lambda
  - Managed training with SageMaker
  - Model deployment for artifact identification API

- **Tier 3:** Production infrastructure with CloudFormation - $100-500/month
  - Multi-region archaeological databases (100GB+)
  - Distributed processing with AWS Batch
  - Real-time artifact identification service
  - Integration with museum collections databases
  - Full CloudFormation deployment

## Resources

- [SageMaker Studio Lab Documentation](https://studiolab.sagemaker.aws/docs)
- [Getting Started Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lab.html)
- [Open Context - Archaeological Data](https://opencontext.org/)
- [Digital Archaeological Record](https://www.tdar.org/)

## Troubleshooting

### Environment Issues
```bash
# Reset environment
conda env remove -n archaeology-studio-lab
conda env create -f environment.yml
```

### Storage Full
```bash
# Check usage
du -sh data/ saved_models/

# Clean old files
rm -rf data/processed/temp_*
rm -rf saved_models/old_*.h5
```

### Session Timeout
Data persists! Just restart and continue where you left off.

### LiDAR Processing Slow
```python
# Use chunked processing for large LiDAR files
lidar_data = process_lidar_chunked(filename, chunk_size=1000)
```

---

**Built for SageMaker Studio Lab with [Claude Code](https://claude.com/claude-code)**
