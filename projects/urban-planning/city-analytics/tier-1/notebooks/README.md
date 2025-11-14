# Urban Planning Analysis Notebooks

This directory contains Jupyter notebooks for multi-city urban analysis.

## Notebook Sequence

Run notebooks in this order for complete analysis:

### 1. `01_data_acquisition.ipynb`
**Duration:** 60 minutes
**Purpose:** Download and cache all datasets

- Download satellite imagery for 5-6 cities (~6GB)
- Fetch OpenStreetMap mobility data (~2GB)
- Pull Census demographic data (~2GB)
- Preprocess and cache all data
- Validate data quality

**Output:** Cached datasets in `../data/` directory

---

### 2. `02_urban_growth_models.ipynb`
**Duration:** 2-3 hours
**Purpose:** Train city-specific urban growth models

- Load cached imagery time series
- Build CNN architecture for each city
- Train models with checkpointing (40-50 min per city)
- Evaluate spatial prediction accuracy
- Generate growth forecasts for 2025-2035

**Output:** Trained models in `../saved_models/` directory

---

### 3. `03_mobility_analysis.ipynb`
**Duration:** 1-2 hours
**Purpose:** Analyze traffic and transit patterns

- Calculate traffic flow metrics (AADT, V/C ratios)
- Assess transit accessibility coverage
- Compute commute pattern statistics
- Identify service gaps and congestion hotspots
- Compare mobility across cities

**Output:** Mobility metrics and visualizations

---

### 4. `04_ensemble_comparison.ipynb`
**Duration:** 1-2 hours
**Purpose:** Cross-city comparative analysis

- Compare urban growth trajectories
- Identify common development patterns
- Quantify disparities in growth rates
- Analyze correlation between growth and mobility
- Policy scenario modeling

**Output:** Comparative analysis results

---

### 5. `05_interactive_dashboard.ipynb`
**Duration:** 45 minutes
**Purpose:** Create interactive visualizations

- Build interactive city comparison dashboard
- Generate spatial-temporal animations
- Create mobility heatmaps
- Export publication-ready figures
- Deploy Plotly/Folium dashboards

**Output:** Interactive visualizations and dashboards

---

## Usage Tips

### Checkpointing
All notebooks support resuming from checkpoints:
```python
# Training automatically saves checkpoints
model = train_city_model('austin', imagery, labels, checkpoint_dir)

# Resume from checkpoint if interrupted
if checkpoint_exists('austin'):
    model = load_checkpoint('austin')
```

### Data Caching
Data is cached after first download:
```python
# First run: downloads data
imagery = load_satellite_imagery('austin', 2024)  # ~20 min

# Subsequent runs: instant
imagery = load_satellite_imagery('austin', 2024)  # <1 sec
```

### Memory Management
For large datasets, use lazy loading:
```python
# Load imagery on-demand
imagery_generator = create_imagery_generator(cities, years)

# Process in batches
for batch in imagery_generator:
    process_batch(batch)
```

## Requirements

- SageMaker Studio Lab account (free)
- 10GB available storage
- 5-8 hours total compute time
- GPU recommended for CNN training

## Next Steps

After completing this tier:
- **Tier 2:** AWS integration with S3 and SageMaker
- **Tier 3:** Production infrastructure with CloudFormation
