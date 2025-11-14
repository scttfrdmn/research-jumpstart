# Climate Science on SageMaker Studio Lab

**Duration:** 1-2 hours
**Platform:** SageMaker Studio Lab ONLY
**Cost:** $0 (free Studio Lab account, no AWS charges)

## Why SageMaker Studio Lab?

This project showcases advantages of SageMaker Studio Lab over Google Colab:

### ✨ Better Compute
- **12 hours** continuous GPU sessions (vs 90 min on Colab free)
- **4 hours** CPU sessions
- No random disconnections
- Better resource allocation

### 💾 Persistence
- **15GB** persistent storage
- Save datasets between sessions
- Keep model checkpoints
- Maintain virtual environments

### 👥 Collaboration
- Share projects with team members
- Better notebook organization
- Git integration
- Project-level management

## What You'll Build

A comprehensive climate analysis workflow across multiple notebooks:

1. **Data Preparation** (20 min)
   - Download and cache multiple climate datasets
   - Process and clean data
   - Save to persistent storage

2. **Multi-Variable Analysis** (30 min)
   - Temperature, precipitation, sea level, CO2
   - Correlation analysis
   - Time series decomposition

3. **Advanced Modeling** (30 min)
   - ARIMA forecasting
   - Prophet time series models
   - Model checkpointing and persistence

4. **Interactive Dashboard** (20 min)
   - Plotly interactive visualizations
   - Multi-dataset exploration
   - Export publication-ready figures

## Datasets

All datasets stored in Studio Lab's persistent storage:

- **NOAA Temperature:** 1880-2024 global land-ocean anomalies
- **NOAA CO2:** Mauna Loa atmospheric CO2 (1958-2024)
- **NOAA Sea Level:** Global mean sea level (1993-2024)
- **GPCP Precipitation:** Global Precipitation Climatology Project

**Total size:** ~50MB (easily fits in 15GB persistent storage)

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

## Advantages Over Colab

| Feature | Colab Free | Studio Lab |
|---------|------------|------------|
| GPU Time | 90 min | 12 hours |
| CPU Time | 12 hours | 4 hours |
| Storage | None (ephemeral) | 15GB persistent |
| Disconnections | Frequent | Rare |
| Environment | Lost on restart | Persists |
| Organization | Single folder | Full project structure |
| Git Integration | Basic | Full featured |
| Collaboration | Limited | Better sharing |

## Time Estimate

- **Setup:** 10-15 minutes (one-time)
- **Data Preparation:** 20 minutes
- **Analysis Workflow:** 60-90 minutes
- **Total:** 1-2 hours

Data persists between sessions, so you can pause and resume!

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
