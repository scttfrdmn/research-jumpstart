# Regional Climate Downscaling with CMIP6

**Duration:** 60-90 minutes
**Platform:** Google Colab or SageMaker Studio Lab
**Cost:** $0 (no AWS account needed)
**Data:** ~1.5GB CMIP6 climate model output

## Research Goal

Train a deep learning model to downscale coarse CMIP6 climate projections (100km resolution) to high-resolution regional predictions (10km resolution) for temperature and precipitation.

## Quick Start

### Run in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/climate-science/ensemble-analysis/tier-0/climate-downscaling.ipynb)

### Run in SageMaker Studio Lab
[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/climate-science/ensemble-analysis/tier-0/climate-downscaling.ipynb)

## What You'll Build

1. **Download CMIP6 data** (~1.5GB from single model, takes 15-20 min)
2. **Preprocess climate grids** (regridding, normalization)
3. **Train CNN downscaling model** (60-75 minutes on GPU)
4. **Evaluate predictions** (RMSE, spatial correlation)
5. **Generate regional forecasts** (2050-2100 projections)

## Dataset

**CMIP6 Historical + SSP2-4.5 Scenario**
- Model: CESM2 (Community Earth System Model)
- Variables: Temperature (tas), Precipitation (pr)
- Period: 1950-2100
- Resolution: 1° × 1° (100km) → 0.1° × 0.1° (10km)
- Size: ~1.5GB netCDF files
- Source: ESGF (Earth System Grid Federation)

## Colab Considerations

This notebook works on Colab but you'll notice:
- **20-minute download** at session start (no persistence)
- **60-75 minute training** (close to timeout limit)
- **Re-download required** if session disconnects
- **~11GB RAM usage** (near Colab's limit)

These limitations become important for real research workflows.

## What's Included

- Single Jupyter notebook (`climate-downscaling.ipynb`)
- CMIP6 data access utilities
- CNN architecture for spatial downscaling
- Training and evaluation pipeline
- Regional projection generation

## Key Methods

- **Statistical downscaling:** ML-based spatial interpolation
- **Convolutional Neural Networks:** Learn spatial patterns
- **Transfer learning:** Pre-trained on historical data
- **Bias correction:** Adjust systematic model errors
- **Uncertainty quantification:** Model confidence intervals

## Next Steps

**Experiencing limitations?** This project pushes Colab to its limits:

- **Tier 1:** [Multi-model ensemble analysis](../tier-1/) on Studio Lab
  - Cache 8-12GB of data (download once, use forever)
  - Train ensemble models (4-6 hours continuous)
  - Persistent environments and checkpoints
  - No session timeouts

- **Tier 2:** [AWS-integrated workflows](../tier-2/) with S3 and SageMaker
  - Store 100GB+ climate data on S3
  - Distributed preprocessing with Lambda
  - Managed training jobs

- **Tier 3:** [Production-scale analysis](../tier-3/) with full CloudFormation
  - 20+ CMIP6 models (5TB+)
  - Distributed Dask clusters
  - Real-time projection updates

## Requirements

Pre-installed in Colab and Studio Lab:
- Python 3.9+, TensorFlow/PyTorch
- xarray, netCDF4, cartopy
- scikit-learn, scipy

**Note:** First run downloads 1.5GB of data (15-20 minutes)
