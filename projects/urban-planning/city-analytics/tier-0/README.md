# Urban Growth Prediction from Satellite Imagery

**Duration:** 60-90 minutes
**Platform:** Google Colab or SageMaker Studio Lab
**Cost:** $0 (no AWS account needed)
**Data:** ~1.5GB satellite imagery time series

## Research Goal

Train a convolutional neural network to predict urban expansion patterns from multi-temporal satellite imagery. Analyze urban growth dynamics over 20 years and forecast future development patterns for a single metropolitan area.

## Quick Start

### Run in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/urban-planning/city-analytics/tier-0/urban-growth-prediction.ipynb)

### Run in SageMaker Studio Lab
[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/urban-planning/city-analytics/tier-0/urban-growth-prediction.ipynb)

## What You'll Build

1. **Download satellite imagery** (~1.5GB Landsat time series, takes 15-20 min)
2. **Preprocess imagery** (cloud masking, normalization, band composition)
3. **Train CNN growth model** (60-75 minutes on GPU)
4. **Evaluate predictions** (accuracy, F1-score, spatial metrics)
5. **Generate growth forecasts** (2025-2035 urban expansion)

## Dataset

**Landsat 8/9 Multi-temporal Imagery**
- Source: Single metropolitan area (e.g., Austin, Texas)
- Variables: Multispectral bands (RGB, NIR, SWIR), NDVI, NDBI
- Period: 2000-2024 (annual composites)
- Resolution: 30m pixel resolution
- Size: ~1.5GB GeoTIFF files
- Source: USGS Earth Explorer / Google Earth Engine

## Colab Considerations

This notebook works on Colab but you'll notice:
- **20-minute download** at session start (no persistence)
- **60-75 minute training** (close to timeout limit)
- **Re-download required** if session disconnects
- **~10GB RAM usage** (near Colab's limit)

These limitations become important for real research workflows.

## What's Included

- Single Jupyter notebook (`urban-growth-prediction.ipynb`)
- Satellite imagery access utilities
- CNN architecture for urban growth prediction
- Training and evaluation pipeline
- Urban expansion forecast generation

## Key Methods

- **Temporal change detection:** Multi-date image analysis
- **Convolutional Neural Networks:** Learn spatial urban patterns
- **Transfer learning:** Pre-trained on ImageNet features
- **Urban indices:** NDBI (Normalized Difference Built-up Index)
- **Accuracy assessment:** Confusion matrices, spatial metrics

## Next Steps

**Experiencing limitations?** This project pushes Colab to its limits:

- **Tier 1:** [Multi-city mobility and growth ensemble](../tier-1/) on Studio Lab
  - Cache 10GB of data (download once, use forever)
  - Train ensemble models across 5-6 cities (4-6 hours continuous)
  - Persistent environments and checkpoints
  - No session timeouts

- **Tier 2:** [AWS-integrated workflows](../tier-2/) with S3 and SageMaker
  - Store 100GB+ imagery and mobility data on S3
  - Distributed preprocessing with Lambda
  - Managed training jobs

- **Tier 3:** [Production-scale analysis](../tier-3/) with full CloudFormation
  - 50+ cities (2TB+ imagery)
  - Distributed Dask clusters
  - Real-time growth monitoring

## Requirements

Pre-installed in Colab and Studio Lab:
- Python 3.9+, TensorFlow/PyTorch
- rasterio, geopandas, shapely
- scikit-learn, scipy
- matplotlib, seaborn

**Note:** First run downloads 1.5GB of data (15-20 minutes)
