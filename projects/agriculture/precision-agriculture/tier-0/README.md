# Crop Disease Detection from Satellite Imagery

**Duration:** 60-90 minutes
**Platform:** Google Colab or SageMaker Studio Lab
**Cost:** $0 (no AWS account needed)
**Data:** ~1.5GB Sentinel-2 imagery time series

## Research Goal

Train a deep learning model to detect crop diseases and assess crop health from Sentinel-2 satellite imagery. Use multi-spectral bands (visible, NIR, SWIR) to identify stressed vegetation and classify crop health conditions.

## Quick Start

### Run in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/agriculture/precision-agriculture/tier-0/crop-disease-detection.ipynb)

### Run in SageMaker Studio Lab
[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/agriculture/precision-agriculture/tier-0/crop-disease-detection.ipynb)

## What You'll Build

1. **Download Sentinel-2 imagery** (~1.5GB time series, takes 15-20 min)
2. **Preprocess multi-spectral bands** (cloud masking, NDVI calculation)
3. **Train CNN for crop health classification** (60-75 minutes on GPU)
4. **Evaluate disease detection** (accuracy, precision, recall)
5. **Generate health maps** (field-level assessment)

## Dataset

**Sentinel-2 Multi-Spectral Imagery**
- Satellite: ESA Sentinel-2A/2B
- Bands: 13 bands (443nm to 2190nm)
- Key bands: Red, Green, Blue, NIR, Red Edge, SWIR
- Resolution: 10m (RGB, NIR), 20m (Red Edge, SWIR)
- Temporal: 6-month time series (growing season)
- Size: ~1.5GB GeoTIFF files
- Source: Copernicus Open Access Hub
- AOI: 50km² agricultural region

## Colab Considerations

This notebook works on Colab but you'll notice:
- **15-20 minute download** at session start (no persistence)
- **60-75 minute training** (close to timeout limit)
- **Re-download required** if session disconnects
- **~11GB RAM usage** (near Colab's limit)

These limitations become important for real research workflows.

## What's Included

- Single Jupyter notebook (`crop-disease-detection.ipynb`)
- Sentinel-2 data access utilities
- CNN architecture for multi-spectral classification
- NDVI and vegetation index calculations
- Training and evaluation pipeline
- Field-level health map generation

## Key Methods

- **Multi-spectral analysis:** Use bands beyond visible light
- **Vegetation indices:** NDVI, EVI, SAVI for crop health
- **Convolutional Neural Networks:** Learn spatial patterns
- **Time series analysis:** Detect temporal changes
- **Cloud masking:** Remove cloudy pixels automatically

## Health Classification Classes

1. **Healthy:** Normal vegetation, high NDVI (>0.6)
2. **Water Stress:** Moderate decline, NDVI 0.4-0.6
3. **Nutrient Deficiency:** Yellowing, abnormal reflectance
4. **Disease:** Severe stress, NDVI <0.4, patchy patterns
5. **Bare Soil:** No vegetation cover

## Next Steps

**Experiencing limitations?** This project pushes Colab to its limits:

- **Tier 1:** [Multi-sensor precision agriculture](../tier-1/) on Studio Lab
  - Cache 10GB multi-sensor data (Sentinel-2, Landsat, MODIS, weather)
  - Train ensemble yield prediction models (4-6 hours continuous)
  - Persistent environments and checkpoints
  - No session timeouts

- **Tier 2:** [AWS-integrated workflows](../tier-2/) with S3 and Lambda
  - Store 100GB+ satellite imagery on S3
  - Automated download pipelines with Lambda
  - Managed training jobs with SageMaker

- **Tier 3:** [Production-scale monitoring](../tier-3/) with full CloudFormation
  - Real-time satellite ingestion (daily updates)
  - Distributed processing with AWS Batch
  - Field-level alerts and dashboards
  - Integration with farm management systems

## Requirements

Pre-installed in Colab and Studio Lab:
- Python 3.9+, TensorFlow/PyTorch
- rasterio, gdal, shapely
- scikit-learn, scipy
- matplotlib, folium

**Note:** First run downloads 1.5GB of data (15-20 minutes)
