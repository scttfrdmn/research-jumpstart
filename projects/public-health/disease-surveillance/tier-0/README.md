# Epidemic Forecasting with Deep Learning

**Duration:** 60-90 minutes
**Platform:** Google Colab or SageMaker Studio Lab
**Cost:** $0 (no AWS account needed)
**Data:** ~1.5GB epidemiological surveillance data

## Research Goal

Train a deep learning model (LSTM) to forecast disease outbreak patterns using historical surveillance data from multiple regions. Predict infection rates 1-4 weeks ahead to enable early public health interventions.

## Quick Start

### Run in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/public-health/disease-surveillance/tier-0/epidemic-forecasting.ipynb)

### Run in SageMaker Studio Lab
[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/public-health/disease-surveillance/tier-0/epidemic-forecasting.ipynb)

## What You'll Build

1. **Download surveillance data** (~1.5GB from CDC/WHO sources, takes 15-20 min)
2. **Preprocess time series** (feature engineering, normalization)
3. **Train LSTM forecasting model** (60-75 minutes on GPU)
4. **Evaluate predictions** (MAE, RMSE, R² metrics)
5. **Generate outbreak forecasts** (1-4 week ahead predictions)

## Dataset

**Public Health Surveillance Data**
- Source: CDC FluView / WHO Disease Outbreak News
- Variables: Cases, deaths, hospitalizations, testing rates
- Regions: Multi-state or multi-country surveillance
- Period: 2010-2024 (weekly reports)
- Resolution: Weekly aggregated data by region
- Size: ~1.5GB CSV files
- Coverage: Influenza-like illness (ILI), COVID-19, other reportable diseases

## Colab Considerations

This notebook works on Colab but you'll notice:
- **15-20 minute download** at session start (no persistence)
- **60-75 minute training** (close to timeout limit)
- **Re-download required** if session disconnects
- **~10GB RAM usage** (near Colab's limit)

These limitations become important for real research workflows.

## What's Included

- Single Jupyter notebook (`epidemic-forecasting.ipynb`)
- CDC/WHO data access utilities
- LSTM architecture for time series forecasting
- Training and evaluation pipeline
- Multi-week ahead prediction generation

## Key Methods

- **Time series forecasting:** LSTM neural networks
- **Feature engineering:** Lag features, rolling averages, trend decomposition
- **Multi-step ahead prediction:** 1-4 week forecasting horizons
- **Uncertainty quantification:** Prediction intervals
- **Epidemiological metrics:** Reproduction number (R₀), growth rates

## Next Steps

**Experiencing limitations?** This project pushes Colab to its limits:

- **Tier 1:** [Multi-disease ensemble surveillance](../tier-1/) on Studio Lab
  - Cache 10GB of data (download once, use forever)
  - Train ensemble models for 5-6 diseases (4-6 hours continuous)
  - Persistent environments and checkpoints
  - No session timeouts

- **Tier 2:** [AWS-integrated workflows](../tier-2/) with S3 and SageMaker
  - Store 100GB+ surveillance data on S3
  - Distributed preprocessing with Lambda
  - Managed training jobs
  - Real-time alerting with SNS

- **Tier 3:** [Production-scale surveillance](../tier-3/) with full CloudFormation
  - Real-time data ingestion from multiple sources
  - Distributed ensemble forecasting
  - Interactive dashboards with QuickSight
  - Automated alert systems

## Requirements

Pre-installed in Colab and Studio Lab:
- Python 3.9+, TensorFlow/PyTorch
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- statsmodels

**Note:** First run downloads 1.5GB of data (15-20 minutes)
