# Economic Forecasting with LSTM

**Duration:** 60-90 minutes
**Platform:** Google Colab or SageMaker Studio Lab
**Cost:** $0 (no AWS account needed)
**Data:** ~1.5GB economic time series data

## Research Goal

Train an LSTM neural network to forecast key economic indicators (GDP growth, inflation, unemployment) using historical macroeconomic data from Federal Reserve, World Bank, and OECD sources.

## Quick Start

### Run in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/economics/time-series-forecasting/tier-0/economic-forecasting.ipynb)

### Run in SageMaker Studio Lab
[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/economics/time-series-forecasting/tier-0/economic-forecasting.ipynb)

## What You'll Build

1. **Download economic data** (~1.5GB from FRED, World Bank, OECD, takes 15-20 min)
2. **Preprocess time series** (stationarity testing, normalization, lag features)
3. **Train LSTM forecasting model** (60-75 minutes on GPU)
4. **Evaluate predictions** (RMSE, MAE, directional accuracy)
5. **Generate economic forecasts** (12-24 month projections)

## Dataset

**Multi-Source Economic Indicators**
- Source: FRED (Federal Reserve), World Bank, OECD
- Variables: GDP growth, inflation (CPI), unemployment, interest rates, trade balance
- Period: 1960-2024 (quarterly and monthly data)
- Coverage: US, EU, China, Japan, major economies
- Size: ~1.5GB CSV files
- Access: Public APIs (no authentication required)

## Colab Considerations

This notebook works on Colab but you'll notice:
- **15-20 minute download** at session start (no persistence)
- **60-75 minute training** (close to timeout limit)
- **Re-download required** if session disconnects
- **~11GB RAM usage** (near Colab's limit)

These limitations become important for real research workflows.

## What's Included

- Single Jupyter notebook (`economic-forecasting.ipynb`)
- FRED API data access utilities
- LSTM architecture for time series forecasting
- Feature engineering pipeline
- Training and evaluation workflow
- Multi-horizon forecast generation

## Key Methods

- **LSTM networks:** Capture long-term economic dependencies
- **Stationarity testing:** Augmented Dickey-Fuller tests
- **Feature engineering:** Lagged variables, moving averages, differencing
- **Multi-step forecasting:** Recursive and direct strategies
- **Model evaluation:** Out-of-sample testing, walk-forward validation

## Next Steps

**Experiencing limitations?** This project pushes Colab to its limits:

- **Tier 1:** [Multi-country ensemble economic modeling](../tier-1/) on Studio Lab
  - Cache 10GB of panel data (download once, use forever)
  - Train ensemble models with 5-6 algorithms (4-6 hours continuous)
  - Cross-country spillover analysis
  - Persistent environments and checkpoints
  - No session timeouts

- **Tier 2:** [AWS-integrated workflows](../tier-2/) with S3 and SageMaker
  - Store 50GB+ economic data on S3
  - Automated data pipelines with Lambda
  - SageMaker training jobs with hyperparameter tuning
  - Real-time indicator updates

- **Tier 3:** [Production-scale forecasting](../tier-3/) with full CloudFormation
  - 100+ countries and 500+ indicators
  - Distributed training with SageMaker
  - API for forecast delivery
  - Automated retraining pipelines

## Requirements

Pre-installed in Colab and Studio Lab:
- Python 3.9+, TensorFlow/PyTorch
- pandas, numpy, statsmodels
- scikit-learn, scipy
- matplotlib, seaborn

**Note:** First run downloads 1.5GB of data (15-20 minutes)
