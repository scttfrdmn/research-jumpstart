# Macroeconomic Time Series Forecasting

**Duration:** 60-90 minutes
**Platform:** Google Colab or SageMaker Studio Lab
**Cost:** $0 (no AWS account needed)
**Data:** FRED API (Federal Reserve Economic Data)

## Research Goal

Build time series forecasting models to predict key macroeconomic indicators including GDP growth, inflation, unemployment, and recession probability using Federal Reserve Economic Data (FRED) with classical statistical models and modern machine learning approaches.

## Quick Start

### Run in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/economics/macro-forecasting/tier-0/macro-forecasting.ipynb)

### Run in SageMaker Studio Lab
[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/economics/macro-forecasting/tier-0/macro-forecasting.ipynb)

## What You'll Build

1. **Fetch FRED economic data** (GDP, CPI, unemployment via FRED API, instant access)
2. **Exploratory data analysis** (stationarity tests, ACF/PACF, seasonality decomposition)
3. **ARIMA model training** (automated order selection, 15-20 minutes)
4. **Prophet forecasting** (Facebook's time series model, 10-15 minutes)
5. **Ensemble predictions** (combine multiple model forecasts)
6. **Recession probability** (classification model for economic downturns)

## Dataset

**Federal Reserve Economic Data (FRED)**
- API: Free access to 800,000+ time series
- Indicators: GDP, CPI, unemployment rate, interest rates, consumer confidence
- Frequency: Monthly, quarterly, annual data
- Historical coverage: 1940s-present
- Source: Federal Reserve Bank of St. Louis
- Update frequency: Daily (for latest releases)

**Key Series:**
- GDP: Real Gross Domestic Product (GDPC1)
- Inflation: Consumer Price Index (CPIAUCSL)
- Unemployment: Civilian Unemployment Rate (UNRATE)
- Interest Rates: Federal Funds Rate (FEDFUNDS)

## Colab Considerations

This notebook works on Colab but you'll notice:
- **Limited model complexity** (simple ARIMA only, no deep learning)
- **Short forecast horizons** (12-24 months vs. multi-year)
- **Single-series focus** (no multivariate VAR models)
- **No ensemble models** (computational constraints)
- **Basic visualizations** (limited interactive dashboards)

These limitations become significant for professional economic research.

## What's Included

- Single Jupyter notebook (`macro-forecasting.ipynb`)
- FRED API integration (fredapi Python library)
- ARIMA/SARIMAX implementation with statsmodels
- Prophet model setup
- Stationarity testing (ADF, KPSS)
- Forecast visualization and evaluation
- Recession probability classifier

## Key Methods

- **Stationarity Tests:** Augmented Dickey-Fuller, KPSS test
- **ARIMA/SARIMAX:** Classical time series forecasting
- **Prophet:** Additive model with trend and seasonality
- **Feature Engineering:** Lagged variables, rolling statistics
- **Model Selection:** AIC, BIC, cross-validation
- **Evaluation:** RMSE, MAE, MAPE, forecast intervals

## Forecast Outputs

1. **GDP Growth:** 12-month ahead quarterly predictions
2. **Inflation:** CPI forecast with confidence intervals
3. **Unemployment Rate:** Monthly predictions
4. **Recession Probability:** Binary classification (next 12 months)
5. **Model Comparison:** Performance metrics across methods

## Next Steps

**Need more sophisticated models?** This project shows basic forecasting:

- **Tier 1:** [Advanced econometric models](../tier-1/) on Studio Lab
  - Vector Autoregression (VAR) for multivariate forecasting
  - LSTM/GRU neural networks (2-3 hour training)
  - Ensemble models combining 10+ forecasters
  - Persistent model checkpoints

- **Tier 2:** [AWS-integrated forecasting](../tier-2/) with Amazon Forecast
  - Automated machine learning (AutoML)
  - 100+ economic indicators simultaneously
  - Lambda for automated data updates
  - SageMaker for custom deep learning models

- **Tier 3:** [Production forecasting system](../tier-3/) with CloudFormation
  - Real-time nowcasting dashboards
  - Daily forecast updates with AWS Batch
  - Multi-model ensemble predictions
  - Economic scenario analysis
  - Integration with business intelligence tools

## Requirements

Pre-installed in Colab and Studio Lab:
- Python 3.9+
- pandas, numpy, scipy
- statsmodels, pmdarima
- prophet (fbprophet)
- scikit-learn
- matplotlib, seaborn, plotly

**FRED API Key:** Free registration at https://fred.stlouisfed.org/docs/api/api_key.html
