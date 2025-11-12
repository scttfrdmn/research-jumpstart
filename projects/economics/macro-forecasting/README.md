# Macroeconomic Forecasting

**Tier 1 Flagship Project**

Large-scale economic forecasting using FRED data, machine learning, and Amazon Forecast.

## Quick Start

The main implementation is in the [`unified-studio/`](unified-studio/) directory.

**[→ View Full Documentation](unified-studio/README.md)**

## Features

- **Data Sources:** FRED (800K+ series), World Bank, OECD
- **Models:** ARIMA, SARIMAX, VAR, LSTM, Prophet, Ensemble
- **Forecasts:** GDP growth, inflation, unemployment, recession probability
- **AWS Integration:** Amazon Forecast AutoML, SageMaker, Bedrock

## Cost Estimate

**$25-50** for complete analysis with multiple forecasting models

## Technologies

- **Data:** Federal Reserve Economic Data (FRED API)
- **Statistical:** statsmodels, pmdarima, prophet
- **ML:** scikit-learn, TensorFlow/Keras, XGBoost
- **AWS:** S3, Glue, Athena, SageMaker, Forecast, Bedrock
- **Compute:** ml.p3.2xlarge for LSTM training

## Quick Links

- [Getting Started Guide](unified-studio/README.md#getting-started)
- [GDP Forecasting](unified-studio/README.md#gdp-forecasting)
- [Recession Prediction](unified-studio/README.md#recession-prediction)
- [CloudFormation Template](unified-studio/cloudformation/economics-stack.yml)
