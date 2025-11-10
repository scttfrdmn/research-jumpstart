# Economic Time Series Forecasting

**Difficulty**: 🟡 Intermediate | **Time**: ⏱️ 2-4 hours (Studio Lab)

Apply statistical and machine learning methods to forecast economic indicators, analyze trends, and evaluate model performance on real financial data.

## Status

**Studio Lab**: 🚧 Lightweight quickstart (in development)
**Unified Studio**: ⏳ Planned

## Quick Start (Studio Lab)

```bash
git clone https://github.com/yourusername/research-jumpstart.git
cd research-jumpstart/projects/economics/time-series-forecasting/studio-lab
conda env create -f environment.yml
conda activate economic-forecasting
jupyter notebook quickstart.ipynb
```

## What You'll Learn

- Load and explore economic time series data
- Test for stationarity (ADF, KPSS tests)
- Decompose time series (trend, seasonality, residuals)
- Fit classical models (ARIMA, SARIMA, VAR)
- Apply machine learning (LSTM, Prophet, XGBoost)
- Evaluate forecast accuracy (RMSE, MAE, MAPE)
- Visualize predictions with confidence intervals

## Key Analyses

1. **Exploratory Data Analysis**
   - Time series visualization
   - Autocorrelation (ACF) and partial autocorrelation (PACF)
   - Trend and seasonality identification
   - Outlier detection

2. **Stationarity Testing**
   - Augmented Dickey-Fuller test
   - KPSS test
   - Phillips-Perron test
   - Differencing for stationarity

3. **Classical Statistical Models**
   - **ARIMA**: Auto-Regressive Integrated Moving Average
   - **SARIMA**: Seasonal ARIMA
   - **VAR**: Vector Auto-Regression (multivariate)
   - **Exponential Smoothing**: Holt-Winters methods

4. **Machine Learning Models**
   - **Prophet**: Facebook's forecasting tool
   - **LSTM**: Long Short-Term Memory networks
   - **XGBoost**: Gradient boosting for time series
   - **Ensemble methods**: Combining multiple models

5. **Model Evaluation**
   - Train-test split for time series
   - Cross-validation (rolling, expanding windows)
   - Forecast accuracy metrics
   - Residual diagnostics

## Sample Datasets

### Included Examples
- **GDP Growth**: Quarterly US GDP (FRED data)
- **Unemployment Rate**: Monthly US unemployment
- **Stock Prices**: Daily S&P 500 index
- **Inflation**: CPI year-over-year changes
- **Retail Sales**: Monthly retail trade data

### External Sources
- Federal Reserve Economic Data (FRED)
- World Bank Open Data
- Yahoo Finance
- Quandl / NASDAQ Data Link

## Cost

**Studio Lab**: Free forever (public data sources)
**Unified Studio**: ~$10-25 per month (AWS for large-scale forecasting, S3 storage)

## Prerequisites

- Basic statistics (mean, variance, correlation)
- Understanding of time series concepts
- Python programming (pandas, numpy)
- Machine learning fundamentals helpful

## Use Cases

- **Macroeconomic Forecasting**: GDP, inflation, unemployment
- **Financial Markets**: Stock prices, forex, commodities
- **Monetary Policy**: Central bank decision support
- **Business Planning**: Sales forecasting, demand planning
- **Risk Management**: VaR calculation, stress testing

## Economic Indicators

### Lagging Indicators
- Unemployment rate
- Corporate profits
- Labor cost per unit

### Leading Indicators
- Stock market returns
- Building permits
- Consumer expectations

### Coincident Indicators
- GDP
- Industrial production
- Personal income

## Typical Workflow

1. **Data Collection**: Download from FRED, Yahoo Finance, etc.
2. **Preprocessing**: Handle missing values, outliers
3. **Visualization**: Plot time series, identify patterns
4. **Stationarity Testing**: ADF test, differencing if needed
5. **Model Selection**: ACF/PACF plots for ARIMA order
6. **Model Fitting**: Fit ARIMA, SARIMA, or ML models
7. **Validation**: Cross-validation, out-of-sample testing
8. **Forecasting**: Generate predictions with confidence intervals
9. **Evaluation**: Calculate error metrics, residual analysis
10. **Interpretation**: Economic significance of results

## Statistical Methods

### ARIMA Models
- **AR(p)**: Auto-regressive order p
- **I(d)**: Differencing order d
- **MA(q)**: Moving average order q
- **Model selection**: AIC, BIC criteria
- **Example**: ARIMA(2,1,2) for US GDP

### Seasonal ARIMA
- **SARIMA(p,d,q)(P,D,Q)s**: Seasonal components
- **s**: Seasonal period (12 for monthly, 4 for quarterly)
- **Example**: SARIMA(1,1,1)(1,1,1,12) for retail sales

### Vector Auto-Regression
- Multivariate time series modeling
- Captures interdependencies between variables
- Granger causality tests
- Impulse response functions

## Machine Learning Approaches

### Prophet
- Developed by Facebook
- Handles missing data and outliers
- Automatic changepoint detection
- Holiday effects modeling
- Intuitive hyperparameter tuning

### LSTM Networks
- Deep learning for sequences
- Captures long-term dependencies
- Suitable for complex patterns
- Requires large datasets
- GPU acceleration helpful

### XGBoost for Time Series
- Feature engineering (lags, rolling windows)
- Handles non-linear relationships
- Feature importance analysis
- Fast training on CPU

## Model Evaluation Metrics

- **MAE** (Mean Absolute Error): Average absolute deviation
- **RMSE** (Root Mean Squared Error): Penalizes large errors
- **MAPE** (Mean Absolute Percentage Error): Scale-independent
- **AIC/BIC**: Information criteria for model selection
- **Ljung-Box test**: Residual autocorrelation test

## Example Results

### US GDP Quarterly Forecast
- **Model**: SARIMA(2,1,1)(1,1,0,4)
- **Forecast horizon**: 4 quarters ahead
- **RMSE**: 0.8% growth rate
- **Interpretation**: Predicts continued expansion with uncertainty

### S&P 500 Daily Returns
- **Model**: LSTM with 60-day window
- **Sharpe ratio improvement**: 0.3
- **Direction accuracy**: 55-58%
- **Note**: Markets are semi-efficient, difficult to predict

## Challenges in Economic Forecasting

### Structural Breaks
- Policy changes (e.g., COVID-19 response)
- Financial crises (2008 recession)
- Technological disruptions
- **Solution**: Regime-switching models, rolling windows

### Non-stationarity
- Most economic data non-stationary
- Requires differencing or detrending
- Cointegration for multivariate series

### Model Uncertainty
- Parameter uncertainty
- Model specification uncertainty
- **Solution**: Ensemble methods, Bayesian approaches

### Data Quality
- Revisions to historical data
- Missing observations
- Measurement error
- **Solution**: Robust estimation methods

## Advanced Topics

- **Cointegration**: Long-run equilibrium relationships
- **GARCH Models**: Volatility forecasting
- **Bayesian Structural Time Series**: Causal impact analysis
- **Dynamic Factor Models**: High-dimensional time series
- **Nowcasting**: Real-time economic monitoring

## Real-World Applications

### Central Banks
- Federal Reserve uses DSGE and VAR models
- Inflation and GDP forecasts for policy decisions
- Stress testing financial systems

### Investment Banks
- Trading strategies based on forecasts
- Risk management (VaR, CVaR)
- Portfolio optimization

### Corporations
- Sales and revenue forecasting
- Inventory management
- Workforce planning

## Resources

### Data Sources
- [FRED (Federal Reserve)](https://fred.stlouisfed.org/)
- [World Bank Data](https://data.worldbank.org/)
- [Yahoo Finance](https://finance.yahoo.com/)
- [Quandl](https://www.quandl.com/)
- [IMF Data](https://www.imf.org/en/Data)

### Python Libraries
- **statsmodels**: ARIMA, SARIMA, VAR
- **prophet**: Facebook's forecasting tool
- **pmdarima**: Auto-ARIMA model selection
- **tensorflow/keras**: LSTM networks
- **xgboost**: Gradient boosting

### Books & Papers
- "Forecasting: Principles and Practice" (Hyndman & Athanasopoulos)
- "Time Series Analysis" (Hamilton)
- Stock & Watson: "Introduction to Econometrics"

### Online Courses
- [Penn State: Applied Time Series Analysis](https://online.stat.psu.edu/stat510/)
- [Coursera: Practical Time Series Analysis](https://www.coursera.org/learn/practical-time-series-analysis)

## Community Contributions Welcome

This is a Tier 3 (starter) project. Contributions welcome:
- Complete Jupyter notebook tutorial
- Additional datasets (international data, crypto, commodities)
- Advanced models (GARCH, state-space models)
- Causal inference examples
- Real-time forecasting pipeline
- Comparison of model performance

See [PROJECT_TEMPLATE.md](../../_template/HOW_TO_USE_THIS_TEMPLATE.md) for contribution guidelines.

## License

Apache 2.0 - Sample code
Data: Check individual source licenses (FRED, World Bank typically public domain)

*Last updated: 2025-11-09*
