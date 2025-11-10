# Economics - Time Series Analysis & Forecasting

**Duration:** 3 hours | **Level:** Beginner-Intermediate | **Cost:** Free

Forecast macroeconomic indicators using ARIMA models and discover causal relationships with Granger causality testing.

[![Open In Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws)

## Overview

Learn econometric time series analysis by forecasting GDP, inflation, unemployment, and interest rates. This tutorial teaches the same methods used by central banks and economic researchers to understand and predict economic trends.

### What You'll Build
- Stationarity test suite (ADF, KPSS)
- ARIMA forecasting models
- Granger causality analyzer
- Economic visualization dashboard
- Confidence interval calculator

### Real-World Applications
- Central bank policy analysis
- Economic forecasting
- Financial market prediction
- Business cycle research
- Macroeconomic research

## Learning Objectives

✅ Test time series stationarity using ADF and KPSS tests
✅ Identify ARIMA model orders (p, d, q)
✅ Fit ARIMA models for economic forecasting
✅ Generate forecasts with confidence intervals
✅ Conduct Granger causality tests
✅ Interpret ACF and PACF plots
✅ Analyze economic relationships and trends

## Dataset

**Quarterly Macroeconomic Indicators (2020-2025)**

| Variable | Description | Range | Frequency |
|----------|-------------|-------|-----------|
| **GDP** | Real GDP (billions) | $19,300 - $23,800 | Quarterly |
| **Inflation** | CPI inflation rate (%) | 0.5% - 8.0% | Quarterly |
| **Unemployment** | Unemployment rate (%) | 3.4% - 13.0% | Quarterly |
| **Interest Rate** | Federal funds rate (%) | 0.25% - 5.5% | Quarterly |
| **Stock Index** | S&P 500 proxy | 2,200 - 4,800 | Quarterly |

**Time Period:** 24 quarters (Q1 2020 - Q4 2025)

**Key Features:**
- COVID-19 pandemic effects (2020-2021)
- Recovery period (2021-2022)
- Inflation surge (2022-2023)
- Interest rate normalization (2023-2025)

## Methods and Techniques

### 1. Stationarity Testing

**Augmented Dickey-Fuller (ADF) Test:**
- Tests for unit root (non-stationarity)
- H₀: Series has unit root (non-stationary)
- p < 0.05: Reject H₀, series is stationary

**KPSS Test:**
- Tests for level/trend stationarity
- H₀: Series is stationary
- p > 0.05: Fail to reject H₀, series is stationary

**Differencing:**
```python
# First difference to achieve stationarity
gdp_diff = gdp.diff().dropna()
```

### 2. ARIMA Modeling

**ARIMA(p, d, q) Components:**
- **p**: Autoregressive order (past values)
- **d**: Differencing order (stationarity)
- **q**: Moving average order (past errors)

**Model Selection:**
- ACF plot: Identify MA order (q)
- PACF plot: Identify AR order (p)
- AIC/BIC criteria: Compare models

```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(data, order=(1, 1, 1))
results = model.fit()
forecast = results.forecast(steps=4)
```

### 3. Granger Causality

**Tests if X "Granger-causes" Y:**
- Does past X help predict Y beyond past Y alone?
- Bidirectional testing: X→Y and Y→X
- Multiple lag testing (1-4 quarters)

```python
from statsmodels.tsa.stattools import grangercausalitytests

# Test if inflation Granger-causes interest rates
grangercausalitytests(data[['interest_rate', 'inflation']], maxlag=4)
```

**Interpretation:**
- p < 0.05: X Granger-causes Y
- Causality ≠ true causation (statistical relationship)

### 4. Forecasting with Uncertainty

**Confidence Intervals:**
- 95% confidence bands around forecast
- Widens with forecast horizon
- Reflects model and parameter uncertainty

## Notebook Structure

### Part 1: Data Exploration (20 min)
- Load and visualize time series
- Summary statistics
- Trend identification
- COVID-19 impact analysis

### Part 2: Stationarity Analysis (25 min)
- ADF and KPSS tests for all series
- Interpret test results
- Apply differencing
- Verify stationarity after transformation

### Part 3: ARIMA Identification (30 min)
- ACF/PACF analysis
- Model order selection
- AIC/BIC comparison
- Residual diagnostics

### Part 4: GDP Forecasting (30 min)
- Fit ARIMA(1,1,1) model
- Parameter interpretation
- Generate 4-quarter ahead forecast
- Plot forecast with 95% CI

### Part 5: Granger Causality (25 min)
- Test inflation → interest rates
- Test unemployment → GDP
- Test interest rates → stock index
- Interpret causal relationships

### Part 6: Economic Interpretation (20 min)
- Policy implications
- Business cycle analysis
- Forecast reliability
- Real-world applications

**Total:** ~2.5-3 hours

## Key Results

### GDP Forecast (Q1-Q4 2026)
| Quarter | Forecast | 95% CI Lower | 95% CI Upper |
|---------|----------|--------------|--------------|
| Q1 2026 | $23,950B | $23,200B | $24,700B |
| Q2 2026 | $24,100B | $22,800B | $25,400B |
| Q3 2026 | $24,250B | $22,400B | $26,100B |
| Q4 2026 | $24,400B | $22,000B | $26,800B |

### Granger Causality Findings

**Significant Relationships (p < 0.05):**
- ✅ Inflation → Interest Rates (lag 1-2 quarters)
- ✅ Interest Rates → GDP (lag 2-3 quarters)
- ✅ Unemployment → GDP (lag 1 quarter)
- ❌ Stock Index → GDP (not significant)

**Economic Interpretation:**
- Federal Reserve responds to inflation (Taylor rule)
- Monetary policy affects real economy with lag
- Labor market is leading indicator for GDP

### Model Performance

**GDP ARIMA(1,1,1):**
- AIC: 245.3
- BIC: 249.8
- RMSE: $320B
- MAPE: 1.8%

## Visualizations

1. **Time Series Dashboard**: All 5 variables over time
2. **Stationarity Tests**: Before/after differencing comparison
3. **ACF/PACF Plots**: For model identification
4. **Forecast Plot**: GDP with 95% confidence bands
5. **Granger Causality Network**: Directional graph of relationships
6. **Residual Diagnostics**: QQ-plot and Ljung-Box test

## Extensions

### Modify the Analysis
- Try different ARIMA orders
- Forecast other variables (inflation, unemployment)
- Add seasonal components (SARIMA)
- Compare with naive forecasts

### Advanced Econometrics
- Vector Autoregression (VAR)
- Error Correction Models (ECM)
- ARCH/GARCH for volatility
- Structural breaks analysis

### Real Data
- Download from [FRED](https://fred.stlouisfed.org/)
- Monthly or weekly frequency
- International comparisons
- Industry-specific indicators

## Resources

- **[FRED](https://fred.stlouisfed.org/)**: Federal Reserve economic data
- **statsmodels**: [Time Series Analysis](https://www.statsmodels.org/stable/tsa.html)
- **[EViews Tutorials](https://www.eviews.com/)**: Econometric software
- **Textbook**: *Introduction to Econometrics* by Stock & Watson

## Getting Started

```bash
# Clone repository
git clone https://github.com/research-jumpstart/research-jumpstart.git
cd projects/economics/time-series-analysis/studio-lab

# Create environment
conda env create -f environment.yml
conda activate economics-ts

# Launch
jupyter lab quickstart.ipynb
```

## FAQs

??? question "Do I need economics knowledge?"
    Basic macro concepts help but aren't required. The notebook explains key economic relationships.

??? question "Why use ARIMA instead of ML?"
    ARIMA provides interpretable parameters, confidence intervals, and handles time dependencies explicitly. Great for small datasets.

??? question "What does Granger causality really mean?"
    Statistical predictive power, not true causation. "X Granger-causes Y" means X helps forecast Y.

---

**[Launch the notebook →](https://studiolab.sagemaker.aws)** 📊
