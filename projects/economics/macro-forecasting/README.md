# Macroeconomic Forecasting at Scale

Large-scale economic forecasting using FRED data, machine learning, and Amazon Forecast. Predict GDP growth, inflation, unemployment, and recession probability with classical econometric and modern ML approaches.

## Quick Start by Tier

**New here?** Start with tier-0 (60-90 min, free) to learn time series forecasting for macroeconomic indicators.

### 🟢 Tier 0: ARIMA & Prophet Forecasting (60-90 min, FREE)
**[Launch tier-0 project →](tier-0/)**

Forecast key economic indicators with classical and modern approaches:
- ✅ Real-time FRED API data (800K+ series: GDP, CPI, unemployment, interest rates)
- ✅ ARIMA/SARIMAX models with automated order selection
- ✅ Facebook Prophet for trend and seasonality
- ✅ Stationarity testing (ADF, KPSS) and time series diagnostics
- ✅ Recession probability classification
- ✅ Complete in 60-90 minutes
- ✅ No AWS account needed (Colab or Studio Lab)

**Platform**: Google Colab or SageMaker Studio Lab
**Cost**: $0

[View tier-0 README →](tier-0/README.md) | [Open in Colab →](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/economics/macro-forecasting/tier-0/macro-forecasting.ipynb)

---

### 🟡 Tier 1: Multivariate Models & Deep Learning (4-8 hours, FREE)
**[Launch tier-1 project →](tier-1/)**

Advanced econometric and neural network forecasting:
- ✅ Vector Autoregression (VAR) for multivariate forecasting (10+ indicators)
- ✅ LSTM/GRU neural networks for time series (2-3 hour training)
- ✅ Ensemble models combining 10+ forecasters (ARIMA, Prophet, XGBoost, LSTM)
- ✅ Granger causality testing and impulse response functions
- ✅ Persistent model checkpoints and iterative refinement (Studio Lab)
- ✅ Still free, no AWS account

**Platform**: SageMaker Studio Lab
**Cost**: $0

[View tier-1 README →](tier-1/README.md)

---

### 🟠 Tier 2: Production Forecasting with Amazon Forecast (2-3 days, $25-50/analysis)
**[Launch tier-2 project →](tier-2/)**

Research-grade forecasting infrastructure:
- ✅ CloudFormation one-click deployment
- ✅ Amazon Forecast AutoML (automatic model selection from 6+ algorithms)
- ✅ 100+ economic indicators processed simultaneously
- ✅ Automated data updates via Lambda (daily FRED API fetches)
- ✅ SageMaker for custom deep learning models
- ✅ Long forecast horizons (multi-year projections)
- ✅ Publication-ready outputs and visualizations

**Platform**: AWS with CloudFormation
**Cost**: $25-50 per complete analysis

[View tier-2 README →](tier-2/README.md)

---

### 🔴 Tier 3: Enterprise Forecasting Platform (Ongoing, $500-1K/month)
**[Launch tier-3 project →](tier-3/)**

Production platform for economic analysis teams:
- ✅ Real-time nowcasting dashboards (QuickSight)
- ✅ Daily automated forecast updates with AWS Batch
- ✅ Multi-model ensemble predictions (ARIMA, VAR, LSTM, Forecast)
- ✅ Economic scenario analysis and stress testing
- ✅ Integration with business intelligence tools
- ✅ AI-assisted interpretation (Amazon Bedrock)
- ✅ Team collaboration with versioned forecasts

**Platform**: AWS multi-account with enterprise support
**Cost**: $500-1K/month

[View tier-3 README →](tier-3/README.md)

---

## What You'll Learn

Across all tiers, this project teaches:
- Time series analysis fundamentals (stationarity, ACF/PACF, seasonality)
- ARIMA/SARIMAX modeling with automated order selection
- Facebook Prophet for trend and seasonality forecasting
- Vector Autoregression (VAR) for multivariate economic models
- LSTM/GRU neural networks for time series
- Amazon Forecast AutoML for production forecasting

## Technologies & Tools

- **Data sources**: FRED API (Federal Reserve Economic Data, 800K+ series), World Bank, OECD
- **Languages**: Python 3.9+
- **Core libraries**: pandas, numpy, scipy, statsmodels, pmdarima
- **Time series**: fbprophet, arch (GARCH), linearmodels (VAR)
- **ML frameworks**: scikit-learn, TensorFlow/Keras (LSTM), XGBoost
- **Cloud services** (tier 2+): S3, Glue, Athena, Lambda (automated updates), SageMaker, Amazon Forecast (AutoML), Bedrock

## Project Structure

```
macro-forecasting/
├── tier-0/              # ARIMA & Prophet (60-90 min, FREE)
│   ├── macro-forecasting.ipynb
│   ├── README.md
│   └── requirements.txt
├── tier-1/              # VAR & LSTM (4-8 hours, FREE)
│   ├── notebooks/
│   ├── src/
│   ├── data/
│   ├── environment.yml
│   └── README.md
├── tier-2/              # Amazon Forecast (2-3 days, $25-50)
│   ├── cloudformation/
│   ├── notebooks/
│   ├── src/
│   ├── tests/
│   └── README.md
└── tier-3/              # Enterprise platform (ongoing, $500-1K/mo)
    ├── cloudformation/
    ├── notebooks/
    ├── src/
    ├── infrastructure/
    └── README.md
```

## Progression Path

```
Tier 0           → Tier 1          → Tier 2            → Tier 3
ARIMA/Prophet      VAR/LSTM           Amazon Forecast     Enterprise
5 indicators       10+ indicators     100+ indicators     Real-time
60-90 min          4-8 hours          2-3 days            Ongoing
FREE               FREE               $25-50/analysis     $500-1K/mo
```

You can:
- ✅ Skip tiers if you have AWS experience and production forecasting needs
- ✅ Stop at any tier - tier-1 is great for research papers, tier-2 for policy analysis
- ✅ Mix and match - use tier-0 for method testing, tier-2 for operational forecasts

[Understanding tiers →](../../../docs/projects/tiers.md)

## Economic Applications

- **GDP growth forecasting**: Quarterly projections 1-4 quarters ahead with confidence intervals
- **Inflation prediction**: CPI forecasts for monetary policy analysis
- **Unemployment rate**: Monthly labor market predictions
- **Recession probability**: Binary classification and early warning signals
- **Nowcasting**: Real-time economic activity estimates before official releases
- **Policy scenario analysis**: Evaluate impacts of fiscal/monetary interventions

## Related Projects

- **[Time Series Forecasting](../time-series-forecasting/)** - Additional econometric methods
- **[Social Science - Social Media Analysis](../../social-science/social-media-analysis/)** - Sentiment indicators for forecasting
- **[Urban Planning - City Analytics](../../urban-planning/city-analytics/)** - Regional economic analysis

## Common Use Cases

- **Central banks**: Monitor economic conditions, calibrate policy models
- **Finance firms**: Asset allocation, risk management, trading strategies
- **Consulting firms**: Economic outlook reports for clients
- **Academic research**: Test economic theories, publish empirical papers
- **Government agencies**: Budget forecasting, policy impact analysis
- **Businesses**: Demand planning, strategic planning, scenario analysis

## Cost Estimates

**Tier 2 Production (Amazon Forecast)**:
- **Amazon Forecast** (10 indicators, 100 time series, 12-month horizon):
  - Data ingestion: $0.088 per 1,000 time series = $0.01
  - Training: $0.24 per hour = $5-10 (automatic)
  - Forecasts: $0.60 per 1,000 forecasts = $0.06
- **Lambda** (daily FRED updates): $0.20/month
- **S3 storage** (historical data): $0.10/month
- **SageMaker** (optional custom LSTM): ml.p3.2xlarge, 3 hours = $10-12
- **Total**: $25-50 per complete analysis

**Optimization tips**:
- Reuse trained Forecast predictors across similar series
- Use spot instances for SageMaker training (60-70% savings)
- Archive old forecasts to S3 Glacier
- Batch FRED API calls to reduce Lambda invocations

## Support

- **Questions**: [GitHub Discussions](https://github.com/scttfrdmn/research-jumpstart/discussions)
- **Issues**: [GitHub Issues](https://github.com/scttfrdmn/research-jumpstart/issues)
- **Office Hours**: [Every Tuesday](../../../docs/community/office-hours.md)

## Citation

If you use this project in your research, please cite:

```bibtex
@software{research_jumpstart_macro_forecasting,
  title = {Macroeconomic Forecasting at Scale: Research Jumpstart},
  author = {Research Jumpstart Community},
  year = {2025},
  url = {https://github.com/scttfrdmn/research-jumpstart},
  note = {Accessed: [date]}
}
```

Also cite the FRED data:
- **FRED**: Federal Reserve Economic Data, Federal Reserve Bank of St. Louis
  - https://fred.stlouisfed.org/
  - FRED API key required (free): https://fred.stlouisfed.org/docs/api/api_key.html

## License

Apache 2.0 - See [LICENSE](../../../LICENSE) for details.

---

*Part of [Research Jumpstart](https://github.com/scttfrdmn/research-jumpstart) - Pre-built research workflows for cloud computing*
