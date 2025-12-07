# Economic Time Series Forecasting at Scale

Advanced time series forecasting for economic indicators using LSTM neural networks, ensemble methods, and Amazon Forecast. Predict GDP growth, inflation, unemployment, and multi-country spillover effects.

## Quick Start by Tier

**New here?** Start with tier-0 (60-90 min, free) to learn LSTM forecasting for economic data.

### 🟢 Tier 0: LSTM Economic Forecasting (60-90 min, FREE)
**[Launch tier-0 project →](tier-0/)**

Train LSTM networks for economic indicator forecasting:
- ✅ Real economic data (~1.5GB from FRED, World Bank, OECD, 1960-2024)
- ✅ LSTM neural networks for multi-step forecasting
- ✅ Stationarity testing and feature engineering (lagged variables, moving averages)
- ✅ Multi-horizon forecasts (12-24 month projections)
- ✅ Walk-forward validation and out-of-sample testing
- ✅ Complete in 60-90 minutes
- ✅ No AWS account needed (Colab or Studio Lab)

**Platform**: Google Colab or SageMaker Studio Lab
**Cost**: $0

[View tier-0 README →](tier-0/README.md) | [Open in Colab →](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/economics/time-series-forecasting/tier-0/economic-forecasting.ipynb)

---

### 🟡 Tier 1: Multi-Country Ensemble Models (4-8 hours, FREE)
**[Launch tier-1 project →](tier-1/)**

Panel data models and cross-country spillover analysis:
- ✅ 10GB multi-country panel data (20+ economies, 500+ indicators)
- ✅ Ensemble models (ARIMA, LSTM, GRU, XGBoost, Prophet)
- ✅ Cross-country spillover effects and contagion analysis
- ✅ Granger causality and impulse response functions
- ✅ Persistent storage for long training runs (Studio Lab)
- ✅ Still free, no AWS account

**Platform**: SageMaker Studio Lab
**Cost**: $0

[View tier-1 README →](tier-1/README.md)

---

### 🟠 Tier 2: Production Forecasting Infrastructure (2-3 days, $50-100/month)
**[Launch tier-2 project →](tier-2/)**

Research-grade forecasting with automated pipelines:
- ✅ CloudFormation one-click deployment
- ✅ 50GB+ economic data archives on S3
- ✅ Automated data pipelines with Lambda (daily FRED/World Bank updates)
- ✅ SageMaker hyperparameter tuning for LSTM optimization
- ✅ Amazon Forecast AutoML for baseline comparisons
- ✅ Real-time indicator updates and nowcasting
- ✅ Publication-ready outputs and visualizations

**Platform**: AWS with CloudFormation
**Cost**: $50-100/month for continuous forecasting

[View tier-2 README →](tier-2/README.md)

---

### 🔴 Tier 3: Enterprise Forecasting Platform (Ongoing, $1K-2K/month)
**[Launch tier-3 project →](tier-3/)**

Production platform for economic research teams:
- ✅ 100+ countries, 500+ indicators with distributed training
- ✅ Real-time forecast API for integration with business systems
- ✅ Automated retraining pipelines (weekly model updates)
- ✅ Economic scenario analysis and stress testing
- ✅ AI-assisted interpretation (Amazon Bedrock)
- ✅ Interactive dashboards (QuickSight) for exploration
- ✅ Team collaboration with versioned forecasts

**Platform**: AWS multi-account with enterprise support
**Cost**: $1K-2K/month

[View tier-3 README →](tier-3/README.md)

---

## What You'll Learn

Across all tiers, this project teaches:
- LSTM/GRU neural networks for time series forecasting
- Stationarity testing (Augmented Dickey-Fuller) and differencing
- Feature engineering for time series (lags, moving averages, indicators)
- Multi-step forecasting strategies (recursive vs. direct)
- Walk-forward validation and out-of-sample testing
- Ensemble forecasting methods (model averaging, stacking)

## Technologies & Tools

- **Data sources**: FRED API (800K+ series), World Bank WDI, OECD Stats, IMF IFS
- **Languages**: Python 3.9+
- **Core libraries**: pandas, numpy, scipy, statsmodels
- **Deep learning**: TensorFlow/Keras (LSTM, GRU), PyTorch
- **ML frameworks**: scikit-learn, XGBoost, fbprophet
- **Cloud services** (tier 2+): S3, Lambda (data pipelines), SageMaker (training, hyperparameter tuning), Amazon Forecast (AutoML), Bedrock

## Project Structure

```
time-series-forecasting/
├── tier-0/              # LSTM forecasting (60-90 min, FREE)
│   ├── economic-forecasting.ipynb
│   ├── README.md
│   └── requirements.txt
├── tier-1/              # Multi-country ensemble (4-8 hours, FREE)
│   ├── notebooks/
│   ├── src/
│   ├── data/
│   ├── environment.yml
│   └── README.md
├── tier-2/              # Production infrastructure (2-3 days, $50-100/mo)
│   ├── cloudformation/
│   ├── notebooks/
│   ├── src/
│   ├── tests/
│   └── README.md
└── tier-3/              # Enterprise platform (ongoing, $1K-2K/mo)
    ├── cloudformation/
    ├── notebooks/
    ├── src/
    ├── infrastructure/
    └── README.md
```

## Progression Path

```
Tier 0           → Tier 1          → Tier 2            → Tier 3
LSTM Forecasting   Multi-Country      Production          Enterprise
5 indicators       20+ countries      500+ indicators     Real-time API
60-90 min          4-8 hours          2-3 days            Ongoing
FREE               FREE               $50-100/mo          $1K-2K/mo
```

You can:
- ✅ Skip tiers if you have AWS experience and production forecasting needs
- ✅ Stop at any tier - tier-1 is great for academic research, tier-2 for policy analysis
- ✅ Mix and match - use tier-0 for prototyping, tier-2 for operational forecasts

[Understanding tiers →](../../../docs/projects/tiers.md)

## Economic Applications

- **GDP growth forecasting**: Quarterly projections using LSTM with macroeconomic features
- **Inflation prediction**: CPI forecasts with supply chain and monetary indicators
- **Unemployment forecasting**: Labor market predictions with leading indicators
- **Multi-country analysis**: Cross-border spillover effects and contagion modeling
- **Nowcasting**: Real-time economic activity estimates using high-frequency data
- **Scenario analysis**: What-if simulations for policy interventions

## Related Projects

- **[Macro Forecasting](../macro-forecasting/)** - ARIMA and classical econometric methods
- **[Social Science - Social Media Analysis](../../social-science/social-media-analysis/)** - Sentiment indicators for forecasting
- **[Climate Science - Ensemble Analysis](../../climate-science/ensemble-analysis/)** - Similar ensemble forecasting techniques

## Common Use Cases

- **Central banks**: Economic surveillance, monetary policy calibration
- **Investment firms**: Asset allocation based on macro forecasts
- **Consulting firms**: Economic outlook reports for corporate clients
- **Academic researchers**: Test forecasting methodologies, publish papers
- **Government agencies**: Budget projections, fiscal policy planning
- **International organizations**: Multi-country comparative forecasts (IMF, World Bank)

## Cost Estimates

**Tier 2 Production (Continuous Forecasting)**:
- **Lambda** (daily data updates): $5/month
- **S3 storage** (50GB historical data): $1.15/month
- **SageMaker training** (weekly LSTM retraining): ml.p3.2xlarge, 4 hours/week = $40/month
- **Amazon Forecast** (optional baseline): 50 series, monthly updates = $15/month
- **CloudWatch** (monitoring): $5/month
- **Total**: $50-100/month for automated forecasting pipeline

**Optimization tips**:
- Use spot instances for SageMaker training (60-70% savings)
- Cache preprocessed features to reduce Lambda compute time
- Archive old forecasts to S3 Glacier ($0.004/GB/month)
- Batch API calls to reduce Lambda invocations

## Support

- **Questions**: [GitHub Discussions](https://github.com/scttfrdmn/research-jumpstart/discussions)
- **Issues**: [GitHub Issues](https://github.com/scttfrdmn/research-jumpstart/issues)
- **Office Hours**: [Every Tuesday](../../../docs/community/office-hours.md)

## Citation

If you use this project in your research, please cite:

```bibtex
@software{research_jumpstart_time_series_forecasting,
  title = {Economic Time Series Forecasting at Scale: Research Jumpstart},
  author = {Research Jumpstart Community},
  year = {2025},
  url = {https://github.com/scttfrdmn/research-jumpstart},
  note = {Accessed: [date]}
}
```

Also cite the appropriate data sources:
- **FRED**: Federal Reserve Economic Data, https://fred.stlouisfed.org/
- **World Bank**: World Development Indicators, https://databank.worldbank.org/
- **OECD**: OECD.Stat, https://stats.oecd.org/

## License

Apache 2.0 - See [LICENSE](../../../LICENSE) for details.

---

*Part of [Research Jumpstart](https://github.com/scttfrdmn/research-jumpstart) - Pre-built research workflows for cloud computing*
