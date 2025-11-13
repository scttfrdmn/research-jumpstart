# Epidemiology and Disease Surveillance at Scale

**Tier 1 Flagship Project**

Real-time disease surveillance, outbreak prediction, and epidemic modeling with AWS analytics and machine learning.

## Quick Start

The main implementation is in the [`unified-studio/`](unified-studio/) directory.

**[→ View Full Documentation](unified-studio/README.md)**

## Features

- **Syndromic surveillance:** Real-time monitoring of ER visits, pharmacy sales
- **Outbreak prediction:** ML models (XGBoost, LSTM) for early warning
- **Epidemic modeling:** SIR/SEIR differential equations, agent-based simulations
- **Contact tracing:** Graph analysis with Neptune, privacy-preserving protocols
- **Data sources:** CDC WONDER, WHO, HealthMap, ProMED, Google Trends
- **Real-time dashboards:** QuickSight for public health decision-making

## Cost Estimate

**County level:** $500-2,000/month
**State level:** $5,000-20,000/month
**National:** $50,000-200,000/month
**Outbreak surge:** $10,000-50,000 for 2-4 weeks

## Technologies

- **Surveillance:** EARS algorithms (C1, C2, C3), statistical process control
- **ML:** XGBoost, LSTM, Prophet, ARIMA for forecasting
- **Modeling:** SciPy (ODE solvers), NetworkX (graph analysis), agent-based
- **AWS:** Kinesis, Timestream, Neptune, Lambda, SageMaker, Batch, QuickSight
- **Data:** CDC, WHO, HealthMap, ProMED, Google Trends, mobility data

## Applications

1. **Syndromic surveillance:** Monitor flu-like illness in real-time
2. **Outbreak prediction:** Forecast dengue outbreaks 2-4 weeks ahead
3. **Epidemic modeling:** Simulate intervention scenarios (lockdown, vaccination)
4. **Contact tracing:** Identify exposure networks, notify contacts
5. **Disease forecasting:** CDC FluSight competition-level predictions

## Quick Links

- [Getting Started Guide](unified-studio/README.md#getting-started)
- [Syndromic Surveillance](unified-studio/README.md#1-real-time-syndromic-surveillance)
- [Outbreak Prediction](unified-studio/README.md#2-outbreak-prediction-with-machine-learning)
- [Epidemic Modeling](unified-studio/README.md#3-epidemic-modeling-sirseir)
