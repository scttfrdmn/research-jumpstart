# Smart Grid Optimization and Energy Systems

**Tier 1 Flagship Project**

Large-scale power grid optimization with load forecasting, renewable integration, battery optimization, and EV charging coordination on AWS.

## Quick Start

The main implementation is in the [`unified-studio/`](unified-studio/) directory.

**[→ View Full Documentation](unified-studio/README.md)**

## Features

- **Load forecasting:** LSTM, Prophet, SARIMAX for 24-72 hour ahead demand (2-5% MAPE)
- **Solar/wind forecasting:** Weather-integrated ML models for renewable generation (10-20% RMSE)
- **Battery optimization:** Arbitrage, frequency regulation, degradation modeling with CVXPY/Pyomo
- **EV charging coordination:** Smart scheduling to minimize grid stress, V2G potential
- **Grid anomaly detection:** LSTM autoencoders for predictive maintenance (20-40% outage reduction)
- **Optimal power flow:** Unit commitment, economic dispatch with Pyomo

## Cost Estimate

**Small microgrid (100 customers):** $150-200/month
**Distribution network (10K):** $1,750-2,500/month
**Utility-scale (100K+):** $12,000-15,000/month
**Regional ISO/RTO:** $50,000-100,000/month

## Technologies

- **Forecasting:** LSTM, Prophet, SARIMAX, XGBoost
- **Optimization:** Pyomo, CVXPY, Gurobi
- **IoT:** AWS IoT Core for smart meters, SCADA sensors
- **Time Series:** Timestream for sensor data, Kinesis for streaming
- **AWS:** S3, IoT Core, Timestream, Lambda, SageMaker, Batch, Kinesis, SNS, QuickSight
- **Data:** EIA, NREL (solar/wind), NOAA weather, OpenEI rates, smart meter AMI

## Applications

1. **Demand forecasting:** Day-ahead load prediction with 2-5% MAPE
2. **Solar forecasting:** PV generation with weather integration (10-20% RMSE)
3. **Battery arbitrage:** Charge low/discharge high for revenue maximization
4. **EV smart charging:** Minimize peak demand, respect transformer limits
5. **Anomaly detection:** Predict equipment failures before outages

## Quick Links

- [Getting Started Guide](unified-studio/README.md#getting-started)
- [Load Forecasting](unified-studio/README.md#1-electricity-load-forecasting-with-lstm)
- [Solar Forecasting](unified-studio/README.md#2-solar-power-forecasting)
- [Battery Optimization](unified-studio/README.md#3-battery-energy-storage-optimization)
