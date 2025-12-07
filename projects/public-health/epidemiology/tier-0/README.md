# Epidemiological Modeling and Disease Surveillance

**Duration:** 60-90 minutes
**Platform:** Google Colab or SageMaker Studio Lab
**Cost:** $0 (no AWS account needed)
**Data:** Public health surveillance data (~200MB)

## Research Goal

Build compartmental epidemiological models (SIR, SEIR) to simulate disease transmission, estimate reproduction numbers (R0), forecast outbreak trajectories, and analyze intervention strategies using real public health surveillance data.

## Quick Start

### Run in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/public-health/epidemiology/tier-0/epidemic-modeling.ipynb)

### Run in SageMaker Studio Lab
[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/public-health/epidemiology/tier-0/epidemic-modeling.ipynb)

## What You'll Build

1. **Load surveillance data** (COVID-19, influenza from public sources)
2. **Exploratory analysis** (epidemic curves, geographic spread, age demographics)
3. **SIR/SEIR models** (simulate disease transmission, 10-15 minutes)
4. **R0 estimation** (reproduction number from early epidemic growth)
5. **Intervention analysis** (model social distancing, vaccination impact)
6. **Forecasting** (7-14 day ahead case predictions)

## Dataset

**Public Health Surveillance Data**
- **COVID-19:** Johns Hopkins CSSE daily case/death data (2020-2023)
- **Influenza:** CDC ILINet surveillance (10+ years)
- **Demographics:** Age-stratified incidence data
- **Geographic:** County/state-level spatial data
- Size: ~200MB total
- Format: CSV time series
- Sources: CDC, WHO, Johns Hopkins, Our World in Data
- **Privacy:** Aggregated public data only (no individual records)

**Variables:**
- Daily cases, hospitalizations, deaths
- Testing rates and positivity
- Vaccination coverage
- Population demographics
- Geographic metadata

## Colab Considerations

This notebook works on Colab but you'll notice:
- **Simple models** (deterministic SIR/SEIR only, no stochastic)
- **Single population** (no age-structured or spatial models)
- **Basic parameter estimation** (curve fitting vs. proper Bayesian inference)
- **Limited interventions** (simple scenarios only)
- **No uncertainty quantification** (no Monte Carlo simulations)

These limitations prevent sophisticated epidemiological research.

## What's Included

- Single Jupyter notebook (`epidemic-modeling.ipynb`)
- COVID-19 and influenza data access
- SIR/SEIR ODE solvers (scipy.integrate)
- R0 estimation methods (exponential growth, generation time)
- Intervention scenario modeling
- Time series forecasting
- Visualization (epidemic curves, phase portraits)

## Key Methods

- **Compartmental Models:** SIR, SEIR, SEIRS with demographics
- **ODE Solvers:** Numerical integration of epidemic dynamics
- **R0 Estimation:** Exponential growth rate, generation interval methods
- **Parameter Fitting:** Least squares, likelihood-based estimation
- **Intervention Modeling:** Contact rate reduction, vaccination
- **Time Series Forecasting:** ARIMA, exponential smoothing
- **Sensitivity Analysis:** Impact of parameter uncertainty

## Model Outputs

1. **Epidemic Curves:** Predicted cases, hospitalizations, deaths over time
2. **R0 Estimates:** Basic reproduction number with confidence intervals
3. **Intervention Impact:** Reduction in peak cases, deaths averted
4. **Forecasts:** 7-14 day ahead predictions with prediction intervals
5. **Phase Portraits:** Susceptible-Infected dynamics visualization
6. **Attack Rate:** Final proportion of population infected

## Next Steps

**Need sophisticated modeling?** This project shows basic SIR/SEIR:

- **Tier 1:** [Advanced epidemiological models](../tier-1/) on Studio Lab
  - Age-structured models (10+ age groups)
  - Stochastic simulations (1000+ realizations)
  - Spatial metapopulation models
  - Bayesian parameter inference (MCMC)
  - Multi-pathogen dynamics

- **Tier 2:** [AWS-integrated surveillance](../tier-2/) with real-time data
  - Automated daily data ingestion (Lambda)
  - Real-time R_t estimation
  - Ensemble forecasting models
  - SageMaker for ML-enhanced forecasts
  - S3 data lake for multi-year historical data

- **Tier 3:** [Production surveillance system](../tier-3/) with CloudFormation
  - State/national-level deployment
  - Real-time dashboards (QuickSight)
  - Automated alert systems
  - Agent-based models (AWS Batch for 1M+ agents)
  - Integration with public health reporting systems
  - Nowcasting with incomplete data

## Requirements

Pre-installed in Colab and Studio Lab:
- Python 3.9+
- scipy (ODE solvers)
- pandas, numpy
- scikit-learn (parameter estimation)
- matplotlib, seaborn, plotly
- statsmodels (time series)

**Data Access:** Public domain surveillance data, no IRB required
