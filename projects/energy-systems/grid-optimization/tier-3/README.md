# Smart Grid Optimization and Energy Systems - AWS Research Jumpstart

**Tier 1 Flagship Project**

Optimize power grid operations, integrate renewable energy sources, forecast electricity demand, and manage distributed energy resources at scale using machine learning, optimization algorithms, and IoT on AWS. Enable the transition to clean, reliable, and resilient smart grids.

## Overview

This flagship project demonstrates how to build smart grid optimization and energy management systems using AWS services. We'll work with smart meter data, renewable energy forecasts, weather data, grid sensor telemetry, and energy market information to optimize power flows, predict demand, integrate renewables, and coordinate distributed energy resources (DER).

### Key Features

- **Load forecasting:** LSTM, Prophet, SARIMAX for electricity demand prediction
- **Renewable forecasting:** Solar/wind power prediction with weather integration
- **Grid optimization:** Optimal power flow (OPF), unit commitment, economic dispatch
- **Energy storage:** Battery optimization for arbitrage and grid services
- **EV charging:** Smart charging coordination to minimize grid stress
- **Demand response:** Price-responsive load management and peak shaving
- **Anomaly detection:** Predictive maintenance for grid equipment
- **AWS services:** IoT Core, Timestream, SageMaker, Batch, Lambda, Kinesis

### Scientific Applications

1. **Demand forecasting:** Predict electricity load hours to days in advance for grid planning
2. **Renewable integration:** Maximize solar/wind usage while maintaining grid stability
3. **Grid optimization:** Minimize generation costs while meeting constraints
4. **Energy storage:** Optimize battery dispatch for revenue and grid services
5. **Resilience:** Detect equipment failures, predict outages, coordinate microgrids

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│            Smart Grid Optimization Pipeline                      │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ Smart Meters │      │ Grid Sensors │      │ Weather APIs │
│ (AMI)        │─────▶│ (SCADA)      │─────▶│ (NOAA, OW)   │
└──────────────┘      └──────────────┘      └──────────────┘
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   IoT Core        │
                    │  (Telemetry       │
                    │   ingestion)      │
                    └─────────┬─────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼───────┐   ┌─────────▼─────────┐   ┌─────▼──────┐
│ Kinesis       │   │ Timestream        │   │ S3 Data    │
│ (Streaming)   │   │ (Time series DB)  │   │ Lake       │
└───────┬───────┘   └─────────┬─────────┘   └─────┬──────┘
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼──────┐   ┌──────────▼────────┐   ┌───────▼───────┐
│ SageMaker    │   │ AWS Batch         │   │ Lambda        │
│ (ML models)  │   │ (Optimization)    │   │ (Real-time)   │
└───────┬──────┘   └──────────┬────────┘   └───────┬───────┘
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼──────┐   ┌──────────▼────────┐   ┌───────▼───────┐
│ Load         │   │ Renewable         │   │ Grid          │
│ Forecasting  │   │ Forecasting       │   │ Optimization  │
└───────┬──────┘   └──────────┬────────┘   └───────┬───────┘
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │ QuickSight/Grafana│
                    │ Grid Dashboards & │
                    │ Control Signals   │
                    └───────────────────┘
```

## Major Data Sources

### 1. Smart Meter Data (AMI - Advanced Metering Infrastructure)

**Household/commercial consumption:**
- **Resolution:** 15-minute to hourly readings
- **Variables:** Active power (kW), reactive power (kVAR), voltage, current
- **Scale:** Millions of meters per utility
- **Privacy:** Aggregation required, anonymization
- **Format:** Time series with meter ID, timestamp, consumption

**Use cases:**
- Load forecasting (aggregate neighborhood/feeder level)
- Demand response verification
- Outage detection
- Non-technical loss detection (theft)

### 2. Grid Sensor Data (SCADA - Supervisory Control and Data Acquisition)

**Substation and feeder monitoring:**
- **Sensors:** Phasor measurement units (PMU), voltage/current transformers
- **Frequency:** 1-60 samples per second for PMUs
- **Variables:** Voltage magnitude, phase angle, frequency, power flows
- **Applications:** State estimation, fault detection, stability monitoring

**Equipment monitoring:**
- Transformer temperature, oil level, load
- Circuit breaker status
- Capacitor bank status

### 3. Weather Data

**For renewable forecasting and load prediction:**

**NOAA/NWS:**
- Temperature, humidity, precipitation
- Wind speed/direction
- Cloud cover
- Solar radiation (GHI, DNI, DHI)
- Numerical Weather Prediction (NWP) models

**OpenWeatherMap API:**
- Historical and forecast data
- Resolution: City-level to grid points
- Updates: Hourly

**NREL NSRDB (National Solar Radiation Database):**
- High-quality solar resource data
- 4km resolution for U.S.
- 30-minute intervals

### 4. Energy Market Data

**ISO/RTO price signals:**
- **CAISO (California):** http://oasis.caiso.com/
- **PJM (Mid-Atlantic):** https://www.pjm.com/
- **ERCOT (Texas):** http://www.ercot.com/
- **Variables:** Locational marginal prices (LMP), ancillary service prices
- **Frequency:** 5-minute to hourly

**EIA (Energy Information Administration):**
- Generation mix by fuel type
- Demand by region
- Wholesale power prices
- Historical data: https://www.eia.gov/opendata/

### 5. Renewable Energy Data

**Solar installations:**
- PV system monitoring (SolarEdge, Enphase APIs)
- Individual panel/inverter data
- Aggregate fleet performance

**Wind farms:**
- Turbine power output
- Wind speed at hub height
- Curtailment events

**NREL APIs:**
- PVWatts for solar estimation
- Wind Toolkit for wind resource data
- System Advisor Model (SAM) data

### 6. Energy Storage and EV Data

**Battery energy storage systems (BESS):**
- State of charge (SOC)
- Charge/discharge power
- Temperature, cycle count
- Available capacity

**Electric vehicle charging:**
- Charging session data (start/end time, energy delivered)
- Vehicle arrival/departure patterns
- State of charge at arrival
- Desired departure time

## Getting Started

### Prerequisites

```bash
# AWS CLI
aws --version

# Python dependencies
pip install -r requirements.txt
```

**requirements.txt:**
```
boto3>=1.26.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
torch>=2.0.0
tensorflow>=2.12.0
prophet>=1.1.0
statsmodels>=0.14.0
pyomo>=6.5.0
cvxpy>=1.3.0
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.14.0
requests>=2.28.0
pvlib>=0.9.0
```

### Deploy Infrastructure

```bash
# Deploy CloudFormation stack
aws cloudformation create-stack \
  --stack-name smart-grid-stack \
  --template-body file://cloudformation/grid-stack.yml \
  --parameters file://cloudformation/parameters.json \
  --capabilities CAPABILITY_NAMED_IAM

# Wait for completion
aws cloudformation wait stack-create-complete \
  --stack-name smart-grid-stack

# Get outputs
aws cloudformation describe-stacks \
  --stack-name smart-grid-stack \
  --query 'Stacks[0].Outputs'
```

### Set Up IoT Core for Smart Meters

```bash
# Create IoT thing type for smart meters
aws iot create-thing-type \
  --thing-type-name SmartMeter \
  --thing-type-properties "thingTypeDescription=Smart electricity meters"

# Create IoT rule to route data to Timestream
aws iot create-topic-rule \
  --rule-name SmartMeterToTimestream \
  --topic-rule-payload file://iot-rules/meter-to-timestream.json
```

### Initialize Data Lake

```python
from src.grid_data import GridDataManager

# Initialize data manager
data_mgr = GridDataManager()

# Create S3 structure
data_mgr.initialize_data_lake(
    bucket='smart-grid-data-lake',
    structure={
        'raw': ['smart_meters', 'scada', 'weather', 'market_prices'],
        'processed': ['load_forecasts', 'renewable_forecasts', 'optimization_results'],
        'models': ['trained_models', 'model_artifacts']
    }
)

# Set up Timestream database
data_mgr.create_timestream_database(
    database_name='GridTimeSeries',
    tables=['SmartMeterReadings', 'ScadaData', 'WeatherData', 'ForecastResults']
)
```

## Core Analyses

### 1. Electricity Load Forecasting with LSTM

Predict hourly electricity demand 24-72 hours ahead using deep learning.

```python
from src.load_forecasting import LoadForecaster
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Initialize forecaster
forecaster = LoadForecaster()

# Load historical load data
historical_load = forecaster.load_historical_data(
    source='s3://smart-grid-data-lake/raw/load_history/',
    utility='ACME_UTILITY',
    start_date='2020-01-01',
    end_date='2024-11-13',
    resolution='hourly'
)

print(f"Loaded {len(historical_load)} hourly observations")
print(f"Date range: {historical_load.index[0]} to {historical_load.index[-1]}")

# Load weather data (correlated with load)
weather_data = forecaster.load_weather_data(
    location='utility_service_area',
    variables=['temperature', 'humidity', 'cloud_cover', 'wind_speed'],
    source='noaa'
)

# Feature engineering
features = forecaster.create_features(
    load_data=historical_load,
    weather_data=weather_data,
    feature_types=[
        'temporal',      # Hour, day of week, month, year
        'calendar',      # Holidays, weekends, DST transitions
        'weather',       # Temperature, humidity, etc.
        'lagged_load',   # Previous hours/days consumption
        'rolling_stats'  # Moving averages, std dev
    ]
)

# Key features for load forecasting:
# - Temperature (strong correlation with heating/cooling load)
# - Hour of day (daily pattern)
# - Day of week (weekday vs weekend)
# - Month/season (seasonal patterns)
# - Holidays (reduced commercial load)
# - Recent load history (momentum)

# Split data: train on 2020-2022, validate on 2023, test on 2024
train_data = features['2020-01-01':'2022-12-31']
val_data = features['2023-01-01':'2023-12-31']
test_data = features['2024-01-01':'2024-11-13']

print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")
print(f"Test samples: {len(test_data)}")

# Prepare sequences for LSTM (look back 168 hours = 1 week)
X_train, y_train = forecaster.create_sequences(
    train_data,
    sequence_length=168,  # 1 week of hourly data
    forecast_horizon=24    # Predict 24 hours ahead
)

X_val, y_val = forecaster.create_sequences(
    val_data,
    sequence_length=168,
    forecast_horizon=24
)

# Build LSTM model
model = forecaster.build_lstm_model(
    sequence_length=168,
    n_features=X_train.shape[2],
    forecast_horizon=24,
    architecture={
        'lstm_layers': [128, 64],
        'dropout': 0.2,
        'dense_layers': [32],
        'activation': 'relu',
        'optimizer': 'adam',
        'loss': 'mse'
    }
)

# Train on SageMaker
training_job = forecaster.train_on_sagemaker(
    model=model,
    train_data=(X_train, y_train),
    val_data=(X_val, y_val),
    instance_type='ml.p3.2xlarge',  # GPU for faster training
    hyperparameters={
        'epochs': 50,
        'batch_size': 256,
        'learning_rate': 0.001,
        'early_stopping_patience': 10
    }
)

# Evaluate on test set
test_predictions = forecaster.predict(model, X_test)

metrics = forecaster.evaluate(
    y_true=y_test,
    y_pred=test_predictions,
    metrics=['mape', 'rmse', 'mae', 'r2']
)

print("\nForecasting Performance:")
print(f"MAPE (Mean Absolute Percentage Error): {metrics['mape']:.2f}%")
print(f"RMSE (Root Mean Squared Error): {metrics['rmse']:.2f} MW")
print(f"MAE (Mean Absolute Error): {metrics['mae']:.2f} MW")
print(f"R² Score: {metrics['r2']:.4f}")

# Typical performance:
# Day-ahead (24h): MAPE 2-5%
# Week-ahead: MAPE 5-10%

# Real-time forecasting with Lambda
endpoint = forecaster.deploy_endpoint(
    model=model,
    instance_type='ml.c5.xlarge',
    endpoint_name='load-forecasting-prod'
)

# Generate current forecast
current_forecast = forecaster.forecast_realtime(
    endpoint=endpoint,
    current_datetime=datetime.now(),
    horizon_hours=24,
    weather_forecast=weather_forecast  # From NOAA API
)

print(f"\nNext 24-hour load forecast:")
print(current_forecast[['datetime', 'predicted_load_mw', 'confidence_lower', 'confidence_upper']])

# Visualize forecast vs actual
forecaster.plot_forecast(
    actual=test_data['load_mw'],
    predicted=test_predictions,
    confidence_intervals=True,
    save_path='load_forecast_results.png'
)

# Peak demand prediction (critical for grid planning)
peak_analysis = forecaster.analyze_peak_demand(
    forecast=current_forecast,
    threshold=0.95  # 95th percentile of historical load
)

if peak_analysis['peak_expected']:
    print(f"\n⚠️  Peak demand event expected:")
    print(f"   Time: {peak_analysis['peak_time']}")
    print(f"   Predicted load: {peak_analysis['peak_load_mw']:.0f} MW")
    print(f"   System capacity: {peak_analysis['capacity_mw']:.0f} MW")
    print(f"   Reserve margin: {peak_analysis['reserve_margin_pct']:.1f}%")
```

### 2. Solar Power Forecasting

Predict solar PV generation using weather forecasts and historical data.

```python
from src.solar_forecasting import SolarForecaster
import pvlib
from pvlib.location import Location

forecaster = SolarForecaster()

# Define solar installation
solar_site = Location(
    latitude=34.05,
    longitude=-118.25,
    tz='America/Los_Angeles',
    altitude=100,
    name='LA_Solar_Farm'
)

# Solar system specifications
system_specs = {
    'capacity_mw': 50,
    'module_type': 'Canadian_Solar_CS6K_280M',
    'inverter_type': 'SMA_America__SB240_US_10',
    'tilt': 25,  # degrees
    'azimuth': 180,  # South-facing
    'tracking': False  # Fixed tilt
}

# Historical solar production data
historical_production = forecaster.load_production_data(
    site_id='LA_Solar_Farm',
    start_date='2021-01-01',
    end_date='2024-11-13',
    resolution='15min'
)

# Historical weather data
historical_weather = forecaster.load_weather_data(
    location=solar_site,
    variables=[
        'ghi',          # Global Horizontal Irradiance
        'dni',          # Direct Normal Irradiance
        'dhi',          # Diffuse Horizontal Irradiance
        'temperature',
        'wind_speed',
        'cloud_cover'
    ],
    source='nrel_nsrdb'
)

# Physics-based model (PVWatts baseline)
baseline_model = forecaster.create_pvwatts_model(
    site=solar_site,
    system_specs=system_specs
)

# Calculate clear-sky irradiance
clearsky = solar_site.get_clearsky(
    times=historical_weather.index,
    model='ineichen'
)

# Persistence model (naive baseline: tomorrow = today)
persistence_forecast = forecaster.persistence_model(
    historical_production,
    horizon_hours=24
)

# Machine learning model (improve on physics-based)
features = forecaster.engineer_features(
    weather_data=historical_weather,
    production_data=historical_production,
    site=solar_site,
    feature_types=[
        'solar_position',   # Sun elevation, azimuth
        'clearsky_index',   # Actual GHI / clearsky GHI
        'weather',          # Temperature, wind, clouds
        'temporal',         # Hour, day of year, season
        'lagged_production' # Recent production history
    ]
)

# Train gradient boosting model
X = features.drop(columns=['production_mw'])
y = features['production_mw']

X_train, X_test = X['2021':'2023'], X['2024']
y_train, y_test = y['2021':'2023'], y['2024']

model = forecaster.train_ml_model(
    X_train, y_train,
    model_type='xgboost',
    hyperparameters={
        'n_estimators': 500,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8
    }
)

# Evaluate models
persistence_metrics = forecaster.evaluate(y_test, persistence_forecast)
ml_metrics = forecaster.evaluate(y_test, model.predict(X_test))

print("Solar Forecasting Performance (24-hour ahead):")
print("\nPersistence Model (baseline):")
print(f"  RMSE: {persistence_metrics['rmse']:.2f} MW ({persistence_metrics['rmse_pct']:.1f}%)")
print(f"  MAE: {persistence_metrics['mae']:.2f} MW")

print("\nML Model (XGBoost + weather forecast):")
print(f"  RMSE: {ml_metrics['rmse']:.2f} MW ({ml_metrics['rmse_pct']:.1f}%)")
print(f"  MAE: {ml_metrics['mae']:.2f} MW")
print(f"  Improvement: {(1 - ml_metrics['rmse']/persistence_metrics['rmse'])*100:.1f}%")

# Typical performance:
# Intraday (1-6h): 5-10% RMSE
# Day-ahead (24h): 10-20% RMSE
# 2-day ahead: 15-25% RMSE

# Probabilistic forecast (quantile regression for uncertainty)
probabilistic_model = forecaster.train_quantile_model(
    X_train, y_train,
    quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]
)

# Get weather forecast from NOAA
weather_forecast = forecaster.get_weather_forecast(
    location=solar_site,
    horizon_hours=48,
    source='noaa_gfs'
)

# Generate solar forecast
solar_forecast = forecaster.forecast(
    model=model,
    probabilistic_model=probabilistic_model,
    weather_forecast=weather_forecast,
    site=solar_site,
    system_specs=system_specs
)

print("\nNext 48-hour solar forecast:")
print(solar_forecast[['datetime', 'forecast_mw', 'p10', 'p50', 'p90', 'ghi_forecast']])

# Ramp event detection (rapid changes in solar output)
ramp_events = forecaster.detect_ramp_events(
    solar_forecast,
    threshold_mw=10,  # 10 MW change
    duration_min=15    # Within 15 minutes
)

if len(ramp_events) > 0:
    print(f"\n⚠️  {len(ramp_events)} ramp events expected in next 48h:")
    for event in ramp_events[:3]:
        print(f"   {event['time']}: {event['change_mw']:+.1f} MW "
              f"({'down-ramp' if event['change_mw'] < 0 else 'up-ramp'})")

# Fleet-level forecast (aggregate multiple sites)
fleet_forecast = forecaster.aggregate_fleet_forecast(
    site_ids=['LA_Solar_Farm', 'SD_Solar_Farm', 'Phoenix_Solar'],
    weather_forecast=weather_forecast,
    correlation_model='copula'  # Account for spatial correlation
)
```

### 3. Battery Energy Storage Optimization

Optimize battery charging/discharging for revenue maximization and grid services.

```python
from src.battery_optimization import BatteryOptimizer
import cvxpy as cp
import numpy as np

optimizer = BatteryOptimizer()

# Battery specifications
battery_specs = {
    'capacity_mwh': 20,           # Energy capacity
    'max_charge_mw': 5,           # Maximum charge rate
    'max_discharge_mw': 5,        # Maximum discharge rate
    'charge_efficiency': 0.95,    # Round-trip efficiency (sqrt)
    'discharge_efficiency': 0.95,
    'min_soc': 0.1,              # Minimum state of charge (10%)
    'max_soc': 0.9,              # Maximum state of charge (90%)
    'initial_soc': 0.5,          # Starting at 50%
    'degradation_cost': 0.50     # $/MWh (cycle degradation)
}

# Load electricity price forecast (day-ahead market)
prices = optimizer.load_price_forecast(
    market='CAISO',
    location='SP15',  # Southern California
    horizon_hours=24,
    resolution='hourly'
)

print(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}/MWh")
print(f"Price volatility: ${prices.std():.2f}/MWh")

# Optimization problem: maximize profit from energy arbitrage
# Decision variables
T = len(prices)  # Time periods
charge = cp.Variable(T, nonneg=True)      # Charge power (MW)
discharge = cp.Variable(T, nonneg=True)   # Discharge power (MW)
soc = cp.Variable(T+1)                    # State of charge (MWh)

# Objective: maximize revenue - costs
revenue = cp.sum(cp.multiply(discharge, prices))  # Revenue from selling
cost = cp.sum(cp.multiply(charge, prices))         # Cost of buying
degradation = battery_specs['degradation_cost'] * cp.sum(discharge)
objective = cp.Maximize(revenue - cost - degradation)

# Constraints
constraints = []

# State of charge dynamics
constraints.append(soc[0] == battery_specs['initial_soc'] * battery_specs['capacity_mwh'])
for t in range(T):
    constraints.append(
        soc[t+1] == soc[t]
        + charge[t] * battery_specs['charge_efficiency']
        - discharge[t] / battery_specs['discharge_efficiency']
    )

# SOC limits
constraints.append(soc >= battery_specs['min_soc'] * battery_specs['capacity_mwh'])
constraints.append(soc <= battery_specs['max_soc'] * battery_specs['capacity_mwh'])

# Power limits
constraints.append(charge <= battery_specs['max_charge_mw'])
constraints.append(discharge <= battery_specs['max_discharge_mw'])

# Cannot charge and discharge simultaneously
# (Approximate with sum constraint: charge + discharge <= max_power)
constraints.append(charge + discharge <= battery_specs['max_charge_mw'])

# Solve optimization
problem = cp.Problem(objective, constraints)
result = problem.solve(solver=cp.GLPK)

print(f"\nOptimization Results:")
print(f"Objective value (profit): ${result:.2f}")
print(f"Revenue from discharge: ${np.sum(discharge.value * prices):.2f}")
print(f"Cost of charging: ${np.sum(charge.value * prices):.2f}")
print(f"Degradation cost: ${battery_specs['degradation_cost'] * np.sum(discharge.value):.2f}")

# Analyze dispatch schedule
schedule = optimizer.create_schedule(
    times=prices.index,
    charge=charge.value,
    discharge=discharge.value,
    soc=soc.value,
    prices=prices
)

print("\nDispatch Schedule:")
print(schedule[schedule['action'] != 'idle'][['hour', 'action', 'power_mw', 'soc_pct', 'price']])

# Strategy interpretation:
# - Charge during low-price hours (typically overnight, high renewable generation)
# - Discharge during high-price hours (typically evening peak)
# - Stay idle during moderate prices

# Multi-objective optimization: arbitrage + frequency regulation
freq_reg_prices = optimizer.load_ancillary_prices(
    market='CAISO',
    service='regulation_up',
    horizon_hours=24
)

# Frequency regulation constraints:
# - Must maintain SOC buffer to provide service
# - Provides additional revenue stream
# - Faster response requirements

multi_obj_problem = optimizer.multi_objective_optimization(
    battery_specs=battery_specs,
    energy_prices=prices,
    freq_reg_prices=freq_reg_prices,
    objectives={
        'arbitrage': 0.6,      # 60% weight
        'freq_regulation': 0.4  # 40% weight
    }
)

multi_obj_result = multi_obj_problem.solve()
print(f"\nMulti-objective profit: ${multi_obj_result['profit']:.2f}")
print(f"  Arbitrage revenue: ${multi_obj_result['arbitrage_revenue']:.2f}")
print(f"  Freq reg revenue: ${multi_obj_result['freq_reg_revenue']:.2f}")

# Degradation modeling (cycle life)
degradation_model = optimizer.estimate_degradation(
    schedule=schedule,
    battery_chemistry='lithium_ion_nmc',
    initial_capacity_mwh=battery_specs['capacity_mwh'],
    years=10
)

print(f"\nBattery Degradation (10-year projection):")
print(f"  Total cycles: {degradation_model['total_cycles']:.0f}")
print(f"  Equivalent full cycles/year: {degradation_model['cycles_per_year']:.0f}")
print(f"  Capacity retention at year 10: {degradation_model['capacity_retention_pct']:.1f}%")
print(f"  Replacement year: {degradation_model['replacement_year']}")

# ROI analysis
roi = optimizer.calculate_roi(
    battery_specs=battery_specs,
    annual_revenue=result * 365,  # Scale daily profit to annual
    capital_cost=400_000,  # $/MWh * capacity (typical: $300-500k/MWh)
    om_cost=10_000,        # $/year operations & maintenance
    replacement_cost=250_000,  # Reduced cost for replacement
    project_lifetime=20    # years
)

print(f"\nROI Analysis:")
print(f"  Capital cost: ${roi['capital_cost']:,.0f}")
print(f"  Annual revenue: ${roi['annual_revenue']:,.0f}")
print(f"  NPV (Net Present Value): ${roi['npv']:,.0f}")
print(f"  IRR (Internal Rate of Return): {roi['irr']:.1f}%")
print(f"  Payback period: {roi['payback_years']:.1f} years")
```

### 4. Electric Vehicle Charging Coordination

Optimize EV charging schedules to minimize grid stress and cost.

```python
from src.ev_charging import EVChargingOptimizer
import pandas as pd

optimizer = EVChargingOptimizer()

# EV charging station fleet
charging_stations = [
    {
        'station_id': f'CS_{i:03d}',
        'location': f'location_{i%10}',  # 10 locations
        'num_ports': 4,
        'power_per_port_kw': 7.2,  # Level 2 charging
        'total_capacity_kw': 28.8
    }
    for i in range(50)  # 50 stations, 200 ports total
]

# Simulate EV arrivals (based on historical patterns)
ev_sessions = optimizer.simulate_ev_arrivals(
    stations=charging_stations,
    date='2024-11-14',
    arrival_patterns={
        'morning_commute': {'time': '07:00-09:00', 'probability': 0.3},
        'work_day': {'time': '09:00-17:00', 'probability': 0.2},
        'evening_commute': {'time': '17:00-19:00', 'probability': 0.4},
        'overnight': {'time': '19:00-07:00', 'probability': 0.1}
    },
    n_sessions=500  # 500 charging sessions expected
)

# Each session has:
# - arrival_time: When EV plugs in
# - departure_time: When EV must leave
# - soc_arrival: Battery state of charge at arrival (%)
# - soc_desired: Desired SOC at departure (%)
# - battery_capacity_kwh: Vehicle battery size
# - max_charge_rate_kw: Vehicle charging limit

print(f"Simulated {len(ev_sessions)} EV charging sessions")

# Load electricity price forecast
prices = optimizer.load_price_forecast(
    date='2024-11-14',
    resolution='hourly'
)

# Grid constraints
grid_constraints = {
    'transformer_capacity_kw': 1000,  # Feeder transformer limit
    'peak_demand_limit_kw': 800,      # Avoid exceeding this
    'baseload_kw': 300                # Non-EV load on same feeder
}

# Optimization objective: minimize total charging cost + peak demand charges
# Decision variables: charge_rate[session, time] for each EV session

ev_sessions_df = pd.DataFrame(ev_sessions)

optimal_schedule = optimizer.optimize_charging(
    ev_sessions=ev_sessions_df,
    prices=prices,
    grid_constraints=grid_constraints,
    objectives={
        'minimize_cost': 0.5,        # Energy cost
        'minimize_peak': 0.3,        # Peak demand charges
        'maximize_convenience': 0.2  # Charge early if possible
    }
)

print("\nCharging Optimization Results:")
print(f"Total energy delivered: {optimal_schedule['total_energy_kwh']:.1f} kWh")
print(f"Total cost: ${optimal_schedule['total_cost']:.2f}")
print(f"Average cost per session: ${optimal_schedule['cost_per_session']:.2f}")
print(f"Peak power demand: {optimal_schedule['peak_demand_kw']:.1f} kW")
print(f"Transformer utilization: {optimal_schedule['transformer_utilization_pct']:.1f}%")

# Compare with uncontrolled charging (immediate charging)
uncontrolled = optimizer.uncontrolled_charging(
    ev_sessions=ev_sessions_df,
    prices=prices
)

savings = {
    'cost_savings': uncontrolled['total_cost'] - optimal_schedule['total_cost'],
    'peak_reduction': uncontrolled['peak_demand_kw'] - optimal_schedule['peak_demand_kw'],
    'cost_savings_pct': (1 - optimal_schedule['total_cost']/uncontrolled['total_cost']) * 100
}

print(f"\nSavings vs Uncontrolled Charging:")
print(f"  Cost savings: ${savings['cost_savings']:.2f} ({savings['cost_savings_pct']:.1f}%)")
print(f"  Peak reduction: {savings['peak_reduction']:.1f} kW ({savings['peak_reduction']/uncontrolled['peak_demand_kw']*100:.1f}%)")

# Visualize charging profile
optimizer.plot_charging_profile(
    optimal_schedule,
    uncontrolled_schedule=uncontrolled,
    prices=prices,
    grid_limit=grid_constraints['peak_demand_limit_kw'],
    save_path='ev_charging_profile.png'
)

# Real-time charging control (V2G - Vehicle to Grid potential)
v2g_enabled_sessions = ev_sessions_df[ev_sessions_df['v2g_capable'] == True]

if len(v2g_enabled_sessions) > 0:
    v2g_schedule = optimizer.optimize_v2g(
        v2g_sessions=v2g_enabled_sessions,
        prices=prices,
        discharge_price_threshold=100,  # $/MWh - only discharge if price exceeds this
        max_discharge_pct=0.2,          # Maximum 20% discharge to preserve battery
        grid_services=['frequency_regulation', 'peak_shaving']
    )

    print(f"\nV2G Results:")
    print(f"  Energy discharged: {v2g_schedule['energy_discharged_kwh']:.1f} kWh")
    print(f"  Revenue from discharge: ${v2g_schedule['discharge_revenue']:.2f}")
    print(f"  Net savings: ${v2g_schedule['net_savings']:.2f}")

# User notification system
notifications = optimizer.generate_notifications(
    optimal_schedule,
    ev_sessions=ev_sessions_df,
    notification_types=['charging_start', 'charging_complete', 'schedule_change']
)

print(f"\nGenerated {len(notifications)} user notifications")
```

### 5. Real-Time Grid Monitoring and Anomaly Detection

Detect equipment failures and grid anomalies before they cause outages.

```python
from src.grid_monitoring import GridAnomalyDetector
import boto3

detector = GridAnomalyDetector()

# Connect to IoT Core for real-time sensor data
iot_client = boto3.client('iot-data', region_name='us-east-1')

# Define grid assets to monitor
grid_assets = [
    {
        'asset_id': 'TX_001',
        'asset_type': 'transformer',
        'location': 'Substation_A',
        'rated_capacity_mva': 50,
        'sensors': ['temperature', 'oil_level', 'dissolved_gas', 'load']
    },
    {
        'asset_id': 'TX_002',
        'asset_type': 'transformer',
        'location': 'Substation_B',
        'rated_capacity_mva': 50,
        'sensors': ['temperature', 'oil_level', 'dissolved_gas', 'load']
    },
    # ... more assets
]

# Load historical sensor data for training
historical_data = detector.load_sensor_history(
    asset_ids=[asset['asset_id'] for asset in grid_assets],
    start_date='2022-01-01',
    end_date='2024-11-13',
    resolution='1min'
)

print(f"Loaded {len(historical_data)} sensor readings")

# Train anomaly detection model (LSTM Autoencoder)
# Normal operation: low reconstruction error
# Anomaly: high reconstruction error

train_data = historical_data['2022':'2023']
val_data = historical_data['2024-01':'2024-06']

# Prepare sequences for LSTM
sequences = detector.create_sequences(
    train_data,
    sequence_length=60,  # 60-minute windows
    stride=1
)

# Build LSTM autoencoder
model = detector.build_lstm_autoencoder(
    sequence_length=60,
    n_features=len(train_data.columns),
    architecture={
        'encoder_layers': [64, 32, 16],
        'latent_dim': 8,
        'decoder_layers': [16, 32, 64],
        'dropout': 0.2
    }
)

# Train on SageMaker
training_job = detector.train_autoencoder(
    model=model,
    train_data=sequences,
    val_data=val_data,
    instance_type='ml.p3.2xlarge',
    epochs=50
)

# Determine anomaly threshold (99th percentile of reconstruction error)
reconstruction_errors = detector.calculate_reconstruction_error(
    model,
    val_data
)

threshold = np.percentile(reconstruction_errors, 99)
print(f"Anomaly threshold (99th percentile): {threshold:.4f}")

# Alternative: Isolation Forest (simpler, faster)
isolation_forest = detector.train_isolation_forest(
    train_data,
    contamination=0.01  # Expect 1% anomalies
)

# Deploy anomaly detection pipeline
# 1. Stream sensor data from IoT Core
# 2. Process with Lambda
# 3. Detect anomalies in real-time
# 4. Alert via SNS

lambda_code = '''
import json
import boto3
import numpy as np
import pickle

# Load model from S3
s3 = boto3.client('s3')
model_data = s3.get_object(Bucket='models', Key='anomaly_detector.pkl')
model = pickle.loads(model_data['Body'].read())

sns = boto3.client('sns')

def lambda_handler(event, context):
    # Parse IoT message
    sensor_data = json.loads(event['body'])

    # Extract features
    features = np.array([
        sensor_data['temperature'],
        sensor_data['oil_level'],
        sensor_data['load_pct'],
        sensor_data['vibration']
    ]).reshape(1, -1)

    # Detect anomaly
    is_anomaly = model.predict(features)[0] == -1
    anomaly_score = model.score_samples(features)[0]

    if is_anomaly:
        # Send alert
        message = f"""
        ALERT: Anomaly detected on {sensor_data['asset_id']}

        Asset: {sensor_data['asset_id']}
        Location: {sensor_data['location']}
        Timestamp: {sensor_data['timestamp']}
        Anomaly Score: {anomaly_score:.4f}

        Sensor Readings:
        - Temperature: {sensor_data['temperature']:.1f}°C
        - Oil Level: {sensor_data['oil_level']:.1f}%
        - Load: {sensor_data['load_pct']:.1f}%
        - Vibration: {sensor_data['vibration']:.2f}

        Recommended Action: Inspect equipment immediately
        """

        sns.publish(
            TopicArn='arn:aws:sns:us-east-1:123456789:grid-alerts',
            Subject=f'Grid Anomaly: {sensor_data["asset_id"]}',
            Message=message
        )

    return {'statusCode': 200, 'body': json.dumps({'anomaly': bool(is_anomaly)})}
'''

# Create Lambda function for real-time detection
lambda_function = detector.create_lambda_function(
    function_name='GridAnomalyDetector',
    code=lambda_code,
    model=isolation_forest,
    timeout=30
)

# Create IoT rule to trigger Lambda
iot_rule = detector.create_iot_rule(
    rule_name='SensorAnomalyDetection',
    topic_filter='grid/sensors/+/data',  # All sensor topics
    lambda_function_arn=lambda_function['FunctionArn']
)

print(f"Real-time anomaly detection deployed")
print(f"Lambda function: {lambda_function['FunctionArn']}")
print(f"IoT rule: {iot_rule['ruleName']}")

# Predictive maintenance: estimate remaining useful life (RUL)
rul_model = detector.train_rul_model(
    historical_data=historical_data,
    failure_events=failure_log,  # Historical failures
    model_type='survival_analysis'  # or 'gradient_boosting'
)

# Predict RUL for current assets
current_state = detector.get_current_sensor_readings(
    asset_ids=[asset['asset_id'] for asset in grid_assets]
)

rul_predictions = rul_model.predict(current_state)

print("\nRemaining Useful Life Predictions:")
for asset_id, rul_days in rul_predictions.items():
    if rul_days < 90:  # Alert if < 90 days
        print(f"  ⚠️  {asset_id}: {rul_days:.0f} days (schedule maintenance)")
    else:
        print(f"  ✓  {asset_id}: {rul_days:.0f} days (healthy)")

# Impact analysis: what if this asset fails?
impact = detector.analyze_outage_impact(
    asset_id='TX_001',
    scenario='failure',
    time='2024-11-14 18:00'  # Evening peak
)

print(f"\nOutage Impact Analysis (TX_001 failure):")
print(f"  Customers affected: {impact['customers_affected']:,}")
print(f"  Load lost: {impact['load_lost_mw']:.1f} MW")
print(f"  Backup capacity available: {impact['backup_available']}")
print(f"  Estimated restoration time: {impact['restoration_hours']:.1f} hours")
print(f"  Estimated cost: ${impact['estimated_cost']:,.0f}")
```

### 6. Optimal Power Flow (OPF)

Optimize generator dispatch to minimize cost while meeting constraints.

```python
from src.optimal_power_flow import OPFOptimizer
import pyomo.environ as pyo

optimizer = OPFOptimizer()

# Define power system (simplified IEEE test case)
system = {
    'generators': [
        {
            'id': 'G1',
            'bus': 1,
            'pmin_mw': 50,
            'pmax_mw': 200,
            'cost_curve': [0.05, 20, 1000],  # a*P² + b*P + c ($/h)
            'fuel': 'coal',
            'emissions_rate': 0.9  # tons CO2/MWh
        },
        {
            'id': 'G2',
            'bus': 2,
            'pmin_mw': 30,
            'pmax_mw': 150,
            'cost_curve': [0.07, 15, 500],
            'fuel': 'natural_gas',
            'emissions_rate': 0.4
        },
        {
            'id': 'G3',
            'bus': 3,
            'pmin_mw': 0,
            'pmax_mw': 100,
            'cost_curve': [0, 0, 0],  # Zero marginal cost
            'fuel': 'solar',
            'emissions_rate': 0,
            'variable': True,  # Renewable (intermittent)
            'forecast_mw': 75   # Current forecast
        },
        {
            'id': 'G4',
            'bus': 4,
            'pmin_mw': 0,
            'pmax_mw': 80,
            'cost_curve': [0, 0, 0],
            'fuel': 'wind',
            'emissions_rate': 0,
            'variable': True,
            'forecast_mw': 60
        }
    ],
    'loads': [
        {'bus': 1, 'demand_mw': 80},
        {'bus': 2, 'demand_mw': 120},
        {'bus': 3, 'demand_mw': 90},
        {'bus': 4, 'demand_mw': 70}
    ],
    'transmission_lines': [
        {'from_bus': 1, 'to_bus': 2, 'capacity_mw': 100, 'reactance': 0.05},
        {'from_bus': 2, 'to_bus': 3, 'capacity_mw': 80, 'reactance': 0.06},
        {'from_bus': 3, 'to_bus': 4, 'capacity_mw': 70, 'reactance': 0.04},
        {'from_bus': 1, 'to_bus': 4, 'capacity_mw': 90, 'reactance': 0.07}
    ]
}

# Build Pyomo optimization model
model = pyo.ConcreteModel()

# Sets
model.GENERATORS = pyo.Set(initialize=[g['id'] for g in system['generators']])
model.BUSES = pyo.Set(initialize=list(set([g['bus'] for g in system['generators']] +
                                          [l['bus'] for l in system['loads']])))

# Parameters
gen_dict = {g['id']: g for g in system['generators']}

# Decision variables: generator output
model.P = pyo.Var(model.GENERATORS, domain=pyo.NonNegativeReals)

# Objective: minimize total generation cost
def cost_rule(m):
    total_cost = 0
    for gen_id in m.GENERATORS:
        gen = gen_dict[gen_id]
        a, b, c = gen['cost_curve']
        total_cost += a * m.P[gen_id]**2 + b * m.P[gen_id] + c
    return total_cost

model.TotalCost = pyo.Objective(rule=cost_rule, sense=pyo.minimize)

# Constraints

# 1. Generator capacity limits
def gen_capacity_rule(m, gen_id):
    gen = gen_dict[gen_id]
    if gen.get('variable', False):
        # Renewable: cannot exceed forecast
        return m.P[gen_id] <= gen['forecast_mw']
    else:
        return (gen['pmin_mw'], m.P[gen_id], gen['pmax_mw'])

model.GenCapacity = pyo.Constraint(model.GENERATORS, rule=gen_capacity_rule)

# 2. Power balance: generation = demand + losses
total_demand = sum(load['demand_mw'] for load in system['loads'])
def power_balance_rule(m):
    return sum(m.P[gen_id] for gen_id in m.GENERATORS) == total_demand

model.PowerBalance = pyo.Constraint(rule=power_balance_rule)

# 3. Transmission line limits (simplified DC power flow)
# Actual power flow uses voltage angles - this is a simplified version
# For full AC power flow, use specialized tools like MATPOWER, PYPOWER, or Pandapower

# Solve
solver = pyo.SolverFactory('glpk')  # Free linear solver
results = solver.solve(model, tee=False)

# Extract solution
dispatch = {}
for gen_id in model.GENERATORS:
    dispatch[gen_id] = pyo.value(model.P[gen_id])

total_cost = pyo.value(model.TotalCost)

print("Optimal Power Flow Results:")
print(f"\nTotal Generation Cost: ${total_cost:.2f}/hour")
print("\nGenerator Dispatch:")
for gen_id, power_mw in dispatch.items():
    gen = gen_dict[gen_id]
    print(f"  {gen_id} ({gen['fuel']}): {power_mw:.1f} MW "
          f"[{gen['pmin_mw']}-{gen['pmax_mw']} MW]")

# Calculate renewable penetration
renewable_gen = sum(dispatch[g['id']] for g in system['generators'] if g.get('variable', False))
total_gen = sum(dispatch.values())
renewable_pct = (renewable_gen / total_gen) * 100

print(f"\nRenewable Penetration: {renewable_pct:.1f}%")

# Calculate emissions
total_emissions = sum(
    dispatch[g['id']] * g['emissions_rate']
    for g in system['generators']
)
print(f"Total Emissions: {total_emissions:.1f} tons CO2/hour")

# Multi-period optimization (24-hour unit commitment)
# Determines which generators to turn on/off each hour
uc_model = optimizer.unit_commitment(
    system=system,
    load_forecast=load_forecast_24h,
    renewable_forecast=renewable_forecast_24h,
    time_periods=24,
    constraints=[
        'min_up_time',     # Generator must stay on for minimum time
        'min_down_time',   # Must stay off for minimum time
        'ramp_rate',       # Cannot change output too quickly
        'startup_cost',    # Cost to start generator
        'spinning_reserve' # Reserve margin for unexpected outages
    ]
)

uc_solution = uc_model.solve()

print("\n24-Hour Unit Commitment:")
print(uc_solution.summary)

# Cost savings from optimal dispatch
baseline_cost = optimizer.calculate_baseline_cost(
    system,
    method='proportional'  # Simple proportional dispatch
)

savings = baseline_cost - total_cost
savings_pct = (savings / baseline_cost) * 100

print(f"\nCost Savings: ${savings:.2f}/hour ({savings_pct:.1f}%)")
print(f"Annual savings: ${savings * 24 * 365:,.0f}")
```

## Cost Estimate

### One-time Setup
- CloudFormation deployment: Free
- Lambda functions: Free tier
- Initial data ingestion: $50-100
- **Total: $50-100**

### Monthly Operating Costs

#### Small Microgrid (100 customers)
- IoT Core (smart meters, 15-min readings): $50
- Timestream (sensor data storage): $30
- S3 (historical data): $5
- Lambda (forecasting): $20
- SageMaker inference: $50
- **Total: ~$150-200/month**

#### Distribution Network (10,000 customers)
- IoT Core: $500
- Timestream: $300
- S3: $50
- Kinesis Data Streams: $200
- Lambda: $100
- SageMaker (training + inference): $400
- AWS Batch (optimization): $200
- **Total: ~$1,750-2,500/month**

#### Utility-Scale (100,000+ customers)
- IoT Core: $3,000
- Timestream: $2,000
- S3: $300
- Kinesis: $1,500
- Lambda: $500
- SageMaker: $2,500
- AWS Batch: $1,500
- Data transfer: $500
- **Total: ~$12,000-15,000/month**

#### Regional ISO/RTO Grid
- Full-scale grid optimization: $25,000-50,000/month
- Multiple SageMaker endpoints: $10,000/month
- High-frequency SCADA data: $15,000/month
- **Total: ~$50,000-100,000/month**

### Return on Investment

**Demand forecasting improvements:**
- 1% forecast error reduction → $100K-500K savings/year for mid-size utility
- Better unit commitment → 2-5% generation cost savings
- Avoided peak capacity charges

**Renewable integration:**
- 5-10% reduction in curtailment → $50K-200K savings/year
- Better wind/solar forecasting → More efficient grid operations

**Battery optimization:**
- Energy arbitrage: $50-200/MWh-year
- Frequency regulation: $100-300/MW-year
- Payback period: 5-10 years

**Peak demand reduction:**
- Demand response: $50-150/kW-year avoided capacity
- EV smart charging: 20-30% cost savings vs uncontrolled

**Outage prevention:**
- Predictive maintenance: 10-30% reduction in outage frequency
- Each outage: $100K-1M+ cost (lost revenue, restoration, penalties)
- ROI: 5-20x for predictive maintenance systems

## Performance Benchmarks

### Load Forecasting
- Day-ahead (24h): 2-5% MAPE
- Week-ahead: 5-10% MAPE
- Model training time: 2-4 hours (3 years hourly data)
- Inference latency: <500ms per forecast

### Renewable Forecasting
- Solar day-ahead: 10-20% RMSE (% of capacity)
- Wind day-ahead: 15-25% RMSE
- Intraday (1-6h): 5-15% RMSE
- Training time: 1-2 hours
- Inference: <200ms

### Optimization
- OPF solve time: 1-10 seconds (100-1000 buses)
- Unit commitment (24h, 50 generators): 1-5 minutes
- Battery optimization (24h): <1 second (LP)
- EV charging optimization (1000 vehicles): 30-60 seconds

### Anomaly Detection
- LSTM autoencoder training: 4-8 hours
- Inference latency: <100ms
- False positive rate: 1-5%
- Outage prediction lead time: 1-48 hours

### Data Processing
- Smart meter data ingestion: 100K messages/second (Kinesis)
- Timestream query (1 month, 1K meters): <2 seconds
- Historical analysis (5 years data): 10-30 minutes (Athena)

## Best Practices

### 1. Data Quality and Privacy

**Smart meter data:**
- Aggregate to neighborhood/feeder level to protect privacy
- Use differential privacy techniques for individual data
- Secure data transmission (TLS, IoT certificates)
- Comply with regulations (GDPR, CCPA, NERC CIP)

**Data validation:**
- Check for outliers, missing data
- Validate meter readings against substation totals
- Handle DST transitions, leap seconds properly
- Time zone consistency

### 2. Forecasting Best Practices

**Load forecasting:**
- Use ensemble models (LSTM + Prophet + SARIMAX)
- Incorporate weather forecasts and calendar effects
- Retrain models regularly (weekly/monthly)
- Monitor forecast accuracy, adjust if degrading
- Have fallback models (persistence, historical average)

**Renewable forecasting:**
- Combine NWP weather models with ML
- Update forecasts as weather predictions improve (intraday)
- Probabilistic forecasts for uncertainty quantification
- Validate against SCADA/revenue meter data

### 3. Optimization

**Solver selection:**
- Linear programs (LP): GLPK, Gurobi, CPLEX
- Mixed-integer (MILP): Gurobi, CPLEX (commercial, faster)
- Nonlinear (NLP): IPOPT, SNOPT
- For large-scale: Use decomposition methods

**Modeling tips:**
- Start simple, add complexity incrementally
- Validate against known solutions
- Check for infeasibility, diagnose constraints
- Use warm-start for similar problems
- Monitor solve times, simplify if too slow

### 4. Grid Reliability

**Reserve margins:**
- Maintain spinning reserve (3-7% of load)
- N-1 contingency planning (any single element failure)
- Frequency regulation capacity

**Renewable integration:**
- Ramp management (sudden solar/wind changes)
- Forecasting errors → require reserves
- Curtailment as last resort
- Energy storage for smoothing

### 5. Cybersecurity

**NERC CIP compliance:**
- Critical infrastructure protection standards
- Access controls, monitoring, incident response
- Network segmentation (IT vs OT)

**IoT security:**
- Device authentication (X.509 certificates)
- Encrypted communication (TLS 1.3)
- Regular security updates
- Intrusion detection

### 6. Ethical and Equity Considerations

**Energy affordability:**
- Time-of-use rates: Benefits those who can shift usage
- May disadvantage those without flexibility (elderly, disabled)
- Ensure baseline affordable rates

**Demand response equity:**
- Smart thermostats: Requires upfront investment
- Provide subsidies for low-income households
- Avoid disproportionate burden on vulnerable populations

**Data access:**
- Customers should access their own data
- Third-party access with consent (energy management services)
- Prevent discriminatory use (insurance, credit scoring)

## Troubleshooting

### Issue: Load Forecast Accuracy Degraded

**Symptoms:**
- MAPE increased from 3% to 8%
- Consistent over/under prediction

**Diagnosis:**
1. Check for data quality issues (missing meters, stuck values)
2. Verify weather data is current
3. Look for structural changes (new loads, closed businesses)
4. Check for seasonal effects (model trained on different season)

**Solutions:**
- Retrain model with recent data
- Add/remove features based on importance
- Adjust for known changes (new EV charging stations)
- Use online learning / incremental updates

### Issue: Optimization Problem Infeasible

**Symptoms:**
- Solver returns "infeasible"
- Cannot find solution satisfying all constraints

**Diagnosis:**
1. Check if demand exceeds total generation capacity
2. Verify transmission line limits aren't too restrictive
3. Check for conflicting constraints (min > max)
4. Test with relaxed constraints

**Solutions:**
- Add slack variables with high penalty
- Relax constraints temporarily to find near-feasible solution
- Check input data for errors
- Consider load shedding if truly impossible

### Issue: IoT Sensor Data Not Arriving

**Symptoms:**
- Missing readings in Timestream
- Gaps in time series

**Diagnosis:**
1. Check device connectivity (cellular, WiFi)
2. Verify IoT Core connection (device logs)
3. Check IoT rule is active and configured correctly
4. Verify IAM permissions for IoT → Timestream

**Solutions:**
- Restart device, check network
- Test with AWS IoT console test client
- Use CloudWatch Logs to debug IoT rules
- Fill gaps with interpolation (for analysis)

### Issue: Battery Optimization Shows No Profit

**Symptoms:**
- Optimization returns zero dispatch
- No charging/discharging scheduled

**Diagnosis:**
1. Check price spread (high - low prices)
2. Verify efficiency and degradation costs aren't too high
3. Check SOC and power constraints

**Solutions:**
- Prices may not be volatile enough for arbitrage
- Consider ancillary services (frequency regulation)
- Reduce degradation cost estimate
- Check if battery is undersized relative to price opportunities

### Issue: EV Charging Violates Transformer Limit

**Symptoms:**
- Total EV load exceeds transformer capacity
- Transformer overheating

**Diagnosis:**
1. Check if optimization includes transformer constraint
2. Verify baseload estimate is accurate
3. Too many EVs for available capacity

**Solutions:**
- Add transformer capacity constraint to optimization
- Stagger charging more aggressively
- Reduce max charge rate per vehicle
- Consider infrastructure upgrade (larger transformer)
- Implement dynamic load management (shed non-critical loads)

## Additional Resources

### Energy Data Sources

**U.S. Energy Information Administration (EIA):**
- https://www.eia.gov/opendata/
- Electricity generation, consumption, prices
- API access available

**National Renewable Energy Laboratory (NREL):**
- https://www.nrel.gov/
- PVWatts API (solar estimation)
- Wind Toolkit
- System Advisor Model (SAM)
- NSRDB (solar resource data)

**ISO/RTO Data:**
- CAISO: http://oasis.caiso.com/
- PJM: https://www.pjm.com/markets-and-operations
- ERCOT: http://www.ercot.com/gridinfo
- NYISO: https://www.nyiso.com/energy-market-operational-data

**Weather Data:**
- NOAA: https://www.ncei.noaa.gov/
- OpenWeatherMap: https://openweathermap.org/api
- Weather.gov API: https://www.weather.gov/documentation/services-web-api

### Software Tools

**Power System Analysis:**
- **MATPOWER:** Free MATLAB/Octave power flow tool
  - https://matpower.org/
- **PyPOWER:** Python port of MATPOWER
  - https://github.com/rwl/PYPOWER
- **Pandapower:** Python power system analysis
  - https://www.pandapower.org/
- **GridLAB-D:** Grid simulation
  - https://www.gridlaب-d.org/

**Optimization:**
- **Pyomo:** Python optimization modeling
  - http://www.pyomo.org/
- **CVXPY:** Convex optimization
  - https://www.cvxpy.org/
- **Gurobi:** Commercial solver (free academic license)
  - https://www.gurobi.com/
- **GLPK:** Free linear programming solver
  - https://www.gnu.org/software/glpk/

**Time Series Forecasting:**
- **Prophet:** Facebook's forecasting tool
  - https://facebook.github.io/prophet/
- **statsmodels:** ARIMA, SARIMAX
  - https://www.statsmodels.org/
- **GluonTS:** Probabilistic time series models
  - https://ts.gluon.ai/

**Solar/Wind:**
- **pvlib:** Solar PV modeling
  - https://pvlib-python.readthedocs.io/
- **windpowerlib:** Wind turbine power curves
  - https://windpowerlib.readthedocs.io/

### Key Research Papers

**Load Forecasting:**
1. Hong, T. & Fan, S. (2016). "Probabilistic electric load forecasting: A tutorial review." *International Journal of Forecasting*, 32(3), 914-938.
2. Kong, W. et al. (2019). "Short-term residential load forecasting based on LSTM recurrent neural network." *IEEE Transactions on Smart Grid*, 10(1), 841-851.

**Renewable Energy Forecasting:**
3. Wan, C. et al. (2015). "Probabilistic forecasting of wind power generation using extreme learning machine." *IEEE Transactions on Power Systems*, 29(3), 1033-1044.
4. Antonanzas, J. et al. (2016). "Review of photovoltaic power forecasting." *Solar Energy*, 136, 78-111.

**Battery Optimization:**
5. Xu, B. et al. (2018). "Optimal battery participation in frequency regulation markets." *IEEE Transactions on Power Systems*, 33(6), 6715-6725.
6. He, G. et al. (2016). "The economic end of life of electrochemical energy storage." *Applied Energy*, 273, 115151.

**EV Charging:**
7. Richardson, P. et al. (2012). "Electric vehicles and the electric grid." *Proceedings of the IEEE*, 99(6), 1116-1138.
8. Mwasilu, F. et al. (2014). "Electric vehicles and smart grid interaction: A review." *Renewable and Sustainable Energy Reviews*, 34, 501-516.

**Grid Optimization:**
9. Molzahn, D. K. et al. (2019). "A survey of distributed optimization and control algorithms for electric power systems." *IEEE Transactions on Smart Grid*, 8(6), 2941-2962.
10. Stoft, S. (2002). *Power System Economics: Designing Markets for Electricity*. Wiley-IEEE Press.

### Standards and Regulations

**NERC (North American Electric Reliability Corporation):**
- CIP (Critical Infrastructure Protection) standards
- https://www.nerc.com/

**IEEE Standards:**
- IEEE 1547: Interconnection of distributed energy resources
- IEEE 2030: Smart grid interoperability

**OpenADR (Automated Demand Response):**
- Protocol for demand response communication
- https://www.openadr.org/

### Online Courses and Tutorials

**Energy Systems:**
- Coursera: "Introduction to Power Electronics" (University of Colorado)
- edX: "Sustainable Energy" (TU Delft)
- NREL Learning: https://www.nrel.gov/research/re-learning.html

**AWS for Energy:**
- AWS Energy & Utilities: https://aws.amazon.com/energy/
- AWS IoT Core tutorials
- SageMaker time series forecasting examples

**Optimization:**
- "Convex Optimization" by Stephen Boyd (Stanford)
- Pyomo documentation and examples

## Next Steps

1. **Deploy Infrastructure**
   - Create CloudFormation stack
   - Set up IoT Core for smart meters
   - Initialize Timestream database

2. **Data Collection**
   - Ingest historical load data
   - Connect weather data sources
   - Set up SCADA/sensor data streams

3. **Build First Models**
   - Train load forecasting LSTM
   - Deploy solar forecasting model
   - Test battery optimization

4. **Real-Time Integration**
   - Deploy Lambda functions for inference
   - Set up anomaly detection
   - Create QuickSight dashboards

5. **Optimization Pipeline**
   - Implement OPF solver
   - Build EV charging coordination
   - Test demand response programs

6. **Validation and Tuning**
   - Backtest forecasts
   - Validate optimization results
   - Tune model hyperparameters

7. **Production Deployment**
   - Set up monitoring and alerts
   - Implement security controls
   - Train operators on new tools

---

**Tier 1 Project Status:** Production-ready

**Estimated Setup Time:** 6-10 hours

**Difficulty Level:** Advanced (requires power systems knowledge + ML + optimization)

**Cost:** $150-15,000/month (depending on scale)

**ROI:** 5-50x through cost savings, improved reliability, renewable integration

**Congratulations!** This is the **FINAL Tier 1 Flagship Project**, completing all **63 projects** in the Research Jumpstart matrix across **21 scientific domains**!

For questions, consult power systems textbooks, NREL resources, or AWS energy solutions documentation.
