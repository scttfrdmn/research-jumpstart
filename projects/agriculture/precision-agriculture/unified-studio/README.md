# Precision Agriculture at Scale - AWS Research Jumpstart

**Tier 1 Flagship Project**

Analyze agricultural data from satellites, drones, sensors, and farm management systems using remote sensing, machine learning, and IoT on AWS. Monitor crop health, predict yields, optimize irrigation, and enable data-driven farming decisions across millions of hectares.

## Overview

This flagship project demonstrates how to build precision agriculture systems at scale using AWS services. We'll work with satellite imagery (Sentinel-2, Landsat), drone data, soil sensors, weather stations, and farm management data to monitor crops, predict yields, detect diseases, and optimize agricultural practices.

### Key Features

- **Multi-scale monitoring:** Satellite (10-30m), drone (5cm), ground sensors (point)
- **Crop health:** NDVI, chlorophyll, water stress indices
- **Yield prediction:** ML models using satellite data, weather, soil
- **Disease detection:** Early identification from multispectral imagery
- **Precision inputs:** Variable rate application for fertilizers, pesticides, water
- **AWS services:** S3, IoT Core, SageMaker, Ground Station, Bedrock

### Scientific Applications

1. **Crop monitoring:** Track growth stages, detect stress, identify problems
2. **Yield forecasting:** Predict harvest volumes weeks/months in advance
3. **Disease detection:** Early warning for pests, pathogens, nutrient deficiency
4. **Irrigation optimization:** Water use efficiency, soil moisture monitoring
5. **Carbon sequestration:** Measure soil carbon, regenerative agriculture impact

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│               Precision Agriculture Pipeline                     │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ Sentinel-2   │      │ Landsat      │      │ Planet       │
│ (10m, 5d)    │─────▶│ (30m, 8d)    │─────▶│ (3m, daily)  │
└──────────────┘      └──────────────┘      └──────────────┘
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   S3 Data Lake    │
                    │  (Imagery, IoT    │
                    │   sensor data)    │
                    └─────────┬─────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼───────┐   ┌─────────▼─────────┐   ┌─────▼──────┐
│ IoT Core      │   │ SageMaker         │   │ AWS Batch  │
│ (Sensors)     │   │ (ML Models)       │   │ (Process)  │
└───────┬───────┘   └─────────┬─────────┘   └─────┬──────┘
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Timestream       │
                    │  (Time series DB) │
                    └─────────┬─────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼──────┐   ┌──────────▼────────┐   ┌───────▼───────┐
│ Crop Health  │   │ Yield Prediction  │   │ Disease       │
│ Monitoring   │   │ (Tons/hectare)    │   │ Detection     │
└───────┬──────┘   └──────────┬────────┘   └───────┬───────┘
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │ QuickSight/Bedrock│
                    │ Farm Dashboards & │
                    │ Recommendations   │
                    └───────────────────┘
```

## Major Data Sources

### 1. Satellite Imagery (Free)

**Sentinel-2 (ESA):**
- **Resolution:** 10m (RGB, NIR), 20m (red edge, SWIR)
- **Revisit:** 5 days (2 satellites)
- **Bands:** 13 bands (perfect for agriculture)
- **Access:** AWS Open Data, Copernicus Hub
- **S3:** `s3://sentinel-cogs/`
- **Applications:** NDVI, LAI, chlorophyll, water stress

**Landsat 8/9 (USGS):**
- **Resolution:** 30m multispectral
- **Revisit:** 8 days (combined)
- **History:** 40+ years of data
- **S3:** `s3://usgs-landsat/`

**MODIS Agriculture Products:**
- **Resolution:** 250m-1km
- **Products:** NDVI, EVI, LAI, FPAR, GPP
- **Frequency:** Daily, 8-day, 16-day composites

### 2. Commercial High-Resolution Imagery

**Planet Labs:**
- **Resolution:** 3-5m
- **Revisit:** Daily
- **Coverage:** Global
- **Cost:** Commercial license required

**Maxar/DigitalGlobe:**
- **Resolution:** 30cm-2m
- **Access:** Commercial

### 3. Drone/UAV Data

**Multispectral cameras:**
- **Sensors:** MicaSense RedEdge, Parrot Sequoia
- **Resolution:** 5-20cm
- **Bands:** RGB + NIR + red edge
- **Use:** Field-level monitoring, scouting

### 4. Ground Sensors (IoT)

**Soil sensors:**
- Soil moisture (volumetric water content)
- Temperature
- EC (electrical conductivity - salinity)
- pH, NPK (nitrogen, phosphorus, potassium)

**Weather stations:**
- Temperature, humidity
- Precipitation
- Wind speed/direction
- Solar radiation
- Evapotranspiration (ET)

**AWS IoT integration:**
- Publish sensor data to IoT Core
- Store in Timestream (time series database)
- Trigger Lambda for anomaly alerts

### 5. Farm Management Data

- Field boundaries (polygons)
- Crop type and variety
- Planting/harvest dates
- Fertilizer/pesticide applications
- Historical yields
- Soil maps

## Getting Started

### Prerequisites

```bash
# AWS CLI
aws --version

# Python dependencies
pip install -r requirements.txt
```

### Deploy Infrastructure

```bash
# Deploy CloudFormation stack
aws cloudformation create-stack \
  --stack-name precision-agriculture-stack \
  --template-body file://cloudformation/agriculture-stack.yml \
  --parameters file://cloudformation/parameters.json \
  --capabilities CAPABILITY_NAMED_IAM

# Wait for completion
aws cloudformation wait stack-create-complete \
  --stack-name precision-agriculture-stack
```

### Access Satellite Data

```python
from src.satellite_agriculture import CropMonitor
import datetime

# Initialize monitor
monitor = CropMonitor()

# Define field boundaries
field_boundary = {
    'type': 'Polygon',
    'coordinates': [[
        [-93.2, 42.0],  # Iowa cornfield example
        [-93.2, 42.05],
        [-93.15, 42.05],
        [-93.15, 42.0],
        [-93.2, 42.0]
    ]]
}

# Get NDVI time series for field
ndvi_series = monitor.get_ndvi_time_series(
    field_boundary=field_boundary,
    start_date='2023-04-01',
    end_date='2023-10-31',
    satellite='sentinel2',
    cloud_threshold=20  # Max cloud cover %
)

print(f"Retrieved {len(ndvi_series)} cloud-free observations")
monitor.plot_ndvi_timeseries(ndvi_series, save_path='field_ndvi.png')
```

## Core Analyses

### 1. Crop Health Monitoring

Calculate vegetation indices and detect stress.

```python
from src.crop_health import CropHealthAnalyzer
import rasterio
import numpy as np

analyzer = CropHealthAnalyzer()

# Load Sentinel-2 image
s2_image = analyzer.load_sentinel2(
    field_boundary=field_boundary,
    date='2023-07-15',
    bands=['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B11', 'B12']
)

# Calculate vegetation indices
indices = analyzer.calculate_vegetation_indices(
    s2_image,
    indices=[
        'NDVI',  # Normalized Difference Vegetation Index
        'EVI',   # Enhanced Vegetation Index
        'NDRE',  # Normalized Difference Red Edge (sensitive to chlorophyll)
        'NDMI',  # Normalized Difference Moisture Index (water stress)
        'SAVI'   # Soil Adjusted Vegetation Index
    ]
)

# NDVI = (NIR - Red) / (NIR + Red)
ndvi = indices['NDVI']
print(f"Mean NDVI: {np.mean(ndvi):.3f}")
print(f"NDVI range: [{np.min(ndvi):.3f}, {np.max(ndvi):.3f}]")

# Interpret NDVI values
# 0.0-0.2: Bare soil, water, snow
# 0.2-0.4: Sparse vegetation, early growth
# 0.4-0.6: Moderate vegetation
# 0.6-0.8: Dense healthy vegetation
# 0.8-1.0: Very dense vegetation

# Detect stress areas (low NDVI)
stress_map = analyzer.detect_stress_areas(
    ndvi,
    threshold_method='percentile',
    percentile=25  # Bottom 25% = stressed
)

# Chlorophyll content (using red edge)
chlorophyll = analyzer.estimate_chlorophyll(
    s2_image,
    method='red_edge_inflection'
)

# Water stress index
water_stress = analyzer.calculate_water_stress(
    s2_image,
    index='NDMI'  # or 'NDWI', 'MSI'
)

# Create management zones
zones = analyzer.create_management_zones(
    indices={'NDVI': ndvi, 'NDMI': water_stress, 'chlorophyll': chlorophyll},
    n_zones=3,  # Low, medium, high productivity
    method='kmeans'
)

# Visualize
analyzer.plot_field_map(
    zones,
    field_boundary=field_boundary,
    title='Management Zones',
    save_path='management_zones.png'
)

# Generate prescription map for variable rate application
prescription = analyzer.create_prescription_map(
    zones,
    input_type='nitrogen',  # or 'water', 'pesticide'
    rates={'low': 180, 'medium': 150, 'high': 120},  # kg N/ha
    output_format='shapefile'  # Compatible with farm equipment
)
```

### 2. Yield Prediction

Predict crop yields using ML models.

```python
from src.yield_prediction import YieldPredictor
import pandas as pd

predictor = YieldPredictor()

# Prepare training data
training_data = predictor.prepare_training_data(
    fields='s3://ag-data/field-boundaries/',
    historical_yields='s3://ag-data/historical-yields.csv',
    satellite_data='sentinel2',
    weather_data='s3://ag-data/weather/',
    soil_data='s3://ag-data/soil-maps/',
    years=range(2015, 2023)
)

# Feature engineering
features = predictor.extract_features(
    training_data,
    feature_types=[
        'vegetation_indices',  # NDVI, EVI peak values, integration
        'phenology',           # Green-up date, senescence, growing season length
        'weather',             # Temperature, precipitation, GDD (growing degree days)
        'soil',                # Texture, organic matter, drainage
        'management'           # Planting date, variety, fertilizer
    ]
)

# Train model
X = features.drop(columns=['field_id', 'year', 'yield'])
y = features['yield']  # Tons per hectare or bushels per acre

model = predictor.train_model(
    X, y,
    model_type='xgboost',  # or 'random_forest', 'neural_network'
    instance_type='ml.m5.4xlarge',
    cv_folds=10
)

# Evaluate
metrics = predictor.evaluate(model, X_test, y_test)
print(f"R² Score: {metrics['r2']:.3f}")
print(f"RMSE: {metrics['rmse']:.2f} tons/ha")
print(f"MAPE: {metrics['mape']:.1f}%")

# Predict current season (mid-season forecast)
current_data = predictor.prepare_prediction_data(
    fields=fields_to_predict,
    current_date='2024-07-15',
    satellite_data='sentinel2',
    weather_data=weather_forecast
)

predictions = predictor.predict(model, current_data)

# Confidence intervals
predictions_with_ci = predictor.predict_with_uncertainty(
    model,
    current_data,
    method='quantile',  # or 'bootstrap', 'gaussian_process'
    confidence=0.90
)

# Aggregate to farm/region level
farm_yield = predictor.aggregate_predictions(
    predictions,
    level='farm',
    return_statistics=True
)

print(f"Predicted farm yield: {farm_yield['mean']:.1f} ± {farm_yield['std']:.1f} tons/ha")

# Feature importance (what drives yields?)
importance = predictor.get_feature_importance(model, top_n=15)
predictor.plot_feature_importance(importance)
```

### 3. Disease and Pest Detection

Early detection of crop diseases using multispectral imagery.

```python
from src.disease_detection import DiseaseDetector
import sagemaker

detector = DiseaseDetector()

# Prepare training data (requires labeled examples)
training_data = detector.prepare_training_data(
    images='s3://ag-data/disease-images/',
    labels='s3://ag-data/disease-labels.csv',
    diseases=[
        'healthy',
        'northern_leaf_blight',
        'gray_leaf_spot',
        'common_rust',
        'nutrient_deficiency'
    ],
    augmentation=True  # Rotation, flip, color jitter
)

# Train CNN classifier
model = detector.train_cnn(
    training_data,
    architecture='resnet50',  # or 'efficientnet', 'mobilenet'
    input_size=224,
    epochs=50,
    batch_size=32,
    instance_type='ml.p3.2xlarge'
)

# Evaluate
metrics = detector.evaluate(model, test_data)
print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"F1 Score: {metrics['f1_weighted']:.3f}")

# Deploy model
endpoint = detector.deploy(
    model,
    instance_type='ml.g4dn.xlarge'
)

# Scan entire field for diseases
field_image = detector.load_field_image(
    field_boundary=field_boundary,
    date='2024-07-20',
    resolution='high'  # Use drone or Planet data
)

# Tile image for processing
disease_map = detector.scan_field(
    endpoint=endpoint,
    field_image=field_image,
    tile_size=224,
    stride=112,  # Overlap for better detection
    confidence_threshold=0.7
)

# Identify disease hotspots
hotspots = detector.identify_hotspots(
    disease_map,
    min_area=100,  # m²
    severity_threshold=0.5
)

# Generate scouting recommendations
recommendations = detector.generate_scouting_plan(
    hotspots,
    priority='severity'  # or 'area', 'spread_risk'
)

print(f"Detected {len(hotspots)} disease hotspots")
print("Top priority areas for scouting:")
for i, spot in enumerate(recommendations[:5], 1):
    print(f"{i}. {spot['disease']} at {spot['location']} "
          f"(severity: {spot['severity']:.2f}, area: {spot['area']:.0f} m²)")

# Early warning system
alert = detector.assess_disease_risk(
    current_conditions={
        'temperature': 25,  # °C
        'humidity': 85,  # %
        'rainfall_last_7d': 35,  # mm
        'crop_stage': 'tasseling',
        'previous_detections': hotspots
    },
    disease='northern_leaf_blight'
)

if alert['risk_level'] == 'high':
    print(f"⚠️  High risk for {alert['disease']}: {alert['recommendation']}")
```

### 4. Irrigation Optimization

Optimize water use with soil moisture monitoring and ET modeling.

```python
from src.irrigation import IrrigationOptimizer
import pandas as pd

optimizer = IrrigationOptimizer()

# Load soil moisture data from IoT sensors
soil_moisture = optimizer.load_soil_moisture(
    field_id='field_001',
    sensor_ids=['sm_001', 'sm_002', 'sm_003'],
    date_range=['2024-06-01', '2024-08-31']
)

# Calculate evapotranspiration (ET)
weather_data = optimizer.load_weather_data(
    station_id='GHCND:USC00132999',
    variables=['tmax', 'tmin', 'precip', 'wind', 'humidity', 'solar_radiation']
)

# Reference ET (Penman-Monteith equation)
et_reference = optimizer.calculate_et_reference(
    weather_data,
    method='penman_monteith'
)

# Crop coefficient (varies by growth stage)
crop_stage_dates = {
    'initial': '2024-05-01',
    'development': '2024-06-01',
    'mid_season': '2024-07-01',
    'late_season': '2024-08-15',
    'harvest': '2024-09-20'
}

kc = optimizer.get_crop_coefficient(
    crop='corn',
    dates=weather_data.index,
    stage_dates=crop_stage_dates
)

# Actual ET (crop water use)
et_crop = et_reference * kc

# Soil water balance
soil_properties = {
    'field_capacity': 0.35,  # Volumetric water content at field capacity
    'wilting_point': 0.15,
    'bulk_density': 1.3,  # g/cm³
    'root_depth': 100  # cm
}

water_balance = optimizer.calculate_water_balance(
    et_crop=et_crop,
    precipitation=weather_data['precip'],
    irrigation=irrigation_events,
    soil_properties=soil_properties,
    initial_moisture=0.30
)

# Determine irrigation needs
irrigation_schedule = optimizer.optimize_irrigation(
    water_balance=water_balance,
    soil_moisture_measured=soil_moisture,
    crop_stress_threshold=0.5,  # Fraction of available water
    irrigation_efficiency=0.85,
    forecast_days=7,
    weather_forecast=weather_forecast
)

print("Recommended irrigation schedule:")
for event in irrigation_schedule:
    print(f"  {event['date']}: {event['amount']:.1f} mm ({event['reason']})")

# Variable rate irrigation (if available)
vri_prescription = optimizer.create_vri_prescription(
    field_zones=zones,
    water_needs_by_zone={
        'low': 25,    # mm
        'medium': 30,
        'high': 35
    }
)

# Cost-benefit analysis
analysis = optimizer.analyze_irrigation_strategy(
    current_strategy=current_irrigation,
    optimized_strategy=irrigation_schedule,
    water_cost=0.05,  # $/m³
    yield_impact_model=model
)

print(f"Water savings: {analysis['water_saved']:.0f} m³ ({analysis['water_saved_pct']:.1f}%)")
print(f"Cost savings: ${analysis['cost_saved']:.2f}")
print(f"Yield impact: {analysis['yield_change']:+.1f} tons/ha")
```

### 5. Carbon Sequestration Measurement

Measure soil carbon and carbon credits potential.

```python
from src.carbon import CarbonAnalyzer

analyzer = CarbonAnalyzer()

# Estimate above-ground biomass from satellite
biomass = analyzer.estimate_biomass(
    field_boundary=field_boundary,
    satellite_data='sentinel2',
    crop_type='corn',
    date='2024-08-15'  # Peak biomass
)

# Soil organic carbon (requires soil sampling + modeling)
soil_samples = pd.read_csv('s3://ag-data/soil-samples.csv')

soc = analyzer.model_soil_organic_carbon(
    soil_samples=soil_samples,
    covariates={
        'satellite_indices': ['NDVI', 'NDRE'],
        'terrain': ['elevation', 'slope', 'aspect'],
        'climate': ['temperature', 'precipitation'],
        'management': ['tillage', 'cover_crops', 'years_no_till']
    },
    method='random_forest'
)

# Carbon sequestration rate
baseline_period = [2015, 2018]  # Before practice change
current_period = [2021, 2024]   # After adopting no-till + cover crops

sequestration_rate = analyzer.calculate_sequestration_rate(
    soc_data=soc,
    baseline_period=baseline_period,
    current_period=current_period,
    depth=30  # cm
)

print(f"Carbon sequestration rate: {sequestration_rate:.2f} tons CO2e/ha/year")

# Carbon credits potential
carbon_credits = analyzer.estimate_carbon_credits(
    sequestration_rate=sequestration_rate,
    field_area_ha=field_boundary_area,
    credit_price=20,  # $/ton CO2e
    project_duration=10  # years
)

print(f"Potential carbon credit revenue: ${carbon_credits['total_revenue']:,.0f} over {carbon_credits['years']} years")
print(f"Annual: ${carbon_credits['annual_revenue']:,.0f}/year")
```

### 6. Crop Type Classification

Classify crop types across large regions.

```python
from src.crop_classification import CropClassifier

classifier = CropClassifier()

# Prepare training data (requires ground truth from USDA Cropland Data Layer)
training_data = classifier.prepare_training_data(
    reference_data='s3://ag-data/cdl/',  # USDA CDL
    satellite='sentinel2',
    year=2023,
    crop_classes=[
        'corn',
        'soybeans',
        'wheat',
        'cotton',
        'alfalfa',
        'fallow',
        'pasture'
    ],
    samples_per_class=10000
)

# Extract time series features (phenology is key!)
features = classifier.extract_temporal_features(
    training_data,
    indices=['NDVI', 'EVI', 'NDWI'],
    temporal_stats=['mean', 'max', 'std', 'peak_date', 'integral']
)

# Train classifier
model = classifier.train(
    features,
    model_type='random_forest',  # Fast and accurate for this task
    instance_type='ml.m5.4xlarge'
)

# Evaluate
metrics = classifier.evaluate(model, test_data)
print(f"Overall accuracy: {metrics['accuracy']:.2%}")
print(f"Kappa coefficient: {metrics['kappa']:.3f}")

print("\nPer-class F1 scores:")
for crop, f1 in metrics['f1_per_class'].items():
    print(f"  {crop}: {f1:.3f}")

# Classify entire county/state
region_bbox = [-95.0, 41.0, -94.0, 42.0]  # Iowa county

crop_map = classifier.classify_region(
    model=model,
    bbox=region_bbox,
    year=2024,
    output_path='s3://ag-results/crop-map-2024.tif'
)

# Calculate crop acreage
acreage = classifier.calculate_acreage(
    crop_map,
    pixel_size=10  # meters
)

print("\nEstimated crop acreage:")
for crop, area_ha in acreage.items():
    area_acres = area_ha * 2.471  # Convert to acres
    print(f"  {crop}: {area_acres:,.0f} acres")
```

## IoT Sensor Integration

Connect soil sensors and weather stations to AWS IoT.

```python
from src.iot_sensors import SensorManager
import json

# Initialize IoT manager
iot = SensorManager(
    iot_endpoint='your-iot-endpoint.iot.us-east-1.amazonaws.com',
    region='us-east-1'
)

# Register new sensor
sensor_id = iot.register_sensor(
    sensor_type='soil_moisture',
    location={'lat': 42.05, 'lon': -93.17},
    field_id='field_001',
    metadata={
        'depth_cm': 30,
        'manufacturer': 'Acclima',
        'model': 'TDR-315'
    }
)

# Publish sensor reading (from device)
reading = {
    'timestamp': '2024-08-15T14:30:00Z',
    'sensor_id': sensor_id,
    'moisture_vwc': 0.28,  # Volumetric water content
    'temperature_c': 22.5,
    'ec_ds_m': 0.8  # Electrical conductivity (salinity)
}

iot.publish_reading(
    topic=f'agriculture/field_001/soil/{sensor_id}',
    payload=json.dumps(reading)
)

# Query sensor data from Timestream
soil_data = iot.query_timeseries(
    sensor_ids=[sensor_id],
    start_time='2024-08-01T00:00:00Z',
    end_time='2024-08-31T23:59:59Z',
    aggregation='5m'  # 5-minute averages
)

# Set up alerts (low soil moisture)
iot.create_alert_rule(
    name='low_soil_moisture_alert',
    condition='moisture_vwc < 0.20',
    action='sns',
    notification_target='arn:aws:sns:us-east-1:123456789:irrigation-alerts'
)
```

## Cost Estimate

**One-time setup:** $50-100

**Per farm costs (1,000 hectares = 2,500 acres):**
- Satellite data: Free (Sentinel-2, Landsat)
- Storage (100 GB imagery/year): $2.30/month
- Processing (50 hours Batch/month): $25-50
- IoT sensors (10 sensors, data ingestion): $5-10/month
- ML inference: $10-20/month
- **Total: ~$50-100/month per farm**

**Regional analysis (100,000 hectares):**
- Storage: $50-100/month
- Processing: $500-1,000/month
- ML training: $200-400/month
- **Total: $750-1,500/month**

**Return on Investment:**
- Increased yields: 3-10% typical
- Reduced input costs: 10-20% (fertilizer, water, pesticides)
- For 1,000 ha farm: $50,000-200,000/year value
- **ROI: 50-200x** in first year

## Performance Benchmarks

**Image processing:**
- Sentinel-2 scene (100 km²): 2-5 minutes
- NDVI for 1,000 hectares: <1 minute
- 10,000 fields processed: 2-4 hours with Batch

**Machine learning:**
- Yield prediction training (10K samples): 5-15 minutes
- Disease detection CNN training: 4-8 hours on GPU
- Inference: 100 fields/second

**IoT data:**
- Sensor message latency: <1 second
- Timestream query (1 month, 10 sensors): <2 seconds

## Best Practices

1. **Ground truth:** Always validate with field observations
2. **Timing:** Image timing critical for crop staging
3. **Cloud-free:** Use composite images or interpolation
4. **Calibration:** Validate models each season
5. **Integration:** Connect with farm management software
6. **Privacy:** Secure farmer data, clear data policies
7. **Actionable:** Provide specific recommendations, not just data

## References

### Resources

- **USDA Cropland Data Layer:** https://nassgeodata.gmu.edu/CropScape/
- **NASA ARSET:** https://appliedsciences.nasa.gov/what-we-do/capacity-building/arset
- **Sentinel Hub:** https://www.sentinel-hub.com/

### Software

- **Google Earth Engine:** https://earthengine.google.com/
- **QGIS:** https://qgis.org/
- **Rasterio:** https://rasterio.readthedocs.io/

### Key Papers

1. Lobell et al. (2015). "Scalable satellite-based crop yield mapper." *Remote Sensing of Environment*
2. Hunt & Daughtry (2018). "What good are unmanned aircraft systems for agricultural remote sensing?" *Remote Sensing*
3. Weiss et al. (2020). "Remote sensing for agricultural applications." *ISPRS J Photogramm Remote Sens*

## Next Steps

1. Deploy CloudFormation stack
2. Access Sentinel-2 data for test field
3. Calculate NDVI time series
4. Train yield prediction model
5. Deploy disease detection CNN
6. Connect IoT sensors

---

**Tier 1 Project Status:** Production-ready
**Estimated Setup Time:** 4-6 hours
**Cost:** $50-100/month per 1,000 ha farm
**ROI:** 50-200x in first year

For questions, consult precision agriculture resources or AWS documentation.
