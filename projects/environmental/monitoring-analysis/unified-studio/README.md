# Environmental Monitoring at Scale - AWS Research Jumpstart

**Tier 1 Flagship Project**

Analyze environmental data from satellites, sensors, and monitoring networks using remote sensing, time series analysis, and machine learning on AWS. Process petabytes of Earth observation data, track pollution levels, monitor deforestation, and predict environmental changes at global scale.

## Overview

This flagship project demonstrates how to analyze massive environmental datasets using AWS services. We'll work with satellite imagery (Landsat, Sentinel, MODIS), air quality sensors, climate stations, and ocean buoys to monitor environmental changes, detect anomalies, and predict future conditions across the globe.

### Key Features

- **Satellite data:** Landsat (40+ years), Sentinel-2 (10m resolution), MODIS (daily global coverage)
- **Sensor networks:** Air quality (EPA, PurpleAir), weather stations, ocean buoys
- **Petabyte scale:** Process years of Earth observation data
- **Machine learning:** Land cover classification, deforestation detection, pollution prediction
- **Time series:** Trend analysis, anomaly detection, forecasting
- **AWS services:** S3, Batch, SageMaker, Ground Station, OpenData Registry

### Scientific Applications

1. **Deforestation monitoring:** Track forest loss with Landsat/Sentinel
2. **Air quality analysis:** Monitor PM2.5, ozone, NO2 from sensors and satellites
3. **Land cover classification:** Map agriculture, urban areas, water bodies
4. **Climate monitoring:** Temperature trends, precipitation patterns, extreme events
5. **Ocean health:** Sea surface temperature, chlorophyll, plastic pollution

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│             Environmental Monitoring Pipeline                    │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ Landsat      │      │ Sentinel-2   │      │ MODIS        │
│ (30m, 16d)   │─────▶│ (10m, 5d)    │─────▶│ (250m, 1d)   │
└──────────────┘      └──────────────┘      └──────────────┘
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │ S3 Data Lake      │
                    │ (Satellite        │
                    │  imagery, COGs)   │
                    └─────────┬─────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼───────┐   ┌─────────▼─────────┐   ┌─────▼──────┐
│ AWS Batch     │   │ SageMaker         │   │ Ground     │
│ (Processing)  │   │ (ML Models)       │   │ Station    │
└───────┬───────┘   └─────────┬─────────┘   └─────┬──────┘
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Glue Catalog     │
                    │  (Metadata)       │
                    └─────────┬─────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼──────┐   ┌──────────▼────────┐   ┌───────▼───────┐
│ Deforestation│   │ Air Quality       │   │ Land Cover    │
│ Detection    │   │ Prediction        │   │ Classification│
└───────┬──────┘   └──────────┬────────┘   └───────┬───────┘
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │ Bedrock (Claude)  │
                    │ Environmental     │
                    │ Insights          │
                    └───────────────────┘
```

## Major Data Sources

### 1. AWS Open Data Registry - Satellite Imagery

**Landsat (USGS):**
- **Coverage:** Global, 1972-present (50+ years!)
- **Resolution:** 30m multispectral, 15m panchromatic
- **Revisit:** 16 days (Landsat 8/9 combined: 8 days)
- **Bands:** 11 bands (visible, NIR, SWIR, thermal)
- **Format:** Cloud Optimized GeoTIFF (COG)
- **S3 Bucket:** `s3://usgs-landsat/`
- **Size:** 1+ PB
- **Cost:** Free (requester pays S3 egress)

**Sentinel-2 (ESA):**
- **Coverage:** Global land, 2015-present
- **Resolution:** 10m (RGB, NIR), 20m (red edge, SWIR), 60m (coastal, cirrus)
- **Revisit:** 5 days (Sentinel-2A/B combined)
- **Bands:** 13 bands
- **S3 Bucket:** `s3://sentinel-cogs/`
- **Applications:** Agriculture, forestry, urban monitoring

**MODIS (NASA):**
- **Coverage:** Global, daily
- **Resolution:** 250m (RGB), 500m (NIR, SWIR), 1km (thermal)
- **Revisit:** 1-2 days
- **Products:** Surface reflectance, vegetation indices (NDVI, EVI), LST
- **Access:** NASA Earth data, AWS Open Data
- **Applications:** Climate monitoring, wildfire detection

### 2. Air Quality Sensors

**EPA AQS (Air Quality System):**
- **Coverage:** USA, 4,000+ monitoring sites
- **Pollutants:** PM2.5, PM10, O3, NO2, SO2, CO
- **Frequency:** Hourly measurements
- **Access:** EPA API, S3 (if ingested)
- **History:** 1980s-present

**PurpleAir:**
- **Coverage:** Global, 20,000+ sensors (crowdsourced)
- **Measurement:** PM2.5 (low-cost sensors)
- **Frequency:** Real-time (every few minutes)
- **Access:** PurpleAir API
- **Use case:** High spatial resolution air quality mapping

**Sentinel-5P:**
- **Satellite:** Atmospheric composition monitoring
- **Products:** NO2, SO2, CO, O3, aerosols
- **Resolution:** 7×3.5 km
- **Coverage:** Daily global

### 3. Climate Data

**NOAA Climate Data:**
- **GHCN (Global Historical Climatology Network):** 100,000+ stations
- **Variables:** Temperature, precipitation, snow
- **Access:** NOAA API, AWS Open Data
- **History:** 1750s-present (varies by station)

**ERA5 Reanalysis:**
- **Provider:** ECMWF
- **Resolution:** 0.25° (~30 km)
- **Variables:** 100+ atmospheric variables
- **Frequency:** Hourly, 1950-present
- **Access:** Climate Data Store, AWS Open Data

### 4. Ocean Data

**NOAA Buoys:**
- **Coverage:** Global oceans, 1,000+ buoys
- **Measurements:** Wave height, water temp, wind, pressure
- **Frequency:** Hourly
- **Access:** NOAA NDBC

**Copernicus Marine:**
- **Products:** SST, chlorophyll, sea level, currents
- **Resolution:** 1-10 km
- **Sources:** Satellite + in-situ

## Getting Started

### Prerequisites

```bash
# AWS CLI
aws --version

# GDAL for geospatial processing
conda install gdal

# Python dependencies
pip install -r requirements.txt
```

### Deploy Infrastructure

```bash
# Deploy CloudFormation stack
aws cloudformation create-stack \
  --stack-name environmental-monitoring-stack \
  --template-body file://cloudformation/environmental-stack.yml \
  --parameters file://cloudformation/parameters.json \
  --capabilities CAPABILITY_NAMED_IAM

# Wait for completion
aws cloudformation wait stack-create-complete \
  --stack-name environmental-monitoring-stack
```

### Access Satellite Data

```python
from src.satellite_data import LandsatLoader, SentinelLoader
import rasterio

# Initialize Landsat loader
landsat = LandsatLoader()

# Search for scenes
scenes = landsat.search_scenes(
    bbox=[-122.5, 37.7, -122.3, 37.9],  # San Francisco
    date_range=['2023-01-01', '2023-12-31'],
    max_cloud_cover=20,  # Percent
    tier='T1'  # Tier 1 (highest quality)
)

print(f"Found {len(scenes)} Landsat scenes")

# Download specific scene
scene = scenes[0]
bands = landsat.download_scene(
    scene_id=scene['scene_id'],
    bands=['B4', 'B3', 'B2'],  # RGB
    output_dir='./data/landsat/'
)

# Load into numpy array
with rasterio.open(bands['B4']) as src:
    red = src.read(1)

# Or load directly from S3
cog_url = f"s3://usgs-landsat/collection02/level-2/{scene['path']}"
```

## Core Analyses

### 1. Deforestation Detection

Detect and quantify forest loss using time series of satellite imagery.

```python
from src.deforestation import DeforestationDetector
import numpy as np

detector = DeforestationDetector()

# Load time series of NDVI (Normalized Difference Vegetation Index)
# NDVI = (NIR - Red) / (NIR + Red), ranges from -1 to 1
# Healthy vegetation: 0.6-0.9, bare soil: 0.1-0.2

region = {
    'bbox': [-60.5, -3.5, -60.0, -3.0],  # Amazon rainforest
    'start_date': '2020-01-01',
    'end_date': '2023-12-31'
}

# Get time series
time_series = detector.get_ndvi_time_series(
    region=region,
    satellite='sentinel2',
    frequency='monthly'
)

# Detect forest loss (NDVI drops)
forest_loss = detector.detect_forest_loss(
    ndvi_time_series=time_series,
    threshold=0.3,  # NDVI drop threshold
    persistence=3    # Must persist for 3 months
)

# Calculate deforestation area
area_km2 = detector.calculate_area(forest_loss, resolution=10)
print(f"Forest loss detected: {area_km2:.1f} km²")

# Change detection using multiple dates
before_image = detector.load_image(date='2020-01-15', bbox=region['bbox'])
after_image = detector.load_image(date='2023-12-15', bbox=region['bbox'])

change_map = detector.detect_changes(
    before_image,
    after_image,
    method='cvaps'  # or 'mad', 'pca'
)

# Classify changes
classified = detector.classify_changes(
    change_map,
    classes=['no_change', 'deforestation', 'regrowth', 'burn_scar']
)

# Visualize
detector.plot_change_map(
    classified,
    bbox=region['bbox'],
    save_path='deforestation_map.png'
)

# Alert system for rapid deforestation
alerts = detector.create_alerts(
    forest_loss,
    severity_threshold=100,  # km²
    notification_endpoint='your-sns-topic'
)
```

### 2. Land Cover Classification

Classify satellite imagery into land cover types using machine learning.

```python
from src.land_cover import LandCoverClassifier
import sagemaker

# Initialize classifier
classifier = LandCoverClassifier()

# Prepare training data (using existing land cover products)
training_data = classifier.prepare_training_data(
    reference_product='ESA WorldCover',  # or 'NLCD', 'Copernicus'
    satellite='sentinel2',
    n_samples=100000,
    classes=[
        'water',
        'trees',
        'grassland',
        'cropland',
        'built_area',
        'bare_soil',
        'snow_ice'
    ]
)

# Train Random Forest classifier
model = classifier.train_random_forest(
    training_data,
    n_estimators=100,
    max_depth=30,
    instance_type='ml.m5.4xlarge'
)

# Or train deep learning model (U-Net)
dl_model = classifier.train_unet(
    training_data,
    architecture='unet_resnet50',
    epochs=50,
    batch_size=16,
    instance_type='ml.p3.8xlarge'
)

# Classify entire region
region_bbox = [-100.0, 30.0, -95.0, 35.0]  # Texas

land_cover_map = classifier.classify_region(
    model=dl_model,
    bbox=region_bbox,
    satellite='sentinel2',
    date='2023-06-15',
    tile_size=256,
    batch_processing=True
)

# Calculate land cover statistics
stats = classifier.calculate_statistics(land_cover_map)
print(stats)

# Visualize
classifier.plot_land_cover(
    land_cover_map,
    legend=True,
    save_path='land_cover_texas.png'
)

# Time series land cover change
years = range(2015, 2024)
land_cover_series = []

for year in years:
    lc_map = classifier.classify_region(
        model=dl_model,
        bbox=region_bbox,
        date=f'{year}-06-15'
    )
    stats = classifier.calculate_statistics(lc_map)
    land_cover_series.append(stats)

# Plot urbanization trend
classifier.plot_urbanization_trend(
    land_cover_series,
    years=years,
    save_path='urbanization_trend.png'
)
```

### 3. Air Quality Monitoring and Prediction

Analyze air quality data and predict pollution levels.

```python
from src.air_quality import AirQualityAnalyzer
import pandas as pd

# Initialize analyzer
aq = AirQualityAnalyzer()

# Load EPA data
epa_data = aq.load_epa_data(
    pollutant='PM25',
    bbox=[-118.5, 33.7, -117.5, 34.2],  # Los Angeles
    date_range=['2020-01-01', '2023-12-31']
)

# Time series analysis
daily_average = aq.compute_daily_average(epa_data)

# Detect exceedances (unhealthy levels)
exceedances = aq.detect_exceedances(
    daily_average,
    standard='EPA',  # EPA PM2.5 standard: 35 μg/m³ (24-hour)
    pollutant='PM25'
)

print(f"Days exceeding standard: {len(exceedances)}")

# Temporal patterns
patterns = aq.analyze_temporal_patterns(
    daily_average,
    patterns=['hourly', 'day_of_week', 'seasonal']
)

# Spatial interpolation
interpolated = aq.spatial_interpolation(
    epa_data,
    method='kriging',  # or 'idw', 'spline'
    grid_resolution=1000  # meters
)

# Integrate satellite data (Sentinel-5P NO2)
sentinel5p = aq.load_sentinel5p(
    product='NO2',
    bbox=[-118.5, 33.7, -117.5, 34.2],
    date_range=['2023-01-01', '2023-12-31']
)

# Correlation with ground measurements
correlation = aq.correlate_satellite_ground(
    satellite_data=sentinel5p,
    ground_data=epa_data,
    pollutant='NO2'
)

# ML prediction model
from src.air_quality_ml import AQPredictor

predictor = AQPredictor()

# Prepare features
features = predictor.prepare_features(
    air_quality_data=epa_data,
    weather_data=weather_df,  # Temperature, wind, humidity
    traffic_data=traffic_df,  # Traffic volume
    calendar_features=True,   # Day of week, holidays
    lagged_values=[1, 7, 365] # 1 day, 1 week, 1 year lags
)

# Train model
X = features.drop(columns=['pm25_next_day'])
y = features['pm25_next_day']

model = predictor.train_xgboost(
    X, y,
    instance_type='ml.m5.4xlarge',
    cv_folds=10
)

# Evaluate
metrics = predictor.evaluate(model, X_test, y_test)
print(f"RMSE: {metrics['rmse']:.2f} μg/m³")
print(f"R²: {metrics['r2']:.3f}")

# Forecast next 7 days
forecast = predictor.forecast(
    model,
    current_conditions=current_data,
    weather_forecast=weather_forecast_df,
    n_days=7
)

# Alert system
alerts = predictor.generate_alerts(
    forecast,
    thresholds={
        'moderate': 35,    # μg/m³
        'unhealthy': 55,
        'very_unhealthy': 150
    }
)
```

### 4. Vegetation Monitoring (NDVI Time Series)

Monitor vegetation health and phenology using NDVI.

```python
from src.vegetation import VegetationAnalyzer

analyzer = VegetationAnalyzer()

# Load MODIS NDVI (250m, 16-day composite)
ndvi_ts = analyzer.load_modis_ndvi(
    bbox=[-100.0, 40.0, -99.0, 41.0],  # Agricultural region
    date_range=['2015-01-01', '2023-12-31']
)

# Smooth time series (remove noise, clouds)
ndvi_smooth = analyzer.smooth_time_series(
    ndvi_ts,
    method='savitzky_golay',  # or 'whittaker', 'harmonic'
    window_size=5
)

# Extract phenology metrics
phenology = analyzer.extract_phenology(
    ndvi_smooth,
    metrics=[
        'start_of_season',
        'end_of_season',
        'peak_ndvi',
        'length_of_season',
        'integrated_ndvi'  # Proxy for productivity
    ]
)

# Trend analysis (is vegetation getting greener/browner?)
trend = analyzer.mann_kendall_trend(
    ndvi_ts,
    alpha=0.05
)

if trend['significant']:
    print(f"Significant trend detected: {trend['direction']} (p={trend['p_value']:.4f})")
    print(f"Sen's slope: {trend['slope']:.4f} NDVI units/year")

# Anomaly detection (drought, disturbance)
anomalies = analyzer.detect_ndvi_anomalies(
    ndvi_smooth,
    baseline_period=['2015-01-01', '2020-12-31'],
    threshold=2.0  # Standard deviations
)

# Crop yield prediction from NDVI
crop_yield = analyzer.predict_crop_yield(
    ndvi_ts,
    crop_type='corn',
    historical_yields=yield_data,
    model='ridge_regression'
)
```

### 5. Climate Extremes Analysis

Detect and analyze extreme weather events.

```python
from src.climate_extremes import ExtremesAnalyzer

analyzer = ExtremesAnalyzer()

# Load climate data
climate_data = analyzer.load_climate_data(
    variables=['tmax', 'tmin', 'precip'],
    stations=station_list,
    date_range=['1980-01-01', '2023-12-31']
)

# Heat wave detection
heat_waves = analyzer.detect_heat_waves(
    temperature_data=climate_data['tmax'],
    definition='ehf',  # Excess Heat Factor, or 'ctfhd', 'hwmid'
    baseline_period=[1980, 2010]
)

print(f"Detected {len(heat_waves)} heat wave events")

# Extreme precipitation
extreme_precip = analyzer.detect_extreme_precipitation(
    precip_data=climate_data['precip'],
    threshold='99th_percentile',  # or absolute value
    duration='1day'  # or '3day', '5day'
)

# Drought indices
drought = analyzer.calculate_drought_indices(
    precip_data=climate_data['precip'],
    temp_data=climate_data['tmax'],
    indices=['spi', 'spei', 'pdsi']  # SPI, SPEI, Palmer Drought Severity Index
)

# Trend in extremes
extremes_trend = analyzer.trend_analysis(
    heat_waves,
    metrics=['frequency', 'duration', 'intensity'],
    method='theil_sen'
)

# Visualize
analyzer.plot_extremes_timeline(
    heat_waves,
    extreme_precip,
    drought,
    save_path='climate_extremes.png'
)

# Return period analysis (100-year event?)
return_periods = analyzer.extreme_value_analysis(
    climate_data['precip'],
    distribution='gev',  # Generalized Extreme Value
    return_periods=[2, 5, 10, 25, 50, 100]
)

print(f"100-year precipitation event: {return_periods[100]:.1f} mm")
```

### 6. Ocean Monitoring

Analyze sea surface temperature and chlorophyll.

```python
from src.ocean import OceanAnalyzer

ocean = OceanAnalyzer()

# Load SST data (Sea Surface Temperature)
sst = ocean.load_sst(
    bbox=[-180, -90, 180, 90],  # Global
    date_range=['2020-01-01', '2023-12-31'],
    source='GHRSST'  # or 'MODIS', 'VIIRS'
)

# Detect marine heat waves
mhw = ocean.detect_marine_heat_waves(
    sst_data=sst,
    baseline_period=[1990, 2020],
    threshold_percentile=90
)

# Coral bleaching risk
bleaching_risk = ocean.calculate_bleaching_risk(
    sst_data=sst,
    threshold_method='degree_heating_weeks'
)

# Chlorophyll (ocean productivity)
chlor = ocean.load_chlorophyll(
    bbox=[-80, 35, -60, 45],  # Gulf Stream region
    date_range=['2023-01-01', '2023-12-31'],
    source='MODIS_Aqua'
)

# Detect algal blooms
blooms = ocean.detect_algal_blooms(
    chlorophyll_data=chlor,
    threshold=10.0,  # mg/m³
    min_area=100     # km²
)

# El Niño detection
nino_indices = ocean.calculate_enso_indices(
    sst_data=sst,
    indices=['nino34', 'oni', 'soi']
)

ocean.plot_enso_timeline(nino_indices, save_path='enso_timeline.png')
```

## Large-Scale Processing with AWS Batch

Process years of satellite imagery in parallel.

```python
from src.batch_processing import SatelliteProcessor

processor = SatelliteProcessor(
    job_queue='environmental-processing-queue',
    job_definition='landsat-ndvi-calculation'
)

# Submit batch jobs for NDVI calculation
region_tiles = processor.get_tiles(
    bbox=[-180, -60, 180, 80],  # Global (excluding poles)
    tile_size=1.0  # 1 degree tiles
)

job_ids = []
for tile in region_tiles:
    job_id = processor.submit_ndvi_job(
        bbox=tile,
        date_range=['2020-01-01', '2023-12-31'],
        satellite='landsat8',
        output_bucket='s3://environmental-results/ndvi/'
    )
    job_ids.append(job_id)

print(f"Submitted {len(job_ids)} jobs")

# Monitor progress
processor.monitor_jobs(job_ids, update_interval=300)

# Mosaic results
ndvi_mosaic = processor.mosaic_tiles(
    tile_paths=[f's3://environmental-results/ndvi/tile_{i}.tif' for i in range(len(region_tiles))],
    output_path='s3://environmental-results/global_ndvi_2020_2023.tif'
)
```

## Machine Learning Applications

### Deforestation Detection with Deep Learning

```python
from src.deep_learning import DeforestationCNN

# Prepare training data (pre/post image pairs with labels)
train_data = DeforestationCNN.prepare_training_data(
    images_before='s3://training/before/',
    images_after='s3://training/after/',
    labels='s3://training/labels/',  # Binary masks
    augmentation=True
)

# Train U-Net model
model = DeforestationCNN.train_unet(
    train_data,
    architecture='unet_efficientnet_b4',
    input_shape=(256, 256, 6),  # 6 bands (RGB + NIR from 2 dates)
    epochs=50,
    batch_size=32,
    instance_type='ml.p3.8xlarge'
)

# Evaluate
metrics = model.evaluate(test_data)
print(f"IoU Score: {metrics['iou']:.3f}")
print(f"F1 Score: {metrics['f1']:.3f}")

# Deploy for inference
endpoint = model.deploy(
    instance_type='ml.g4dn.xlarge',
    initial_instance_count=1
)

# Predict deforestation in new region
predictions = model.predict(
    endpoint=endpoint,
    image_pairs='s3://new-region/image-pairs/',
    batch_size=100
)
```

## Cost Estimate

**One-time setup:** $50-100

**Monthly costs (global monitoring):**
- Data storage (100 TB processed imagery): $2,300/month
- Batch processing (1,000 hours/month): $500-1,000
- SageMaker training: $200-500/month
- Data egress: $200-500/month
- **Total: $3,200-4,300/month**

**Research project (regional study, 1 year):**
- Data download and storage: $200-500
- Processing (AWS Batch): $500-1,000
- ML model training: $300-600
- Analysis: $200-400
- **Total: $1,200-2,500**

## Performance Benchmarks

**Satellite data processing:**
- Landsat scene (7 bands, 30m): 2-5 minutes per scene
- Sentinel-2 scene (13 bands, 10/20m): 5-10 minutes
- NDVI calculation: 100 scenes/hour with 10 instances

**Machine learning:**
- Land cover classification (Random Forest, 100K samples): 10-20 minutes
- U-Net training (50K image pairs): 8-12 hours on ml.p3.8xlarge
- Inference: 100 images/second on GPU

**Time series analysis:**
- NDVI trend analysis (10 years, 1000 pixels): 1 second
- 1 million pixels: ~15 minutes

## Best Practices

1. **Cloud-Optimized formats:** Use COGs for efficient partial reads
2. **Tiling:** Process large regions in tiles to manage memory
3. **Temporal compositing:** Create cloud-free mosaics
4. **Validation:** Always validate with ground truth data
5. **Atmospheric correction:** Use Level-2 products when available
6. **Cost control:** Use S3 Intelligent-Tiering, Spot instances
7. **Reproducibility:** Version data, code, and model weights

## References

### Resources

- **AWS Open Data Registry:** https://registry.opendata.aws/
- **Google Earth Engine:** https://earthengine.google.com/
- **Copernicus Data Space:** https://dataspace.copernicus.eu/

### Software

- **GDAL:** https://gdal.org/
- **Rasterio:** https://rasterio.readthedocs.io/
- **Google Earth Engine Python API:** https://developers.google.com/earth-engine/guides/python_install
- **Sentinel Hub:** https://www.sentinel-hub.com/

### Key Papers

1. Gorelick et al. (2017). "Google Earth Engine: Planetary-scale geospatial analysis." *Remote Sensing of Environment*
2. Hansen et al. (2013). "High-Resolution Global Maps of 21st-Century Forest Cover Change." *Science*
3. Zhu et al. (2019). "Deep Learning in Remote Sensing." *IEEE GRSS*
4. Drusch et al. (2012). "Sentinel-2: ESA's Optical High-Resolution Mission." *Remote Sensing of Environment*

## Next Steps

1. Deploy CloudFormation stack
2. Access Landsat data from S3
3. Calculate NDVI time series
4. Train land cover classifier
5. Build deforestation monitoring system

---

**Tier 1 Project Status:** Production-ready
**Estimated Setup Time:** 4-6 hours
**Processing:** Petabyte-scale satellite data
**Cost:** $1,200-2,500 for regional study, $3,200-4,300/month for global monitoring

For questions, consult remote sensing textbooks or AWS documentation.
