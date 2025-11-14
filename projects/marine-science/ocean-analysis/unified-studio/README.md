# Ocean and Marine Ecosystem Analysis at Scale - AWS Research Jumpstart

**Tier 1 Flagship Project**

Large-scale ocean data analysis using satellite observations, in-situ measurements, and machine learning on AWS. Process terabytes of oceanographic data from satellites (MODIS, VIIRS, Sentinel-3), Argo floats, buoys, and gliders to monitor sea surface temperature, ocean currents, marine species distribution, coral reef health, and climate change impacts across global oceans.

## Overview

This flagship project demonstrates how to build production-grade ocean analysis systems on AWS. We'll integrate data from NASA Ocean Color, NOAA CoralTemp, Copernicus Marine, Argo floats, and biodiversity databases (OBIS, GBIF), apply machine learning for species distribution modeling and coral bleaching detection, perform Lagrangian particle tracking for ocean currents, and analyze climate impacts including marine heat waves and ocean acidification.

### Key Features

- **Satellite ocean observations:** Sea surface temperature (SST), ocean color (chlorophyll-a), sea level anomaly, sea ice
- **In-situ data integration:** 4,000+ Argo floats, drifting buoys, CTD casts, moored sensors
- **Species tracking:** Acoustic telemetry, satellite tags, eDNA metabarcoding analysis
- **Habitat mapping:** Coral reefs, seagrass meadows, kelp forests with Landsat/Sentinel-2
- **Ocean current analysis:** OSCAR, HYCOM models, Lagrangian particle tracking
- **Climate impacts:** Marine heat waves, ocean acidification, sea level rise trends
- **AWS services:** S3, Batch, SageMaker, Lambda, Athena, Timestream, QuickSight

### Scientific Applications

1. **Marine heat wave detection:** Track SST anomalies and thermal stress on ecosystems
2. **Coral bleaching monitoring:** Satellite-based early warning systems for reef managers
3. **Species distribution modeling:** Predict habitat suitability under climate change scenarios
4. **Ocean circulation analysis:** Visualize currents and track particle trajectories
5. **Fisheries management:** Stock assessment and marine protected area (MPA) effectiveness

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              Ocean Analysis Architecture                         │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ MODIS-Aqua   │    │ VIIRS        │    │ Sentinel-3   │
│ (SST, Chl)   │───▶│ (SST)        │───▶│ (Ocean Color)│
└──────────────┘    └──────────────┘    └──────────────┘
        │                   │                   │
        └───────────────────┴───────────────────┘
                            │
┌──────────────┐    ┌───────▼───────┐    ┌──────────────┐
│ Argo Floats  │───▶│ S3 Data Lake  │◀───│ Copernicus   │
│ (T/S prof.)  │    │ (netCDF, Zarr)│    │ Marine       │
└──────────────┘    └───────┬───────┘    └──────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼───────┐   ┌───────▼───────┐   ┌─────▼──────┐
│ AWS Batch     │   │ Lambda        │   │ SageMaker  │
│ (netCDF proc.)│   │ (ERDDAP sync) │   │ (ML models)│
└───────┬───────┘   └───────┬───────┘   └─────┬──────┘
        │                   │                   │
        └───────────────────┴───────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼───────┐   ┌───────▼───────┐   ┌─────▼──────┐
│ Athena        │   │ Timestream    │   │ RDS        │
│ (Species DB)  │   │ (Buoy data)   │   │ (Metadata) │
└───────┬───────┘   └───────┬───────┘   └─────┬──────┘
        │                   │                   │
        └───────────────────┴───────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼───────┐   ┌───────▼───────┐   ┌─────▼──────┐
│ QuickSight    │   │ CloudWatch    │   │ SNS Alerts │
│ (Dashboards)  │   │ (Monitoring)  │   │ (Bleaching)│
└───────────────┘   └───────────────┘   └────────────┘
```

## Data Sources

### 1. NASA Ocean Color (OceanData)

**What:** Satellite-derived ocean color, SST, and biogeochemical properties
**Coverage:** Global, daily to 8-day composites, 1997-present
**Satellites:** MODIS-Aqua (2002-present), VIIRS (2012-present), SeaWiFS (1997-2010)
**Access:** NASA Earthdata, ERDDAP servers, OPeNDAP
**URL:** https://oceancolor.gsfc.nasa.gov/

**Key Products:**
- **Chlorophyll-a concentration:** Ocean productivity (0.01-100 mg/m³)
- **Sea Surface Temperature (SST):** 11 μm thermal infrared, 1 km resolution
- **Photosynthetically Available Radiation (PAR):** Light available for photosynthesis
- **Colored Dissolved Organic Matter (CDOM):** Water quality indicator
- **Particulate Organic Carbon (POC):** Carbon export estimates

**Resolution:**
- MODIS: 1 km (SST), 4 km (ocean color)
- VIIRS: 750 m (SST), 1 km (ocean color)
- SeaWiFS: 1 km (ocean color)

**Formats:** netCDF-4, HDF5, GeoTIFF

### 2. NOAA CoralTemp (Coral Reef Watch)

**What:** SST and thermal stress products for coral bleaching monitoring
**Coverage:** Global coral reef locations, 5 km resolution, daily updates
**Period:** 1985-present (historical climatology) + real-time
**URL:** https://coralreefwatch.noaa.gov/

**Products:**
- **Daily SST:** From multiple satellite sensors (AVHRR, MODIS, VIIRS)
- **SST Anomaly:** Difference from 30-year climatology
- **Degree Heating Weeks (DHW):** Accumulated thermal stress (°C-weeks)
- **Bleaching Alert Level:**
  - No Stress (DHW < 4)
  - Bleaching Watch (DHW 4-8)
  - Alert Level 1 (DHW 8-12) - Significant bleaching expected
  - Alert Level 2 (DHW > 12) - Severe bleaching and mortality

**S3 Bucket:** `s3://noaa-coral-reef-watch-pds/` (Public Dataset)

### 3. Copernicus Marine Service

**What:** Global ocean analysis, forecasts, and reanalysis products
**Coverage:** Global oceans, 1993-present + 10-day forecasts
**Access:** Copernicus Marine Data Store API (CMEMS)
**URL:** https://marine.copernicus.eu/

**Products:**
- **GLORYS12 reanalysis:** Temperature, salinity, currents (1993-present, 1/12° resolution)
- **Ocean Surface Currents (OSCAR):** 1/3° resolution, 5-day averages
- **Global Ocean Physics Analysis (CMEMS):** Daily analysis, 1/12° resolution
- **Sea Level Anomaly:** From altimetry (Jason, Sentinel-3)
- **Sea Ice Concentration:** Arctic and Antarctic

**Variables:**
- Temperature (surface to 6000m depth)
- Salinity (PSU)
- Zonal/meridional currents (u/v components)
- Sea surface height
- Mixed layer depth

### 4. Argo Float Data

**What:** Autonomous profiling floats measuring temperature and salinity
**Coverage:** Global oceans (4,000+ active floats), 2000-present
**Vertical Range:** 0-2000m depth (some floats to 4000m or 6000m)
**Sampling:** 10-day cycles, ~300 profiles per float lifetime
**Access:** NOAA ERDDAP, Argo GDAC, AWS Open Data Registry
**URL:** https://argo.ucsd.edu/

**Data Products:**
- **Core Argo:** Temperature, salinity (conductivity), pressure
- **Biogeochemical Argo:** Oxygen, pH, nitrate, chlorophyll, backscatter
- **Deep Argo:** Profiles to 6000m depth
- **Quality Control:** Real-time and delayed-mode QC flags

**S3 Bucket:** `s3://argo-gdac-sandbox/` (via NOAA)

**Uses:**
- Ocean heat content calculations
- Mixed layer depth trends
- Ocean warming rates
- Salinity changes (freshwater cycle)
- Subsurface temperature for hurricane forecasting

### 5. OBIS (Ocean Biodiversity Information System)

**What:** Global marine species occurrence database
**Coverage:** 120+ million species records, 2,600+ datasets
**Taxa:** Fish, mammals, invertebrates, algae, plankton
**Access:** OBIS API, R package (robis), Python
**URL:** https://obis.org/

**Data Fields:**
- Scientific name, taxonomy
- Latitude/longitude
- Date/time of observation
- Depth
- Data source and quality flags
- Environmental context (when available)

**Uses:**
- Species distribution modeling (SDM)
- Marine biodiversity hotspots
- Climate change impacts on species ranges
- Invasion risk assessment

### 6. GBIF (Global Biodiversity Information Facility)

**What:** Global species occurrence database (terrestrial and marine)
**Coverage:** 2+ billion records (subset marine)
**Access:** GBIF API, rgbif (R), pygbif (Python)
**URL:** https://www.gbif.org/

**Marine Subset:** ~150 million marine records from museums, surveys, citizen science

### 7. NOAA ERDDAP Servers

**What:** Unified data access to oceanographic datasets
**Coverage:** 50+ ERDDAP servers worldwide
**Protocol:** OPeNDAP, RESTful API
**URL:** https://coastwatch.pfeg.noaa.gov/erddap/

**Access Methods:**
- **Direct URLs:** CSV, JSON, netCDF downloads
- **OPeNDAP:** Subsetting and striding before download
- **WMS:** Web Map Service for visualization

**Example Datasets:**
- NOAA buoy data (National Data Buoy Center)
- Satellite SST and chlorophyll
- Ocean model outputs (HYCOM, ROMS)
- Glider deployments

### 8. GEBCO (General Bathymetric Chart of the Oceans)

**What:** Global ocean bathymetry and undersea features
**Resolution:** 15 arc-seconds (~450m at equator)
**Coverage:** Global
**URL:** https://www.gebco.net/

**Uses:**
- Habitat suitability modeling (depth predictor)
- Ocean circulation studies
- Marine spatial planning
- Submarine canyon and seamount mapping

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
boto3==1.28.25
pandas==2.0.3
numpy==1.24.3
xarray==2023.7.0
netCDF4==1.6.4
h5netcdf==1.2.0
zarr==2.16.0
dask==2023.7.1
cartopy==0.21.1
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
scipy==1.11.1
rasterio==1.3.8
geopandas==0.13.2
shapely==2.0.1
pyproj==3.6.0
requests==2.31.0
aiohttp==3.8.5
erddapy==2.0.0
oceanparcels==2.4.2  # Lagrangian particle tracking
gsw==3.6.16  # TEOS-10 seawater properties
cmocean==3.0.3  # Oceanographic colormaps
awswrangler==3.2.0
```

### Deploy Infrastructure

```bash
# Deploy CloudFormation stack
aws cloudformation create-stack \
  --stack-name ocean-analysis \
  --template-body file://cloudformation/ocean-analysis-stack.yml \
  --parameters file://cloudformation/parameters.json \
  --capabilities CAPABILITY_NAMED_IAM

# Wait for completion (15-20 minutes)
aws cloudformation wait stack-create-complete \
  --stack-name ocean-analysis

# Get outputs
aws cloudformation describe-stacks \
  --stack-name ocean-analysis \
  --query 'Stacks[0].Outputs'
```

### Initial Data Setup

```python
from src.data_ingestion import ERDDAPLoader, ArgoLoader, OBISLoader
import xarray as xr

# Initialize data loaders
erddap = ERDDAPLoader(bucket_name='ocean-data-lake')
argo = ArgoLoader(bucket_name='ocean-data-lake')

# Download MODIS chlorophyll-a (last month)
modis_chl = erddap.download_dataset(
    dataset_id='erdMH1chla1day',
    variables=['chlorophyll'],
    bbox=[-180, -90, 180, 90],
    start_date='2024-01-01',
    end_date='2024-01-31'
)

# Download Argo float profiles (Pacific Ocean)
argo_profiles = argo.download_region(
    bbox=[-160, 10, -120, 40],
    start_date='2024-01-01',
    end_date='2024-01-31',
    parameters=['temperature', 'salinity']
)

# Download species occurrences from OBIS
from src.data_ingestion import OBISLoader

obis = OBISLoader()
species_data = obis.download_occurrences(
    species='Thunnus albacares',  # Yellowfin tuna
    bbox=[-180, -90, 180, 90],
    start_date='2000-01-01',
    end_date='2024-01-01'
)

print(f"Downloaded {len(species_data)} yellowfin tuna occurrences")
```

## Core Analyses

### 1. Sea Surface Temperature Analysis and Marine Heat Waves

Detect and characterize marine heat waves using satellite SST data.

```python
import xarray as xr
import numpy as np
import pandas as pd
from src.marine_heatwaves import detect_mhw
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Download NOAA SST data from ERDDAP
from erddapy import ERDDAP

e = ERDDAP(
    server='https://coastwatch.pfeg.noaa.gov/erddap',
    protocol='griddap'
)

e.dataset_id = 'ncdcOisst21Agg_LonPM180'  # NOAA OI SST V2.1
e.constraints = {
    'time>=': '1990-01-01T00:00:00Z',
    'time<=': '2024-01-01T00:00:00Z',
    'latitude>=': 30,
    'latitude<=': 50,
    'longitude>=': -130,
    'longitude<=': -110
}
e.variables = ['sst']

# Download data
sst_ds = e.to_xarray()
print(f"SST data shape: {sst_ds.sst.shape}")
# Output: (12,419 days, 80 lats, 80 lons)

# Calculate climatology (baseline: 1991-2020)
climatology_period = slice('1991-01-01', '2020-12-31')
sst_climatology = sst_ds.sel(time=climatology_period).groupby('time.dayofyear').mean('time')

# Calculate daily anomalies
sst_anomaly = sst_ds.groupby('time.dayofyear') - sst_climatology

# Detect marine heat waves (Hobday et al. 2016 definition)
def detect_marine_heatwave(sst_timeseries, threshold_percentile=90, min_duration=5):
    """
    Detect marine heat waves following Hobday et al. (2016)

    Parameters:
    -----------
    sst_timeseries : xr.DataArray
        Daily SST time series
    threshold_percentile : float
        Percentile threshold (typically 90th)
    min_duration : int
        Minimum duration in days (typically 5)

    Returns:
    --------
    mhw_events : list of dicts
        Each dict contains: start_date, end_date, duration, intensity_max,
                           intensity_mean, intensity_cumulative
    """
    # Calculate threshold (90th percentile for each day of year)
    threshold = sst_timeseries.groupby('time.dayofyear').quantile(
        threshold_percentile / 100.0,
        dim='time'
    )

    # Compare each day to its threshold
    anomaly = sst_timeseries.groupby('time.dayofyear') - threshold
    exceeds = anomaly > 0

    # Find continuous periods exceeding threshold
    events = []
    in_event = False
    event_start = None
    event_values = []

    for i, (date, exceed) in enumerate(zip(sst_timeseries.time.values, exceeds.values)):
        if exceed and not in_event:
            # Start of new event
            in_event = True
            event_start = date
            event_values = [anomaly.values[i]]
        elif exceed and in_event:
            # Continue event
            event_values.append(anomaly.values[i])
        elif not exceed and in_event:
            # End of event
            if len(event_values) >= min_duration:
                events.append({
                    'start_date': pd.Timestamp(event_start),
                    'end_date': pd.Timestamp(date) - pd.Timedelta(days=1),
                    'duration': len(event_values),
                    'intensity_max': np.max(event_values),
                    'intensity_mean': np.mean(event_values),
                    'intensity_cumulative': np.sum(event_values)
                })
            in_event = False
            event_start = None
            event_values = []

    return events

# Detect MHWs at a specific location (Pacific Northwest coast)
lat, lon = 45.0, -125.0
location_sst = sst_ds.sel(latitude=lat, longitude=lon, method='nearest').sst

mhw_events = detect_marine_heatwave(location_sst)
print(f"Detected {len(mhw_events)} marine heat wave events")

# Analyze major events
mhw_df = pd.DataFrame(mhw_events)
mhw_df = mhw_df.sort_values('intensity_cumulative', ascending=False)

print("\nTop 5 most intense marine heat waves:")
print(mhw_df[['start_date', 'end_date', 'duration', 'intensity_max', 'intensity_cumulative']].head())

# Expected output for Pacific Northwest:
# 2014-2016 "Blob" event: duration ~700 days, intensity_cumulative ~1500 °C-days
# 2019 event: duration ~100 days
# 2021 event: duration ~80 days

# Visualize the 2014-2016 "Blob" event
blob_start = pd.Timestamp('2014-01-01')
blob_end = pd.Timestamp('2016-12-31')
blob_period = slice(blob_start, blob_end)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Plot SST time series
ax1 = axes[0]
location_sst.sel(time=blob_period).plot(ax=ax1, label='Observed SST')
threshold_ts = threshold.sel(dayofyear=location_sst.sel(time=blob_period).time.dt.dayofyear)
ax1.plot(location_sst.sel(time=blob_period).time, threshold_ts.values,
         'r--', label='90th percentile threshold')
ax1.set_ylabel('SST (°C)')
ax1.set_title(f'Pacific Northwest Marine Heat Wave (2014-2016)\nLocation: {lat}°N, {abs(lon)}°W')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot anomaly
ax2 = axes[1]
anomaly_blob = sst_anomaly.sel(time=blob_period, latitude=lat, longitude=lon, method='nearest')
anomaly_blob.plot(ax=ax2)
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax2.fill_between(anomaly_blob.time.values, 0, anomaly_blob.values,
                  where=anomaly_blob.values > 0, alpha=0.3, color='red')
ax2.set_ylabel('SST Anomaly (°C)')
ax2.set_xlabel('Date')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('marine_heatwave_2014_2016.png', dpi=300, bbox_inches='tight')

# Spatial map of MHW intensity
def plot_mhw_spatial(sst_ds, start_date, end_date):
    """
    Create spatial map of marine heat wave intensity
    """
    period = slice(start_date, end_date)

    # Calculate mean anomaly during event
    mean_anomaly = sst_anomaly.sel(time=period).mean('time')

    fig = plt.figure(figsize=(14, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Plot
    im = mean_anomaly.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='RdBu_r',
        vmin=-3,
        vmax=3,
        cbar_kwargs={'label': 'Mean SST Anomaly (°C)', 'shrink': 0.6}
    )

    # Add features
    ax.coastlines(resolution='50m', color='black', linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)
    ax.gridlines(draw_labels=True, alpha=0.3)

    ax.set_title(f'Marine Heat Wave Intensity\n{start_date} to {end_date}')

    plt.savefig(f'mhw_spatial_{start_date}_{end_date}.png', dpi=300, bbox_inches='tight')
    return fig

plot_mhw_spatial(sst_ds, '2015-06-01', '2015-08-31')

# Calculate MHW metrics for entire region
def calculate_mhw_trends(sst_ds, start_year=1990, end_year=2024):
    """
    Calculate trends in MHW frequency and intensity
    """
    results = []

    for year in range(start_year, end_year + 1):
        year_data = sst_ds.sel(time=str(year))

        # Calculate spatial mean of daily max anomaly
        daily_max = sst_anomaly.sel(time=str(year)).max(dim=['latitude', 'longitude'])

        # Count days with widespread extreme warmth
        extreme_days = (sst_anomaly.sel(time=str(year)) > 2).sum(dim='time').mean()

        results.append({
            'year': year,
            'max_anomaly': float(daily_max.max().values),
            'mean_extreme_days': float(extreme_days.values)
        })

    return pd.DataFrame(results)

mhw_trends = calculate_mhw_trends(sst_ds)

# Plot trends
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax1 = axes[0]
ax1.plot(mhw_trends['year'], mhw_trends['max_anomaly'], 'o-')
ax1.set_ylabel('Maximum SST Anomaly (°C)')
ax1.set_title('Marine Heat Wave Trends (1990-2024)')
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(mhw_trends['year'], mhw_trends['mean_extreme_days'], 'o-', color='red')
ax2.set_ylabel('Mean Days with Extreme Warmth')
ax2.set_xlabel('Year')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mhw_trends.png', dpi=300, bbox_inches='tight')

print("\nMarine Heat Wave Trends:")
print(f"1990s average max anomaly: {mhw_trends[mhw_trends['year'] < 2000]['max_anomaly'].mean():.2f}°C")
print(f"2020s average max anomaly: {mhw_trends[mhw_trends['year'] >= 2020]['max_anomaly'].mean():.2f}°C")
```

**AWS Batch Processing for Global MHW Analysis:**

```python
# batch_job.py - Process global MHW detection in parallel

import boto3
import xarray as xr
import numpy as np
from datetime import datetime

def process_tile(lat_min, lat_max, lon_min, lon_max, bucket, prefix):
    """
    Process one geographic tile for MHW detection
    Run as AWS Batch job
    """
    # Download SST data for tile
    s3_path = f's3://{bucket}/{prefix}/sst_tile_{lat_min}_{lon_min}.nc'
    sst_ds = xr.open_dataset(s3_path, engine='h5netcdf')

    # Calculate climatology and detect MHWs
    climatology = sst_ds.groupby('time.dayofyear').mean('time')
    anomaly = sst_ds.groupby('time.dayofyear') - climatology

    threshold = sst_ds.groupby('time.dayofyear').quantile(0.90, dim='time')
    exceeds = anomaly > threshold

    # Calculate MHW metrics
    mhw_frequency = exceeds.sum('time') / len(sst_ds.time)
    mhw_max_intensity = anomaly.where(exceeds).max('time')
    mhw_mean_intensity = anomaly.where(exceeds).mean('time')

    # Save results
    results = xr.Dataset({
        'mhw_frequency': mhw_frequency,
        'mhw_max_intensity': mhw_max_intensity,
        'mhw_mean_intensity': mhw_mean_intensity
    })

    output_path = f's3://{bucket}/{prefix}/mhw_results_{lat_min}_{lon_min}.nc'
    results.to_netcdf(output_path, engine='h5netcdf')

    return output_path

# Submit batch jobs for all tiles
def submit_global_mhw_analysis():
    batch = boto3.client('batch')

    # Divide globe into 10° x 10° tiles
    tiles = []
    for lat in range(-90, 90, 10):
        for lon in range(-180, 180, 10):
            tiles.append((lat, lat + 10, lon, lon + 10))

    print(f"Submitting {len(tiles)} batch jobs...")

    job_ids = []
    for lat_min, lat_max, lon_min, lon_max in tiles:
        response = batch.submit_job(
            jobName=f'mhw_tile_{lat_min}_{lon_min}',
            jobQueue='ocean-analysis-queue',
            jobDefinition='ocean-analysis-job',
            containerOverrides={
                'command': [
                    'python', 'batch_job.py',
                    '--lat-min', str(lat_min),
                    '--lat-max', str(lat_max),
                    '--lon-min', str(lon_min),
                    '--lon-max', str(lon_max)
                ],
                'vcpus': 4,
                'memory': 16384
            }
        )
        job_ids.append(response['jobId'])

    return job_ids

# Monitor job progress
def monitor_jobs(job_ids):
    batch = boto3.client('batch')

    while True:
        response = batch.describe_jobs(jobs=job_ids)

        status_counts = {}
        for job in response['jobs']:
            status = job['status']
            status_counts[status] = status_counts.get(status, 0) + 1

        print(f"Job status: {status_counts}")

        if all(job['status'] in ['SUCCEEDED', 'FAILED'] for job in response['jobs']):
            break

        time.sleep(60)

    print("All jobs completed")
```

### 2. Argo Float Analysis for Ocean Stratification

Analyze temperature and salinity profiles from Argo floats to assess ocean heat content and stratification changes.

```python
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.argo_analysis import ArgoDataset
import gsw  # TEOS-10 Gibbs SeaWater toolbox

# Load Argo data from S3
# Data structure: One netCDF file per float
from src.data_access import load_argo_profiles

profiles = load_argo_profiles(
    region='north_pacific',
    time_range=('2010-01-01', '2024-01-01'),
    quality_flags=['good', 'probably_good']
)

print(f"Loaded {len(profiles)} Argo profiles")

# Calculate derived quantities using TEOS-10
def calculate_ocean_properties(profile):
    """
    Calculate derived oceanographic properties
    """
    # Extract variables
    pressure = profile['PRES']  # Pressure (dbar)
    temperature = profile['TEMP']  # In-situ temperature (°C)
    salinity = profile['PSAL']  # Practical salinity (PSU)
    latitude = profile['LATITUDE']
    longitude = profile['LONGITUDE']

    # Convert to absolute salinity and conservative temperature
    SA = gsw.SA_from_SP(salinity, pressure, longitude, latitude)
    CT = gsw.CT_from_t(SA, temperature, pressure)

    # Calculate potential density (referenced to surface)
    sigma0 = gsw.sigma0(SA, CT)

    # Calculate buoyancy frequency (N²)
    N2, p_mid = gsw.Nsquared(SA, CT, pressure, latitude)

    # Calculate mixed layer depth (MLD)
    # Definition: depth where sigma0 exceeds surface value by 0.03 kg/m³
    surface_sigma0 = sigma0[0]
    mld_indices = np.where(sigma0 > surface_sigma0 + 0.03)[0]
    mld = pressure[mld_indices[0]] if len(mld_indices) > 0 else np.nan

    # Calculate ocean heat content (0-700m)
    depth_mask = pressure <= 700
    heat_content = np.trapz(
        CT[depth_mask] * 1026 * 3991,  # ρ * cp * T
        pressure[depth_mask]
    ) / 1e9  # Convert to GJ/m²

    return {
        'SA': SA,
        'CT': CT,
        'sigma0': sigma0,
        'N2': N2,
        'mld': mld,
        'heat_content': heat_content
    }

# Process all profiles
results = []
for profile in profiles:
    try:
        props = calculate_ocean_properties(profile)
        results.append({
            'time': profile['JULD'],
            'latitude': profile['LATITUDE'],
            'longitude': profile['LONGITUDE'],
            'mld': props['mld'],
            'heat_content': props['heat_content'],
            'surface_temp': props['CT'][0],
            'surface_salinity': props['SA'][0]
        })
    except Exception as e:
        print(f"Error processing profile: {e}")
        continue

df = pd.DataFrame(results)
df['time'] = pd.to_datetime(df['time'], origin='julian', unit='D')

print(f"Successfully processed {len(df)} profiles")

# Analyze mixed layer depth trends
df_yearly = df.set_index('time').resample('Y').mean()

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Plot 1: Mixed layer depth trend
ax1 = axes[0]
ax1.plot(df_yearly.index, df_yearly['mld'], 'o-', linewidth=2)
z = np.polyfit(range(len(df_yearly)), df_yearly['mld'], 1)
p = np.poly1d(z)
ax1.plot(df_yearly.index, p(range(len(df_yearly))), 'r--',
         label=f'Trend: {z[0]:.2f} m/year')
ax1.set_ylabel('Mixed Layer Depth (m)')
ax1.set_title('North Pacific Ocean Properties from Argo Floats (2010-2024)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.invert_yaxis()

# Plot 2: Ocean heat content trend
ax2 = axes[1]
ax2.plot(df_yearly.index, df_yearly['heat_content'], 'o-', color='red', linewidth=2)
z = np.polyfit(range(len(df_yearly)), df_yearly['heat_content'], 1)
p = np.poly1d(z)
ax2.plot(df_yearly.index, p(range(len(df_yearly))), 'k--',
         label=f'Trend: {z[0]:.3f} GJ/m²/year')
ax2.set_ylabel('Ocean Heat Content (GJ/m²)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Surface temperature trend
ax3 = axes[2]
ax3.plot(df_yearly.index, df_yearly['surface_temp'], 'o-', color='blue', linewidth=2)
z = np.polyfit(range(len(df_yearly)), df_yearly['surface_temp'], 1)
p = np.poly1d(z)
ax3.plot(df_yearly.index, p(range(len(df_yearly))), 'k--',
         label=f'Trend: {z[0]:.3f} °C/year')
ax3.set_ylabel('Surface Temperature (°C)')
ax3.set_xlabel('Year')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('argo_ocean_properties_trends.png', dpi=300, bbox_inches='tight')

# Spatial distribution of heat content change
def plot_heat_content_change(df, start_year, end_year):
    """
    Map spatial pattern of ocean heat content change
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # Calculate change
    period1 = df[df['time'].dt.year.between(start_year, start_year + 2)]
    period2 = df[df['time'].dt.year.between(end_year - 2, end_year)]

    # Grid data
    lat_bins = np.arange(20, 50, 2)
    lon_bins = np.arange(-180, -110, 2)

    heat_change = np.zeros((len(lat_bins) - 1, len(lon_bins) - 1))

    for i in range(len(lat_bins) - 1):
        for j in range(len(lon_bins) - 1):
            mask1 = (
                (period1['latitude'] >= lat_bins[i]) &
                (period1['latitude'] < lat_bins[i + 1]) &
                (period1['longitude'] >= lon_bins[j]) &
                (period1['longitude'] < lon_bins[j + 1])
            )
            mask2 = (
                (period2['latitude'] >= lat_bins[i]) &
                (period2['latitude'] < lat_bins[i + 1]) &
                (period2['longitude'] >= lon_bins[j]) &
                (period2['longitude'] < lon_bins[j + 1])
            )

            if mask1.sum() > 0 and mask2.sum() > 0:
                heat_change[i, j] = (
                    period2.loc[mask2, 'heat_content'].mean() -
                    period1.loc[mask1, 'heat_content'].mean()
                )
            else:
                heat_change[i, j] = np.nan

    # Plot
    fig = plt.figure(figsize=(14, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
    lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2

    im = ax.pcolormesh(
        lon_bins, lat_bins, heat_change,
        transform=ccrs.PlateCarree(),
        cmap='RdBu_r',
        vmin=-2, vmax=2,
        shading='auto'
    )

    ax.coastlines(resolution='50m', color='black', linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.gridlines(draw_labels=True, alpha=0.3)

    plt.colorbar(im, ax=ax, label='Ocean Heat Content Change (GJ/m²)', shrink=0.6)
    ax.set_title(f'Ocean Heat Content Change from Argo Floats\n{start_year}-{start_year+2} to {end_year-2}-{end_year}')

    plt.savefig(f'heat_content_change_{start_year}_{end_year}.png', dpi=300, bbox_inches='tight')

plot_heat_content_change(df, 2010, 2024)

# Temperature-Salinity diagram
def plot_ts_diagram(profiles, year=None):
    """
    Create Temperature-Salinity diagram
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    for profile in profiles[:100]:  # Plot subset for clarity
        if year is None or profile['JULD'].year == year:
            props = calculate_ocean_properties(profile)
            ax.scatter(props['SA'], props['CT'], c=profile['PRES'],
                      cmap='viridis', s=1, alpha=0.5)

    # Add density contours
    SA_range = np.linspace(32, 36, 100)
    CT_range = np.linspace(0, 30, 100)
    SA_grid, CT_grid = np.meshgrid(SA_range, CT_range)
    sigma0_grid = gsw.sigma0(SA_grid, CT_grid)

    contours = ax.contour(SA_grid, CT_grid, sigma0_grid,
                          levels=np.arange(20, 28, 0.5),
                          colors='gray', linewidths=0.5, alpha=0.5)
    ax.clabel(contours, inline=True, fontsize=8)

    ax.set_xlabel('Absolute Salinity (g/kg)')
    ax.set_ylabel('Conservative Temperature (°C)')
    ax.set_title(f'Temperature-Salinity Diagram\nNorth Pacific Argo Profiles')
    ax.grid(True, alpha=0.3)

    plt.colorbar(ax.collections[0], ax=ax, label='Pressure (dbar)')
    plt.savefig('ts_diagram.png', dpi=300, bbox_inches='tight')

plot_ts_diagram(profiles)

# Calculate ocean warming rate
def calculate_warming_rate(df, depth_layer='0-700m'):
    """
    Calculate decadal ocean warming rate
    """
    # Fit linear trend to heat content
    years = (df['time'] - df['time'].min()).dt.days / 365.25

    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(years, df['heat_content'])

    # Convert to more intuitive units
    warming_rate_decade = slope * 10  # GJ/m²/decade
    warming_watts = slope * 1e9 / (365.25 * 24 * 3600)  # W/m²

    return {
        'warming_rate_decade': warming_rate_decade,
        'warming_watts_m2': warming_watts,
        'r_squared': r_value**2,
        'p_value': p_value
    }

warming = calculate_warming_rate(df)
print("\nOcean Warming Analysis (0-700m):")
print(f"  Warming rate: {warming['warming_rate_decade']:.2f} GJ/m²/decade")
print(f"  Equivalent: {warming['warming_watts_m2']:.3f} W/m²")
print(f"  R²: {warming['r_squared']:.3f}")
print(f"  p-value: {warming['p_value']:.2e}")

# Context: IPCC AR6 reports global ocean (0-700m) warming of ~0.4-0.6 W/m²
```

### 3. Coral Reef Health Monitoring with Satellite Imagery

Detect coral bleaching risk and map benthic habitats using Landsat and Sentinel-2.

```python
import rasterio
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from src.coral_monitoring import download_landsat, calculate_indices

# Download Landsat 8/9 imagery for coral reef site
# Example: Great Barrier Reef
reef_bbox = (145.5, -16.5, 146.0, -16.0)  # lon_min, lat_min, lon_max, lat_max

from src.data_access import LandsatDownloader

downloader = LandsatDownloader(
    collection='landsat-c2l2-sr',  # Surface reflectance
    cloud_cover_max=10
)

scenes = downloader.search(
    bbox=reef_bbox,
    date_range=('2024-01-01', '2024-01-31')
)

print(f"Found {len(scenes)} Landsat scenes")

# Download best scene
best_scene = scenes[0]
imagery = downloader.download(best_scene, bands=['B2', 'B3', 'B4', 'B5', 'B6'])

# Load data
with rasterio.open(imagery['B2']) as src:
    blue = src.read(1).astype(float)
    transform = src.transform
    crs = src.crs

with rasterio.open(imagery['B3']) as src:
    green = src.read(1).astype(float)

with rasterio.open(imagery['B4']) as src:
    red = src.read(1).astype(float)

with rasterio.open(imagery['B5']) as src:
    nir = src.read(1).astype(float)

with rasterio.open(imagery['B6']) as src:
    swir1 = src.read(1).astype(float)

# Apply scaling factors (Landsat Collection 2)
scale_factor = 0.0000275
offset = -0.2

blue = blue * scale_factor + offset
green = green * scale_factor + offset
red = red * scale_factor + offset
nir = nir * scale_factor + offset
swir1 = swir1 * scale_factor + offset

# Calculate water quality indices
def calculate_water_indices(blue, green, red, nir, swir1):
    """
    Calculate indices for coral reef monitoring
    """
    # Normalized Difference Water Index (NDWI) - identify water
    ndwi = (green - nir) / (green + nir + 1e-8)

    # Normalized Difference Vegetation Index (NDVI) - for context
    ndvi = (nir - red) / (nir + red + 1e-8)

    # Chlorophyll index (approximate)
    chl_index = (green - blue) / (green + blue + 1e-8)

    # Turbidity index
    turbidity = red / blue

    # Depth-invariant index (Lyzenga 1978) for benthic classification
    # Assumes Beer-Lambert law: L = L_bottom * exp(-2 * k * depth)
    # Ratio blue/green is less sensitive to depth
    depth_invariant_bg = np.log(blue + 1e-8) / np.log(green + 1e-8)
    depth_invariant_br = np.log(blue + 1e-8) / np.log(red + 1e-8)

    return {
        'ndwi': ndwi,
        'ndvi': ndvi,
        'chl_index': chl_index,
        'turbidity': turbidity,
        'depth_inv_bg': depth_invariant_bg,
        'depth_inv_br': depth_invariant_br
    }

indices = calculate_water_indices(blue, green, red, nir, swir1)

# Mask to water only (NDWI > 0)
water_mask = indices['ndwi'] > 0

# Create RGB visualization
rgb = np.dstack([red, green, blue])
rgb = np.clip(rgb * 3, 0, 1)  # Brighten for visualization

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(rgb)
axes[0, 0].set_title('True Color (RGB)')
axes[0, 0].axis('off')

axes[0, 1].imshow(indices['ndwi'], cmap='RdBu', vmin=-1, vmax=1)
axes[0, 1].set_title('NDWI (Water Index)')
axes[0, 1].axis('off')

axes[0, 2].imshow(np.ma.masked_where(~water_mask, indices['turbidity']),
                   cmap='YlOrRd', vmin=0, vmax=2)
axes[0, 2].set_title('Turbidity')
axes[0, 2].axis('off')

axes[1, 0].imshow(np.ma.masked_where(~water_mask, indices['chl_index']),
                   cmap='Greens', vmin=-0.5, vmax=0.5)
axes[1, 0].set_title('Chlorophyll Index')
axes[1, 0].axis('off')

axes[1, 1].imshow(np.ma.masked_where(~water_mask, indices['depth_inv_bg']),
                   cmap='viridis', vmin=-2, vmax=2)
axes[1, 1].set_title('Depth-Invariant Index (B/G)')
axes[1, 1].axis('off')

axes[1, 2].imshow(water_mask, cmap='Blues')
axes[1, 2].set_title('Water Mask')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('coral_reef_indices.png', dpi=300, bbox_inches='tight')

# Benthic habitat classification with Random Forest
# Requires training data (labeled polygons of coral, sand, algae, etc.)

def classify_benthic_habitats(imagery, training_data_path):
    """
    Classify coral reef benthic habitats

    Classes: coral, sand, algae, seagrass, deep_water
    """
    # Load training data (GeoJSON with labeled polygons)
    import geopandas as gpd

    training = gpd.read_file(training_data_path)

    # Extract features for training pixels
    X_train = []
    y_train = []

    for idx, row in training.iterrows():
        # Get pixels within polygon
        # (Simplified - in production, use proper rasterization)
        class_label = row['class']

        # Extract spectral values
        # Add code to extract pixel values within polygon geometry
        pass

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Train Random Forest classifier
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        random_state=42
    )

    rf.fit(X_train, y_train)

    # Predict for entire image
    # Stack all bands into feature array
    h, w = blue.shape
    features = np.dstack([
        blue, green, red, nir, swir1,
        indices['ndwi'], indices['chl_index'], indices['turbidity'],
        indices['depth_inv_bg'], indices['depth_inv_br']
    ])

    X_predict = features.reshape(-1, features.shape[2])

    # Mask to water only
    water_mask_flat = water_mask.flatten()
    X_predict_water = X_predict[water_mask_flat]

    # Predict
    predictions = rf.predict(X_predict_water)

    # Reshape to image
    classified = np.zeros(h * w, dtype=int)
    classified[water_mask_flat] = predictions
    classified = classified.reshape(h, w)

    return classified, rf

# Simulated example (requires actual training data)
# classified, model = classify_benthic_habitats(imagery, 'training_polygons.geojson')

# Coral bleaching risk assessment using SST
from src.data_access import download_coral_temp

# Download NOAA CoralTemp data for reef location
coral_temp = download_coral_temp(
    bbox=reef_bbox,
    date_range=('2023-01-01', '2024-01-31')
)

# Calculate Degree Heating Weeks (DHW)
def calculate_dhw(sst_daily, climatology, threshold_c=1.0):
    """
    Calculate Degree Heating Weeks (DHW) - accumulated thermal stress

    Parameters:
    -----------
    sst_daily : xr.DataArray
        Daily SST time series
    climatology : xr.DataArray
        Monthly climatology (maximum monthly mean)
    threshold_c : float
        Temperature threshold above MMM (default 1°C)
    """
    # Calculate hotspot (SST - MMM)
    # For each day, compare to maximum monthly mean for that month
    mmm = climatology.sel(month=sst_daily.time.dt.month)
    hotspot = sst_daily - mmm

    # Positive values only
    hotspot = hotspot.where(hotspot > threshold_c, 0)

    # Accumulate over 12-week rolling window
    # DHW = sum of positive hotspots over 84 days / 7 days
    dhw = hotspot.rolling(time=84, min_periods=1).sum() / 7

    return dhw, hotspot

# Load climatology (maximum monthly mean from 1985-1993 baseline)
climatology = coral_temp.groupby('time.month').max('time')

dhw, hotspot = calculate_dhw(coral_temp['sst'], climatology)

# Plot bleaching alert
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Plot 1: SST and MMM
ax1 = axes[0]
coral_temp['sst'].plot(ax=ax1, label='Daily SST')
mmm_expanded = climatology.sel(month=coral_temp.time.dt.month)
ax1.plot(coral_temp.time, mmm_expanded, 'r--', label='Maximum Monthly Mean (MMM)')
ax1.plot(coral_temp.time, mmm_expanded + 1, 'orange', linestyle=':',
         label='Bleaching Threshold (MMM + 1°C)')
ax1.set_ylabel('SST (°C)')
ax1.set_title('Coral Bleaching Risk Assessment - Great Barrier Reef')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Hotspot
ax2 = axes[1]
hotspot.plot(ax=ax2, color='orange')
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Bleaching threshold')
ax2.fill_between(hotspot.time.values, 0, hotspot.values,
                  where=hotspot.values > 0, alpha=0.3, color='orange')
ax2.set_ylabel('Hotspot (°C)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Degree Heating Weeks
ax3 = axes[2]
dhw.plot(ax=ax3, color='red', linewidth=2)

# Add alert level thresholds
ax3.axhline(y=4, color='yellow', linestyle='--', label='Bleaching Watch')
ax3.axhline(y=8, color='orange', linestyle='--', label='Alert Level 1')
ax3.axhline(y=12, color='red', linestyle='--', label='Alert Level 2')
ax3.fill_between(dhw.time.values, 0, dhw.values,
                  where=dhw.values >= 4, alpha=0.3, color='red')
ax3.set_ylabel('Degree Heating Weeks (°C-weeks)')
ax3.set_xlabel('Date')
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0, max(16, float(dhw.max()) + 2))

plt.tight_layout()
plt.savefig('coral_bleaching_risk.png', dpi=300, bbox_inches='tight')

# Alert system
def generate_bleaching_alert(dhw_current):
    """
    Generate coral bleaching alert based on current DHW
    """
    if dhw_current < 4:
        return {
            'level': 'No Stress',
            'color': 'green',
            'message': 'No significant thermal stress detected.'
        }
    elif dhw_current < 8:
        return {
            'level': 'Bleaching Watch',
            'color': 'yellow',
            'message': 'Thermal stress present. Monitor closely for bleaching.'
        }
    elif dhw_current < 12:
        return {
            'level': 'Alert Level 1',
            'color': 'orange',
            'message': 'Significant bleaching likely. Expect coral mortality in sensitive species.'
        }
    else:
        return {
            'level': 'Alert Level 2',
            'color': 'red',
            'message': 'Severe bleaching expected. Widespread coral mortality likely.'
        }

current_dhw = float(dhw.isel(time=-1).values)
alert = generate_bleaching_alert(current_dhw)

print(f"\nCurrent Bleaching Alert:")
print(f"  Level: {alert['level']}")
print(f"  Current DHW: {current_dhw:.1f} °C-weeks")
print(f"  Message: {alert['message']}")

# Lambda function for automated alerts
"""
# lambda_function.py - Automated coral bleaching alerts

import boto3
import xarray as xr
from datetime import datetime, timedelta

sns = boto3.client('sns')

def lambda_handler(event, context):
    '''
    Check coral bleaching risk daily and send SNS alerts
    '''
    # Download latest CoralTemp data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    coral_temp = download_coral_temp(
        bbox=reef_bbox,
        date_range=(start_date, end_date)
    )

    # Calculate DHW
    dhw, _ = calculate_dhw(coral_temp['sst'], climatology)
    current_dhw = float(dhw.isel(time=-1).values)

    # Generate alert
    alert = generate_bleaching_alert(current_dhw)

    # Send SNS notification if alert level >= Watch
    if current_dhw >= 4:
        message = f'''
        Coral Bleaching Alert - Great Barrier Reef

        Alert Level: {alert['level']}
        Current DHW: {current_dhw:.1f} °C-weeks
        Date: {end_date.strftime('%Y-%m-%d')}

        {alert['message']}

        Visit dashboard: https://example.com/coral-monitoring
        '''

        sns.publish(
            TopicArn='arn:aws:sns:us-east-1:123456789:coral-alerts',
            Subject=f"Coral Bleaching {alert['level']}",
            Message=message
        )

    return {'statusCode': 200, 'alert_level': alert['level']}
"""
```

### 4. Marine Species Distribution Modeling

Model habitat suitability for marine species using occurrence data and environmental predictors.

```python
import pandas as pd
import numpy as np
import xarray as xr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Download species occurrence data from OBIS
from src.data_access import download_obis_occurrences

# Example: Yellowfin tuna (Thunnus albacares)
species_name = 'Thunnus albacares'

occurrences = download_obis_occurrences(
    scientific_name=species_name,
    start_date='2010-01-01',
    end_date='2024-01-01',
    has_coords=True,
    marine_only=True
)

print(f"Downloaded {len(occurrences)} occurrence records for {species_name}")

# Clean data
occurrences = occurrences.dropna(subset=['decimalLatitude', 'decimalLongitude'])
occurrences = occurrences[
    (occurrences['decimalLatitude'] >= -90) &
    (occurrences['decimalLatitude'] <= 90) &
    (occurrences['decimalLongitude'] >= -180) &
    (occurrences['decimalLongitude'] <= 180)
]

# Generate pseudo-absences (background points)
# MaxEnt approach: sample from study area
n_background = len(occurrences) * 3

# Define study area (global oceans between 40°N and 40°S for tropical tuna)
study_area = {
    'lat_min': -40,
    'lat_max': 40,
    'lon_min': -180,
    'lon_max': 180
}

background = pd.DataFrame({
    'decimalLatitude': np.random.uniform(
        study_area['lat_min'], study_area['lat_max'], n_background
    ),
    'decimalLongitude': np.random.uniform(
        study_area['lon_min'], study_area['lon_max'], n_background
    ),
    'presence': 0
})

occurrences['presence'] = 1

# Combine presence and background
all_points = pd.concat([
    occurrences[['decimalLatitude', 'decimalLongitude', 'presence']],
    background
], ignore_index=True)

print(f"Total points: {len(all_points)} ({occurrences['presence'].sum()} presences, {n_background} background)")

# Download environmental predictors
from src.data_access import download_environmental_layers

# Get annual mean environmental data
env_data = download_environmental_layers(
    variables=[
        'sea_surface_temperature',
        'sea_surface_salinity',
        'chlorophyll_a',
        'bathymetry',
        'sea_surface_height',
        'current_velocity'
    ],
    year=2020,
    resolution=0.25  # 0.25 degree (~25 km)
)

# Extract environmental values at occurrence points
def extract_environmental_values(points, env_data):
    """
    Extract environmental predictor values at point locations
    """
    features = []

    for idx, row in points.iterrows():
        lat = row['decimalLatitude']
        lon = row['decimalLongitude']

        # Extract values (nearest neighbor)
        values = {
            'presence': row['presence'],
            'latitude': lat,
            'longitude': lon
        }

        for var_name, var_data in env_data.items():
            try:
                value = var_data.sel(
                    latitude=lat, longitude=lon, method='nearest'
                ).values
                values[var_name] = float(value)
            except:
                values[var_name] = np.nan

        features.append(values)

    return pd.DataFrame(features)

print("Extracting environmental values...")
data = extract_environmental_values(all_points, env_data)

# Remove rows with missing data
data_clean = data.dropna()
print(f"Clean data: {len(data_clean)} points")

# Prepare features and target
X = data_clean.drop(['presence', 'latitude', 'longitude'], axis=1)
y = data_clean['presence']

# Add derived features
X['sst_squared'] = X['sea_surface_temperature'] ** 2
X['chl_log'] = np.log10(X['chlorophyll_a'] + 0.01)
X['depth_abs'] = np.abs(X['bathymetry'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)} points")
print(f"Test set: {len(X_test)} points")

# Train Random Forest model
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    class_weight='balanced',  # Handle imbalanced data
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# Evaluate model
from sklearn.metrics import roc_auc_score, classification_report, roc_curve

# Predict probabilities
y_pred_proba = rf.predict_proba(X_test)[:, 1]
y_pred = rf.predict(X_test)

# Calculate AUC
auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nModel Performance:")
print(f"AUC: {auc:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Background', 'Presence']))

# Cross-validation
cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='roc_auc')
print(f"\nCross-validation AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

# Plot feature importance
fig, ax = plt.subplots(figsize=(10, 6))
feature_importance.head(10).plot.barh(x='feature', y='importance', ax=ax)
ax.set_xlabel('Feature Importance')
ax.set_title(f'Feature Importance for {species_name} Distribution')
plt.tight_layout()
plt.savefig('species_feature_importance.png', dpi=300, bbox_inches='tight')

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(fpr, tpr, linewidth=2, label=f'Random Forest (AUC = {auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title(f'ROC Curve - {species_name}')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('species_roc_curve.png', dpi=300, bbox_inches='tight')

# Generate habitat suitability map
def predict_habitat_suitability(rf_model, env_data):
    """
    Create spatial map of habitat suitability
    """
    # Get dimensions
    lats = env_data['sea_surface_temperature'].latitude.values
    lons = env_data['sea_surface_temperature'].longitude.values

    # Stack environmental layers
    features_stack = []
    feature_names = []

    for var_name in ['sea_surface_temperature', 'sea_surface_salinity',
                     'chlorophyll_a', 'bathymetry', 'sea_surface_height',
                     'current_velocity']:
        features_stack.append(env_data[var_name].values)
        feature_names.append(var_name)

    # Reshape for prediction
    n_lat, n_lon = len(lats), len(lons)
    features_array = np.array(features_stack).reshape(len(features_stack), -1).T

    # Add derived features
    sst = features_array[:, 0]
    chl = features_array[:, 2]
    depth = features_array[:, 3]

    features_array = np.column_stack([
        features_array,
        sst ** 2,
        np.log10(chl + 0.01),
        np.abs(depth)
    ])

    # Remove NaN rows (land)
    valid_mask = ~np.isnan(features_array).any(axis=1)
    features_clean = features_array[valid_mask]

    # Predict
    predictions = rf_model.predict_proba(features_clean)[:, 1]

    # Reshape to map
    suitability_map = np.full(n_lat * n_lon, np.nan)
    suitability_map[valid_mask] = predictions
    suitability_map = suitability_map.reshape(n_lat, n_lon)

    return suitability_map, lats, lons

print("Generating habitat suitability map...")
suitability, lats, lons = predict_habitat_suitability(rf, env_data)

# Plot habitat suitability
fig = plt.figure(figsize=(16, 10))
ax = plt.axes(projection=ccrs.PlateCarree())

# Plot suitability
im = ax.pcolormesh(
    lons, lats, suitability,
    transform=ccrs.PlateCarree(),
    cmap='YlOrRd',
    vmin=0, vmax=1,
    shading='auto'
)

# Add features
ax.coastlines(resolution='110m', color='black', linewidth=0.5)
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.gridlines(draw_labels=True, alpha=0.3)

# Plot occurrence points
presence_points = occurrences.sample(n=1000)  # Subsample for visibility
ax.scatter(
    presence_points['decimalLongitude'],
    presence_points['decimalLatitude'],
    c='blue', s=1, alpha=0.3,
    transform=ccrs.PlateCarree(),
    label='Occurrences'
)

plt.colorbar(im, ax=ax, label='Habitat Suitability', shrink=0.6)
ax.set_title(f'Habitat Suitability Map - {species_name}')
ax.legend(loc='lower left')

plt.savefig('species_habitat_suitability.png', dpi=300, bbox_inches='tight')

# Project under climate change scenarios
def project_future_distribution(rf_model, env_data_baseline, env_data_future):
    """
    Project species distribution under future climate scenario
    """
    # Predict for baseline
    suit_baseline, lats, lons = predict_habitat_suitability(rf_model, env_data_baseline)

    # Predict for future
    suit_future, _, _ = predict_habitat_suitability(rf_model, env_data_future)

    # Calculate change
    suit_change = suit_future - suit_baseline

    return suit_baseline, suit_future, suit_change, lats, lons

# Note: Requires future climate data (e.g., CMIP6 projections)
# Download from Copernicus Climate Data Store or CMIP6 archives

# Identify climate refugia (areas that remain suitable)
def identify_refugia(suit_baseline, suit_future, threshold=0.5):
    """
    Identify climate refugia - areas that remain highly suitable
    """
    refugia = (suit_baseline > threshold) & (suit_future > threshold)
    loss = (suit_baseline > threshold) & (suit_future <= threshold)
    gain = (suit_baseline <= threshold) & (suit_future > threshold)

    return {
        'refugia': refugia,
        'loss': loss,
        'gain': gain,
        'refugia_area_pct': 100 * refugia.sum() / (suit_baseline > threshold).sum()
    }

# Calculate suitability statistics by region
def summarize_suitability_by_region(suitability, lats, lons, regions):
    """
    Summarize habitat suitability by ocean region
    """
    results = []

    for region_name, bbox in regions.items():
        lat_mask = (lats >= bbox['lat_min']) & (lats <= bbox['lat_max'])
        lon_mask = (lons >= bbox['lon_min']) & (lons <= bbox['lon_max'])

        region_suit = suitability[np.ix_(lat_mask, lon_mask)]

        results.append({
            'region': region_name,
            'mean_suitability': np.nanmean(region_suit),
            'suitable_area_pct': 100 * np.sum(region_suit > 0.5) / np.sum(~np.isnan(region_suit))
        })

    return pd.DataFrame(results)

ocean_regions = {
    'Western Pacific': {'lat_min': -40, 'lat_max': 40, 'lon_min': 100, 'lon_max': 180},
    'Eastern Pacific': {'lat_min': -40, 'lat_max': 40, 'lon_min': -140, 'lon_max': -80},
    'Atlantic': {'lat_min': -40, 'lat_max': 40, 'lon_min': -80, 'lon_max': 0},
    'Indian Ocean': {'lat_min': -40, 'lat_max': 30, 'lon_min': 40, 'lon_max': 100}
}

regional_summary = summarize_suitability_by_region(suitability, lats, lons, ocean_regions)
print("\nHabitat Suitability by Region:")
print(regional_summary)
```

### 5. Ocean Current Visualization and Particle Tracking

Analyze ocean currents and track particle trajectories using Lagrangian methods.

```python
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import animation
from scipy.interpolate import RegularGridInterpolator

# Download ocean current data from Copernicus Marine
from src.data_access import download_copernicus_currents

# Get OSCAR (Ocean Surface Current Analysis Real-time) data
currents = download_copernicus_currents(
    product='OSCAR',
    bbox=[-160, 10, -120, 40],  # Northeast Pacific
    date_range=('2024-01-01', '2024-01-31')
)

print(f"Current data shape: {currents.u.shape}")
# Output: (31 days, 120 lats, 160 lons)

# Visualize current velocity field
def plot_current_vectors(currents, date_index=0):
    """
    Plot ocean current vectors
    """
    # Select time slice
    u = currents.u.isel(time=date_index)
    v = currents.v.isel(time=date_index)

    # Calculate current speed
    speed = np.sqrt(u**2 + v**2)

    # Create figure
    fig = plt.figure(figsize=(14, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Plot current speed as background
    im = ax.pcolormesh(
        currents.longitude, currents.latitude, speed,
        transform=ccrs.PlateCarree(),
        cmap='YlOrRd',
        vmin=0, vmax=1,
        shading='auto'
    )

    # Subsample for vector plotting
    stride = 5
    lon_sub = currents.longitude.values[::stride]
    lat_sub = currents.latitude.values[::stride]
    u_sub = u.values[::stride, ::stride]
    v_sub = v.values[::stride, ::stride]

    # Plot vectors
    ax.quiver(
        lon_sub, lat_sub, u_sub, v_sub,
        transform=ccrs.PlateCarree(),
        scale=10, scale_units='inches',
        width=0.002, headwidth=3, headlength=4,
        color='white', alpha=0.7
    )

    # Add features
    ax.coastlines(resolution='50m', color='black', linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.gridlines(draw_labels=True, alpha=0.3)

    plt.colorbar(im, ax=ax, label='Current Speed (m/s)', shrink=0.6)

    date_str = str(currents.time.isel(time=date_index).values)[:10]
    ax.set_title(f'Ocean Surface Currents - {date_str}')

    plt.savefig(f'ocean_currents_{date_str}.png', dpi=300, bbox_inches='tight')
    return fig

plot_current_vectors(currents, date_index=15)

# Lagrangian particle tracking
class ParticleTracker:
    """
    Track particles in ocean current fields
    """
    def __init__(self, currents_dataset):
        """
        Initialize with ocean current data

        Parameters:
        -----------
        currents_dataset : xr.Dataset
            Must contain 'u', 'v' (velocity components), 'time', 'latitude', 'longitude'
        """
        self.currents = currents_dataset
        self.time_coords = currents_dataset.time.values
        self.lat_coords = currents_dataset.latitude.values
        self.lon_coords = currents_dataset.longitude.values

        # Create interpolators for each time step
        self.u_interpolators = []
        self.v_interpolators = []

        for t in range(len(self.time_coords)):
            u_data = currents_dataset.u.isel(time=t).values
            v_data = currents_dataset.v.isel(time=t).values

            self.u_interpolators.append(
                RegularGridInterpolator(
                    (self.lat_coords, self.lon_coords),
                    u_data,
                    bounds_error=False,
                    fill_value=0
                )
            )
            self.v_interpolators.append(
                RegularGridInterpolator(
                    (self.lat_coords, self.lon_coords),
                    v_data,
                    bounds_error=False,
                    fill_value=0
                )
            )

    def get_velocity(self, lat, lon, time_index):
        """
        Get velocity at a specific location and time
        """
        u = self.u_interpolators[time_index]((lat, lon))
        v = self.v_interpolators[time_index]((lat, lon))
        return u, v

    def track_particle(self, start_lat, start_lon, n_days, dt_hours=6):
        """
        Track single particle using RK4 integration

        Parameters:
        -----------
        start_lat, start_lon : float
            Starting position
        n_days : int
            Number of days to track
        dt_hours : float
            Time step in hours

        Returns:
        --------
        trajectory : dict
            Contains 'lat', 'lon', 'time' arrays
        """
        # Initialize
        lat = start_lat
        lon = start_lon

        lats = [lat]
        lons = [lon]
        times = [0]

        dt_seconds = dt_hours * 3600
        n_steps = int(n_days * 24 / dt_hours)

        # Convert m/s to degrees per second (approximate)
        # 1 degree latitude ≈ 111 km = 111000 m
        # 1 degree longitude ≈ 111 km * cos(lat)
        meters_per_deg_lat = 111000

        for step in range(n_steps):
            # Determine time index (interpolate between time steps if needed)
            time_frac = step * dt_hours / 24  # Days
            time_index = min(int(time_frac), len(self.time_coords) - 1)

            # RK4 integration
            # k1
            u1, v1 = self.get_velocity(lat, lon, time_index)
            dlat1 = v1 / meters_per_deg_lat
            dlon1 = u1 / (meters_per_deg_lat * np.cos(np.radians(lat)))

            # k2
            lat2 = lat + 0.5 * dlat1 * dt_seconds
            lon2 = lon + 0.5 * dlon1 * dt_seconds
            u2, v2 = self.get_velocity(lat2, lon2, time_index)
            dlat2 = v2 / meters_per_deg_lat
            dlon2 = u2 / (meters_per_deg_lat * np.cos(np.radians(lat2)))

            # k3
            lat3 = lat + 0.5 * dlat2 * dt_seconds
            lon3 = lon + 0.5 * dlon2 * dt_seconds
            u3, v3 = self.get_velocity(lat3, lon3, time_index)
            dlat3 = v3 / meters_per_deg_lat
            dlon3 = u3 / (meters_per_deg_lat * np.cos(np.radians(lat3)))

            # k4
            lat4 = lat + dlat3 * dt_seconds
            lon4 = lon + dlon3 * dt_seconds
            u4, v4 = self.get_velocity(lat4, lon4, time_index)
            dlat4 = v4 / meters_per_deg_lat
            dlon4 = u4 / (meters_per_deg_lat * np.cos(np.radians(lat4)))

            # Update position
            lat += (dlat1 + 2*dlat2 + 2*dlat3 + dlat4) * dt_seconds / 6
            lon += (dlon1 + 2*dlon2 + 2*dlon3 + dlon4) * dt_seconds / 6

            # Wrap longitude
            if lon > 180:
                lon -= 360
            elif lon < -180:
                lon += 360

            # Check bounds
            if lat < -90 or lat > 90:
                break

            lats.append(lat)
            lons.append(lon)
            times.append((step + 1) * dt_hours / 24)

        return {
            'latitude': np.array(lats),
            'longitude': np.array(lons),
            'time_days': np.array(times)
        }

    def track_multiple_particles(self, start_positions, n_days):
        """
        Track multiple particles

        Parameters:
        -----------
        start_positions : list of tuples
            Each tuple is (lat, lon)
        n_days : int
            Number of days to track

        Returns:
        --------
        trajectories : list of dicts
        """
        trajectories = []

        for lat, lon in start_positions:
            print(f"Tracking particle from ({lat:.2f}, {lon:.2f})...")
            traj = self.track_particle(lat, lon, n_days)
            trajectories.append(traj)

        return trajectories

# Initialize tracker
tracker = ParticleTracker(currents)

# Example: Simulate oil spill dispersal
spill_location = (35.0, -140.0)  # Offshore California
n_particles = 100

# Generate particle starting positions (random within 0.5° radius)
start_positions = []
for i in range(n_particles):
    offset_lat = np.random.uniform(-0.5, 0.5)
    offset_lon = np.random.uniform(-0.5, 0.5)
    start_positions.append((
        spill_location[0] + offset_lat,
        spill_location[1] + offset_lon
    ))

# Track particles for 30 days
print(f"Tracking {n_particles} particles for 30 days...")
trajectories = tracker.track_multiple_particles(start_positions, n_days=30)

# Visualize trajectories
fig = plt.figure(figsize=(14, 10))
ax = plt.axes(projection=ccrs.PlateCarree())

# Plot background (mean current speed)
mean_speed = np.sqrt(currents.u.mean('time')**2 + currents.v.mean('time')**2)
im = ax.pcolormesh(
    currents.longitude, currents.latitude, mean_speed,
    transform=ccrs.PlateCarree(),
    cmap='Blues',
    alpha=0.3,
    shading='auto'
)

# Plot trajectories
for traj in trajectories:
    ax.plot(
        traj['longitude'], traj['latitude'],
        transform=ccrs.PlateCarree(),
        color='red', alpha=0.3, linewidth=0.5
    )

# Plot start and end points
for traj in trajectories:
    ax.scatter(
        traj['longitude'][0], traj['latitude'][0],
        c='green', s=20, transform=ccrs.PlateCarree(),
        zorder=5, alpha=0.5
    )
    ax.scatter(
        traj['longitude'][-1], traj['latitude'][-1],
        c='red', s=20, transform=ccrs.PlateCarree(),
        zorder=5, alpha=0.5
    )

# Add features
ax.coastlines(resolution='50m', color='black', linewidth=0.5)
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.gridlines(draw_labels=True, alpha=0.3)

ax.set_title('Lagrangian Particle Tracking - Simulated Oil Spill (30 days)')
ax.set_extent([-165, -115, 20, 50], crs=ccrs.PlateCarree())

# Add legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Start'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='End'),
    Line2D([0], [0], color='red', alpha=0.5, label='Trajectory')
]
ax.legend(handles=legend_elements, loc='upper left')

plt.savefig('particle_tracking_oil_spill.png', dpi=300, bbox_inches='tight')

# Calculate dispersal statistics
final_distances = []
for traj in trajectories:
    # Calculate distance from start to end
    start_lat, start_lon = traj['latitude'][0], traj['longitude'][0]
    end_lat, end_lon = traj['latitude'][-1], traj['longitude'][-1]

    # Haversine distance
    dlat = np.radians(end_lat - start_lat)
    dlon = np.radians(end_lon - start_lon)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(start_lat)) * np.cos(np.radians(end_lat)) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance_km = 6371 * c

    final_distances.append(distance_km)

print(f"\nDispersal Statistics:")
print(f"  Mean distance: {np.mean(final_distances):.1f} km")
print(f"  Max distance: {np.max(final_distances):.1f} km")
print(f"  Median distance: {np.median(final_distances):.1f} km")

# Calculate dispersal area (convex hull of final positions)
from scipy.spatial import ConvexHull

final_positions = np.array([
    [traj['longitude'][-1], traj['latitude'][-1]]
    for traj in trajectories
])

try:
    hull = ConvexHull(final_positions)
    # Approximate area (simplified, not accounting for Earth curvature)
    dispersal_area_deg2 = hull.volume
    dispersal_area_km2 = dispersal_area_deg2 * (111 ** 2)  # Rough conversion

    print(f"  Dispersal area: {dispersal_area_km2:.0f} km²")
except:
    print("  Could not calculate dispersal area")

# Marine connectivity analysis (e.g., for MPA network design)
def calculate_connectivity_matrix(source_locations, n_days=90, n_particles_per_source=50):
    """
    Calculate connectivity between marine locations

    Returns matrix where element (i,j) represents fraction of particles
    from source i that reach destination j
    """
    n_sources = len(source_locations)
    connectivity = np.zeros((n_sources, n_sources))

    for i, source in enumerate(source_locations):
        # Generate particles around source
        particles = [
            (source[0] + np.random.uniform(-0.1, 0.1),
             source[1] + np.random.uniform(-0.1, 0.1))
            for _ in range(n_particles_per_source)
        ]

        # Track particles
        trajs = tracker.track_multiple_particles(particles, n_days)

        # Check which destinations were reached
        for j, dest in enumerate(source_locations):
            count_reached = 0

            for traj in trajs:
                # Check if particle ever came within 1° of destination
                distances = np.sqrt(
                    (traj['latitude'] - dest[0])**2 +
                    (traj['longitude'] - dest[1])**2
                )
                if np.min(distances) < 1.0:
                    count_reached += 1

            connectivity[i, j] = count_reached / n_particles_per_source

        print(f"Processed source {i+1}/{n_sources}")

    return connectivity

# Example: MPA connectivity in California Current
mpa_locations = [
    (36.0, -122.0),  # Monterey Bay
    (34.0, -120.0),  # Channel Islands
    (32.5, -117.5),  # San Diego
]

print("\nCalculating MPA connectivity...")
# connectivity_matrix = calculate_connectivity_matrix(mpa_locations, n_days=90)

# Note: This would take significant time, so we'll skip actual execution
print("(Skipped for brevity - would take ~30 minutes)")
```

## Cost Estimates

Detailed cost breakdown for different scales of ocean analysis:

### Single Research Cruise Analysis
**Scope:** Process 1 month of data from one research cruise (Argo floats, CTD casts, satellite imagery)
**Data Volume:** 100 GB
**Compute:** 20 hours EC2 processing (r5.2xlarge)

**Costs:**
- **S3 storage:** 100 GB × $0.023/GB = $2.30/month
- **EC2 compute:** 20 hours × $0.504/hour = $10.08
- **Data transfer:** 50 GB out × $0.09/GB = $4.50
- **Lambda (data ingestion):** 1,000 invocations × $0.20/million = negligible
- **SageMaker (ML models):** 5 hours ml.m5.xlarge × $0.23/hour = $1.15
- **Total: ~$500-1,500 one-time** (depending on analysis complexity)

### Regional Ocean Study
**Scope:** Monitor 1,000 km² region (e.g., coral reef system) for 1 month
**Data Volume:** 500 GB satellite imagery + 100 GB in-situ
**Compute:** Continuous processing, ML inference

**Monthly Costs:**
- **S3 storage:** 600 GB × $0.023/GB = $13.80
- **EC2 Batch:** 200 hours × $0.504/hour = $100.80
- **Timestream (buoy data):** 10M writes × $0.50/million + 1 GB storage = $6.00
- **SageMaker inference:** 100 hours × $0.23/hour = $23.00
- **Athena queries:** 500 GB scanned × $5/TB = $2.50
- **QuickSight:** 1 user × $24/month = $24.00
- **Total: ~$2,000-5,000/month**

### Basin-Scale Analysis
**Scope:** North Pacific basin (10M km²), 1 month
**Data Volume:** 10 TB satellite + 500 GB Argo
**Compute:** Distributed AWS Batch processing

**Monthly Costs:**
- **S3 storage:** 10.5 TB × $0.023/GB = $241.15
- **S3 requests:** 1M GET × $0.40/million = $0.40
- **AWS Batch:** 2,000 hours × $0.504/hour = $1,008
- **Data transfer:** 1 TB × $0.09/GB = $92.16
- **SageMaker training:** 50 hours ml.p3.2xlarge × $3.06/hour = $153
- **Neptune (contact tracing):** db.r5.large × $0.348/hour × 730 = $254
- **Total: ~$8,000-20,000/month**

### Global Ocean Monitoring
**Scope:** Global oceans, continuous monitoring
**Data Volume:** 100 TB/month
**Compute:** 24/7 processing pipeline

**Monthly Costs:**
- **S3 storage (with Glacier transition):** 100 TB × $0.023/GB (hot) + 500 TB × $0.004/GB (Glacier) = $4,360
- **AWS Batch:** 10,000 hours × $0.504/hour = $5,040
- **Kinesis:** 100 shards × $0.015/hour × 730 = $1,095
- **Lambda:** 100M invocations × $0.20/million = $20
- **SageMaker endpoints:** 10 instances × ml.m5.xlarge × $0.23/hour × 730 = $1,679
- **Athena:** 50 TB scanned × $5/TB = $250
- **QuickSight:** 10 users × $24/month = $240
- **CloudWatch:** $200
- **Total: ~$30,000-75,000/month**

**Cost Optimization Tips:**
- Use Spot Instances for Batch jobs (70% savings)
- Enable S3 Intelligent-Tiering
- Cache frequently accessed datasets
- Use Zarr format for fast cloud-optimized access
- Implement data lifecycle policies (archive to Glacier after 90 days)
- Use Lambda for small tasks instead of EC2

## Performance Benchmarks

### netCDF Processing with xarray
- **Read performance:** 500 MB/s from S3 (with chunking)
- **Processing throughput:** 10 GB/hour per core (statistics, regridding)
- **Dask parallel processing:** Near-linear scaling up to 100 cores

### Satellite Data Processing
- **Landsat scene:** 1 GB → processed in 5 minutes (EC2 r5.xlarge)
- **MODIS daily global:** 10 GB → processed in 30 minutes (AWS Batch, 20 workers)
- **Argo profile extraction:** 1M profiles → queried in 2 seconds (Athena)

### Machine Learning
- **Species distribution model training:** 100K samples, 20 features → 10 minutes (SageMaker ml.m5.xlarge)
- **Coral classification inference:** 10,000 pixels → 1 second (GPU)
- **Particle tracking:** 10,000 particles, 30 days → 5 minutes (vectorized NumPy)

## Best Practices

### Oceanographic Data Standards
- **Follow CF Conventions** for netCDF metadata
- **Use TEOS-10** for seawater properties (not EOS-80)
- **BODC vocabularies** for parameter names
- **ISO 19115** for dataset-level metadata
- **Quality flags:** IODE standard (1=good, 2=probably good, 3=probably bad, 4=bad, 9=missing)

### Cloud-Optimized Formats
- **Zarr:** Best for large multidimensional arrays, cloud-native
- **Cloud-Optimized GeoTIFF (COG):** For satellite imagery
- **Parquet:** For species occurrence tables
- **netCDF-4 with compression:** deflate level 1 (good compression/speed tradeoff)

### Reproducible Science
- **Version control:** Git for code, DVC for data
- **Containerization:** Docker for analysis environments
- **Documentation:** README with DOI, methods, citations
- **Data provenance:** Track all transformations
- **Notebooks:** Jupyter with clear explanations

### Cost Optimization
- **Smart caching:** S3 Select for subsetting before download
- **Compute placement:** EC2 in same region as data (us-east-1 for NOAA/NASA)
- **Batch scheduling:** Off-peak hours for non-urgent jobs
- **Right-sizing:** Monitor CloudWatch metrics, adjust instance types
- **Reserved capacity:** For long-running monitoring systems

### Collaboration
- **Open data:** Publish to NOAA NCEI, Pangaea, Zenodo
- **APIs:** Provide ERDDAP or OPeNDAP access
- **Jupyter Hub:** Shared analysis environment
- **Code sharing:** GitHub with MIT/BSD license
- **Publications:** Preprints to EarthArXiv, bioRxiv

### Ethical Considerations
- **Indigenous rights:** Consult coastal communities on MPA boundaries
- **Fisheries data:** Anonymize vessel locations
- **Dual use:** Consider military applications of ocean models
- **Equitable access:** Provide tools to researchers in developing nations
- **Conservation:** Share coral bleaching alerts with reef managers promptly

## Troubleshooting

### netCDF Read Errors

**Problem:** `ValueError: Unable to open netCDF file`

**Solutions:**
```python
# Check file format
import netCDF4
try:
    ds = netCDF4.Dataset('file.nc')
    print(f"Format: {ds.file_format}")
except Exception as e:
    print(f"Error: {e}")

# Try different engines
import xarray as xr
ds = xr.open_dataset('file.nc', engine='h5netcdf')  # or 'scipy', 'netcdf4'

# Check for compression issues
ds = xr.open_dataset('file.nc', chunks={'time': 1})  # Lazy loading
```

### Coordinate System Issues

**Problem:** Longitude range mismatch (-180 to 180 vs 0 to 360)

**Solutions:**
```python
# Convert 0-360 to -180-180
ds['longitude'] = xr.where(ds['longitude'] > 180, ds['longitude'] - 360, ds['longitude'])
ds = ds.sortby('longitude')

# Or use built-in method
ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
```

### Missing Data Gaps

**Problem:** Satellite data has missing values due to clouds

**Solutions:**
```python
# Temporal interpolation
ds_filled = ds.interpolate_na(dim='time', method='linear')

# Spatial interpolation
ds_filled = ds.interpolate_na(dim='latitude', method='nearest')

# Composite over time
weekly_composite = ds.resample(time='7D').mean()
```

### Memory Issues

**Problem:** `MemoryError: Unable to allocate array`

**Solutions:**
```python
# Use Dask for lazy loading
ds = xr.open_dataset('file.nc', chunks={'time': 10, 'lat': 100, 'lon': 100})

# Process in batches
for year in range(2010, 2024):
    ds_year = ds.sel(time=str(year))
    result = process(ds_year)
    save(result, f'output_{year}.nc')

# Reduce precision
ds['temperature'] = ds['temperature'].astype('float32')  # Instead of float64
```

### Slow S3 Access

**Problem:** Data downloads taking too long

**Solutions:**
```python
# Use S3 Select for subsetting
import awswrangler as wr
df = wr.s3.select_query(
    sql='SELECT * FROM s3object WHERE latitude > 30',
    path='s3://bucket/data.parquet'
)

# Enable S3 Transfer Acceleration
s3_client = boto3.client('s3', config=Config(s3={'use_accelerate_endpoint': True}))

# Use fsspec for efficient access
import fsspec
fs = fsspec.filesystem('s3')
with fs.open('s3://bucket/file.nc') as f:
    ds = xr.open_dataset(f)
```

## Additional Resources

### Oceanography Resources
- **Ocean Data Portal:** https://www.ocean-ops.org/ (Argo, drifters, moorings)
- **ERDDAP Servers List:** https://coastwatch.pfeg.noaa.gov/erddap/index.html
- **Marine Regions:** https://www.marineregions.org/ (Shapefiles for EEZs, MPAs)
- **OBIS Data Portal:** https://obis.org/ (Species occurrences)
- **Copernicus Marine:** https://marine.copernicus.eu/ (Ocean models, forecasts)

### Tutorials and Documentation
- **xarray tutorial:** https://tutorial.xarray.dev/
- **OceanParcels docs:** https://oceanparcels.org/ (Lagrangian tracking)
- **GSW Python:** https://teos-10.github.io/GSW-Python/ (Seawater properties)
- **Cartopy gallery:** https://scitools.org.uk/cartopy/docs/latest/gallery/
- **CF Conventions:** http://cfconventions.org/

### Data Access Tools
- **erddapy (Python):** https://github.com/ioos/erddapy
- **netCDF4-python:** https://unidata.github.io/netcdf4-python/
- **h5netcdf:** https://github.com/h5netcdf/h5netcdf (faster reads)
- **pystac-client:** https://pystac-client.readthedocs.io/ (STAC catalogs)

### Machine Learning for Ocean Science
- **Ocean Data Science Guide:** https://oceanhackweek.github.io/
- **scikit-learn marine examples:** https://github.com/scikit-learn/scikit-learn
- **TensorFlow ocean models:** https://github.com/google-research/oceanbench

### AWS-Specific Resources
- **AWS Open Data Registry - Earth:** https://registry.opendata.aws/
- **AWS Batch Best Practices:** https://docs.aws.amazon.com/batch/latest/userguide/best-practices.html
- **SageMaker for Science:** https://aws.amazon.com/sagemaker/science/

### Ocean Modeling Resources
- **HYCOM:** https://www.hycom.org/ (Global ocean model)
- **ROMS:** https://www.myroms.org/ (Regional ocean model)
- **MOM6:** https://github.com/NOAA-GFDL/MOM6 (Modular Ocean Model)
- **NEMO:** https://www.nemo-ocean.eu/ (European ocean model)

### Contacts and Communities
- **Ocean Hack Week:** https://oceanhackweek.org/ (Annual workshop)
- **IOOS:** https://ioos.noaa.gov/ (US Integrated Ocean Observing System)
- **Ocean Best Practices:** https://www.oceanbestpractices.org/
- **GO-SHIP:** https://www.go-ship.org/ (Global ocean surveys)

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{ocean_analysis_aws,
  title = {Ocean and Marine Ecosystem Analysis at Scale},
  author = {AWS Research Jumpstart Team},
  year = {2024},
  url = {https://github.com/research-jumpstart/research-jumpstart},
  note = {AWS Tier 1 Flagship Project}
}
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Support

For questions or issues:
- Open an issue on GitHub
- Email: research-jumpstart@example.com
- AWS Support: https://aws.amazon.com/support/

---

**Last Updated:** 2024-11-13
**Project Status:** Production Ready
**Tier:** 1 Flagship
