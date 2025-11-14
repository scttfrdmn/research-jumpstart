# Agriculture Data Storage

This directory stores downloaded and processed agricultural datasets. Data persists between SageMaker Studio Lab sessions!

## Directory Structure

```
data/
├── raw/                          # Original downloaded files
│   ├── sentinel2/               # Sentinel-2 imagery (~6GB)
│   │   ├── T10TGK_20230401.tif
│   │   ├── T10TGK_20230406.tif
│   │   └── ...
│   ├── landsat8/                # Landsat-8 imagery (~2GB)
│   ├── modis/                   # MODIS vegetation products (~500MB)
│   ├── weather/                 # Weather data (~100MB)
│   │   ├── temperature.nc
│   │   ├── precipitation.nc
│   │   └── gdd.csv
│   └── soil/                    # Soil data (~50MB)
│
└── processed/                   # Cleaned and processed files
    ├── features/               # Extracted features
    │   ├── ndvi_timeseries.csv
    │   ├── evi_timeseries.csv
    │   └── weather_features.csv
    ├── training/               # Training datasets
    │   ├── X_train.npy
    │   ├── y_train.npy
    │   ├── X_val.npy
    │   └── y_val.npy
    └── predictions/            # Model predictions
```

## Datasets

### Sentinel-2 (ESA Copernicus)
- **Source:** AWS S3 Public Dataset
- **URL:** s3://sentinel-s2-l2a/
- **Resolution:** 10-20m
- **Bands:** B2 (Blue), B3 (Green), B4 (Red), B8 (NIR), B11/B12 (SWIR)
- **Temporal:** 5-day revisit, 6-month time series
- **Size:** ~6GB (50km² AOI)
- **Format:** GeoTIFF/JP2

### Landsat-8 (USGS)
- **Source:** AWS S3 Public Dataset
- **URL:** s3://landsat-pds/
- **Resolution:** 30m
- **Bands:** RGB, NIR, SWIR, Thermal
- **Temporal:** 16-day revisit
- **Size:** ~2GB
- **Format:** GeoTIFF

### MODIS (NASA)
- **Source:** NASA EarthData
- **Product:** MOD13Q1 (Vegetation Indices)
- **Resolution:** 250m
- **Variables:** NDVI, EVI, quality flags
- **Temporal:** 16-day composite
- **Size:** ~500MB
- **Format:** HDF

### Weather Data (NOAA/PRISM)
- **Source:** NOAA Global Summary of the Day
- **Variables:** Daily temp (min/max), precipitation, GDD
- **Period:** Growing season (6 months)
- **Size:** ~100MB
- **Format:** NetCDF/CSV

### Soil Data (SSURGO)
- **Source:** USDA NRCS Soil Survey
- **Variables:** Texture, organic matter, pH, AWC
- **Resolution:** Field-level polygons
- **Size:** ~50MB
- **Format:** Shapefile/GeoJSON

## Usage

Data is automatically downloaded and cached on first use:

```python
from src.data_utils import load_sentinel2, load_weather

# First run: downloads and caches (~45-60 min)
s2_data = load_sentinel2(
    aoi='field_001',
    date_range='2023-04-01/2023-10-01'
)

# Subsequent runs: uses cache (instant!)
s2_data = load_sentinel2(
    aoi='field_001',
    date_range='2023-04-01/2023-10-01'
)

# Force re-download
s2_data = load_sentinel2(
    aoi='field_001',
    date_range='2023-04-01/2023-10-01',
    force_download=True
)
```

## Storage Management

Check current usage:
```bash
du -sh data/
du -sh data/raw/sentinel2/
du -sh data/processed/
```

Clean old files:
```bash
# Remove old sentinel-2 tiles
rm -rf data/raw/sentinel2/.old_*

# Remove intermediate processing files
rm -rf data/processed/temp_*

# Keep only final models
rm -rf ../saved_models/*_epoch_[0-9].h5
```

## Persistence

- **Persistent:** This directory survives Studio Lab session restarts
- **15GB Limit:** Studio Lab provides 15GB persistent storage
- **Shared:** All notebooks in this project share this data directory
- **.gitignore:** Data files excluded from version control

## Data Processing Pipeline

1. **Raw Download:** Sentinel-2, Landsat, MODIS, weather
2. **Preprocessing:** Cloud masking, atmospheric correction
3. **Co-registration:** Align all sensors to common grid
4. **Feature Extraction:** Calculate vegetation indices
5. **Temporal Aggregation:** Create time series features
6. **Training Split:** Generate train/validation sets

## Notes

- Data stored in standard formats (GeoTIFF, NetCDF, CSV)
- Raw files preserved for reproducibility
- Processed files optimized for ML workflows
- Large arrays stored in NumPy binary format
- Cloud-optimized GeoTIFF (COG) for faster access
