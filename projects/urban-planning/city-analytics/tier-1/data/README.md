# Urban Planning Data Storage

This directory stores downloaded and processed urban datasets. Data persists between SageMaker Studio Lab sessions!

## Directory Structure

```
data/
├── raw/                          # Original downloaded files
│   ├── imagery/                  # Satellite imagery by city
│   │   ├── austin_2000.tif
│   │   ├── austin_2024.tif
│   │   └── ...
│   ├── mobility/                 # OpenStreetMap and transit data
│   │   ├── austin_roads.geojson
│   │   ├── austin_transit.gtfs
│   │   └── ...
│   └── demographics/             # Census data
│       ├── austin_census.csv
│       └── ...
│
└── processed/                    # Cleaned and processed files
    ├── austin_2024_imagery.npy
    ├── austin_roads.geojson
    ├── austin_demographics.csv
    └── ...
```

## Datasets

### Satellite Imagery (Landsat 8/9)
- **Source:** USGS Earth Explorer / Google Earth Engine
- **Cities:** Austin, Denver, Phoenix, Portland, Seattle, Charlotte
- **Period:** 2000-2024 (annual composites)
- **Resolution:** 30m pixel resolution
- **Size:** ~6GB total (~1GB per city)
- **Variables:** RGB, NIR, SWIR bands, NDVI, NDBI

### Mobility Data (OpenStreetMap)
- **Source:** OpenStreetMap API / Overpass Turbo
- **Data types:** Roads, bike lanes, pedestrian paths
- **Attributes:** Road type, speed limit, capacity, lanes
- **Size:** ~2GB total
- **Format:** GeoJSON

### Transit Data (GTFS)
- **Source:** Transit agency APIs
- **Data types:** Routes, stops, schedules, frequencies
- **Size:** ~500MB total
- **Format:** GTFS (General Transit Feed Specification)

### Demographics (US Census)
- **Source:** US Census Bureau API
- **Dataset:** American Community Survey (ACS) 5-year estimates
- **Period:** 2010-2024
- **Size:** ~1.5GB total
- **Variables:** Population, median income, employment, education
- **Granularity:** Census tract level

## Usage

Data is automatically downloaded and cached on first use:

```python
from src.data_utils import load_satellite_imagery, load_mobility_data

# First run: downloads and caches
imagery = load_satellite_imagery('austin', 2024)  # Downloads ~1GB

# Subsequent runs: uses cache
imagery = load_satellite_imagery('austin', 2024)  # Instant!

# Force re-download
imagery = load_satellite_imagery('austin', 2024, force_download=True)
```

## Storage Management

Check current usage:
```bash
du -sh data/
du -sh data/raw/imagery/
du -sh data/raw/mobility/
```

Clean old files:
```bash
rm -rf data/raw/imagery/*_old.tif
rm -rf data/processed/*.backup
```

## Persistence

- **Persistent:** This directory survives Studio Lab session restarts
- **15GB Limit:** Studio Lab provides 15GB persistent storage
- **Shared:** All notebooks in this project share this data directory

## Data Sources

### Satellite Imagery
- USGS Earth Explorer: https://earthexplorer.usgs.gov/
- Google Earth Engine: https://earthengine.google.com/

### Mobility Data
- OpenStreetMap: https://www.openstreetmap.org/
- Overpass API: https://overpass-api.de/

### Transit Data
- Transit Feeds: https://transitfeeds.com/
- MobilityData: https://mobilitydata.org/

### Demographics
- Census Bureau API: https://www.census.gov/data/developers.html
- TIGER/Line Shapefiles: https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html

## Notes

- Raw data preserved for reproducibility
- Processed data optimized for analysis
- .gitignore excludes data/ from version control
- Coordinate reference systems standardized to EPSG:4326 (WGS84)
