# Ocean Monitoring Data Storage

This directory contains cached ocean sensor data for persistent access.

## Directory Structure

```
data/
├── satellite/          # MODIS-Aqua ocean color data (~4GB)
│   ├── chlorophyll/   # Chlorophyll-a concentration
│   └── sst/           # Sea surface temperature
├── argo/              # Argo float profiles (~3GB)
│   ├── temperature/   # Temperature profiles
│   └── salinity/      # Salinity profiles
├── acoustic/          # Acoustic sensor data (~3GB)
│   ├── marine_mammals/
│   └── fish_schools/
└── processed/         # Fused and processed data
    ├── aligned/       # Spatiotemporally aligned
    └── quality_controlled/
```

## Data Sources

### Satellite Ocean Color (MODIS-Aqua)
- **Variable:** Chlorophyll-a, SST
- **Resolution:** 4km spatial, daily temporal
- **Period:** 2015-2024
- **Format:** NetCDF (.nc)
- **Source:** NASA Ocean Color Web

### Argo Float Profiles
- **Variables:** Temperature, salinity, pressure
- **Resolution:** Point measurements, ~10-day cycles
- **Period:** 2015-2024
- **Format:** NetCDF (.nc)
- **Source:** Argo Global Data Assembly Center

### Acoustic Sensors
- **Variables:** Sound intensity, frequency spectra
- **Resolution:** Continuous recordings, station-based
- **Period:** 2015-2024
- **Format:** WAV files + CSV metadata
- **Source:** Simulated for educational purposes

## Usage

Data is downloaded once and cached for all future sessions:

```python
from src.ocean_data import load_satellite_data, load_argo_floats

# Loads from cache if available
satellite = load_satellite_data(
    variable='chlorophyll',
    region={'lat': (20, 40), 'lon': (-140, -120)},
    dates=('2020-01-01', '2024-12-31')
)
```

## Storage Management

Monitor disk usage:
```bash
du -sh data/*
```

Clean old cached files if needed:
```bash
# Remove files older than 30 days
find data/ -mtime +30 -type f -delete
```

## Data Download Times

First-time download estimates:
- Satellite data: ~20 minutes (4GB)
- Argo floats: ~15 minutes (3GB)
- Acoustic data: ~15 minutes (3GB)
- **Total: ~50-60 minutes**

Subsequent sessions: **Instant** (loaded from cache)

## Notes

- All data directories are in `.gitignore`
- Data persists between Studio Lab sessions
- Maximum storage: 15GB total in Studio Lab
- Consider cleaning old analyses periodically
