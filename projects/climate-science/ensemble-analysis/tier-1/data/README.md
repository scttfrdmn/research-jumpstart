# Climate Data Storage

This directory stores downloaded and processed climate datasets. Data persists between SageMaker Studio Lab sessions!

## Directory Structure

```
data/
├── raw/                    # Original downloaded files
│   ├── gistemp_global.csv
│   ├── co2_mlo.txt
│   └── sea_level.csv
│
└── processed/              # Cleaned and processed files
    ├── temperature_monthly.csv
    ├── co2_monthly.csv
    └── sea_level_monthly.csv
```

## Datasets

### Temperature (NOAA GISTEMP)
- **Source:** NASA Goddard Institute for Space Studies
- **URL:** https://data.giss.nasa.gov/gistemp/
- **Period:** 1880-2024
- **Size:** ~500KB
- **Variables:** Monthly temperature anomalies (°C)

### CO2 (Mauna Loa)
- **Source:** NOAA Global Monitoring Laboratory
- **URL:** https://gml.noaa.gov/ccgg/trends/
- **Period:** 1958-2024
- **Size:** ~100KB
- **Variables:** Atmospheric CO2 concentration (ppm)

### Sea Level (Satellite Altimetry)
- **Source:** NOAA Laboratory for Satellite Altimetry
- **Period:** 1993-2024
- **Size:** ~200KB
- **Variables:** Global mean sea level (mm)

## Usage

Data is automatically downloaded and cached on first use:

```python
from src.data_utils import load_temperature_data, load_co2_data

# First run: downloads and caches
temp_df = load_temperature_data()  # Downloads ~500KB

# Subsequent runs: uses cache
temp_df = load_temperature_data()  # Instant!

# Force re-download
temp_df = load_temperature_data(force_download=True)
```

## Storage Management

Check current usage:
```bash
du -sh data/
```

Clean old files:
```bash
rm -rf data/raw/*.old
rm -rf data/processed/*.backup
```

## Persistence

✅ **Persistent:** This directory survives Studio Lab session restarts
✅ **15GB Limit:** Studio Lab provides 15GB persistent storage
✅ **Shared:** All notebooks in this project share this data directory

## Notes

- Data is stored in CSV format for easy inspection
- Raw files preserved for reproducibility
- Processed files optimized for analysis
- .gitignore excludes data/ from version control
