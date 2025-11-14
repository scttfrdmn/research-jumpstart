# Disease Surveillance Data Storage

This directory stores downloaded and processed disease surveillance datasets. Data persists between SageMaker Studio Lab sessions!

## Directory Structure

```
data/
├── raw/                    # Original downloaded files
│   ├── cdc_ili/           # CDC FluView ILI data
│   ├── cdc_covid/         # CDC COVID-19 data
│   ├── cdc_rsv/           # CDC RSV surveillance
│   ├── who_data/          # WHO disease outbreak data
│   └── state_health/      # State health department data
│
└── processed/              # Cleaned and processed files
    ├── ili_by_state.csv
    ├── covid_by_state.csv
    ├── rsv_by_state.csv
    ├── multi_disease_panel.csv
    └── spatiotemporal_features.csv
```

## Datasets

### Influenza-Like Illness (CDC ILINet)
- **Source:** CDC FluView
- **URL:** https://www.cdc.gov/flu/weekly/
- **Period:** 2015-2024
- **Size:** ~2GB
- **Variables:** ILI rate, total patients, age groups, geographic regions
- **Granularity:** Weekly by state/region

### COVID-19 (CDC COVID Data Tracker)
- **Source:** CDC COVID-19 Case Surveillance
- **URL:** https://covid.cdc.gov/covid-data-tracker/
- **Period:** 2020-2024
- **Size:** ~3GB
- **Variables:** Cases, deaths, hospitalizations, test positivity, vaccinations
- **Granularity:** Daily/Weekly by state/county

### Respiratory Syncytial Virus (CDC NREVSS)
- **Source:** CDC National Respiratory and Enteric Virus Surveillance System
- **Period:** 2015-2024
- **Size:** ~1.5GB
- **Variables:** RSV detections, test volumes, age groups
- **Granularity:** Weekly by state

### WHO Global Surveillance
- **Source:** WHO Disease Outbreak News
- **URL:** https://www.who.int/emergencies/disease-outbreak-news
- **Period:** 2015-2024
- **Size:** ~2GB
- **Variables:** Multiple diseases, case counts, geographic distribution
- **Granularity:** Weekly by country/region

### Other Diseases
- **Norovirus:** State health department reports (~500MB)
- **Pertussis:** CDC NNDSS (~300MB)
- **Measles:** CDC NNDSS (~200MB)

## Usage

Data is automatically downloaded and cached on first use:

```python
from src.data_utils import load_ili_data, load_covid_data, load_rsv_data

# First run: downloads and caches
ili_df = load_ili_data()  # Downloads ~2GB

# Subsequent runs: uses cache
ili_df = load_ili_data()  # Instant!

# Force re-download
ili_df = load_ili_data(force_download=True)

# Load multiple diseases
multi_df = load_multi_disease_panel(
    diseases=['ILI', 'COVID-19', 'RSV'],
    start_date='2020-01-01',
    end_date='2024-01-01'
)
```

## Feature Engineering

Processed data includes spatiotemporal features:

```python
from src.feature_engineering import create_spatiotemporal_features

features = create_spatiotemporal_features(
    data=ili_df,
    lag_weeks=[1, 2, 3, 4],
    rolling_windows=[2, 4, 8],
    include_geographic_neighbors=True,
    include_seasonal_decomp=True
)
```

## Storage Management

Check current usage:
```bash
du -sh data/
du -sh data/raw/
du -sh data/processed/
```

Clean old files:
```bash
rm -rf data/raw/*/old_*.csv
rm -rf data/processed/*.backup
```

Free up space (remove oldest raw data, keep processed):
```bash
# Keep only last 2 years of raw data
find data/raw -name "*.csv" -mtime +730 -delete
```

## Persistence

- **Persistent:** This directory survives Studio Lab session restarts
- **15GB Limit:** Studio Lab provides 15GB persistent storage
- **Shared:** All notebooks in this project share this data directory
- **Git:** Data directory is in .gitignore (not version controlled)

## Data Privacy & Compliance

- All data sources are publicly available
- Data is aggregated (no individual-level data)
- CDC and WHO data have no usage restrictions for research
- State data: Check individual state data use agreements
- No PHI (Protected Health Information) is stored

## Notes

- Data is stored in CSV/Parquet format for easy inspection
- Raw files preserved for reproducibility
- Processed files optimized for analysis
- Spatial shapefiles cached for geographic analysis
- Data quality checks logged in `data/logs/`
