# Data Directory

This directory stores downloaded and processed grid data. It is gitignored to prevent committing large datasets.

## Structure

```
data/
├── raw/                    # Downloaded datasets (~10GB)
│   ├── load/              # Smart meter load data
│   ├── solar/             # Solar generation data
│   ├── wind/              # Wind power data
│   ├── weather/           # Weather forecasts
│   └── storage/           # Battery storage data
│
└── processed/             # Processed data ready for training
    ├── load_features.parquet
    ├── solar_features.parquet
    ├── wind_features.parquet
    └── grid_stability.parquet
```

## Data Sources

- **Load data:** ISO/RTO public datasets
- **Solar data:** NREL PVDAQ, WIND Toolkit
- **Wind data:** NREL WIND Toolkit
- **Weather data:** NOAA GFS/HRRR
- **Storage data:** Synthetic utility SCADA

## Note

Data files are downloaded by `01_data_acquisition.ipynb` and persist between Studio Lab sessions.
