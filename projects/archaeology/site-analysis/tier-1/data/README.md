# Data Directory

This directory stores archaeological datasets used in the analysis.

## Structure

```
data/
├── raw/                    # Original downloaded datasets
│   ├── artifacts/         # Artifact imagery by site
│   ├── lidar/             # LiDAR terrain scans
│   └── geophysical/       # GPR and magnetometry data
├── processed/             # Preprocessed and aligned data
│   ├── artifacts/         # Extracted features, classifications
│   ├── terrain/           # Processed LiDAR products
│   └── geophysical/       # Processed survey data
└── README.md              # This file
```

## Data Sources

### Artifact Imagery
- **Source**: Archaeological databases and museum collections
- **Format**: JPEG/PNG images
- **Size**: ~5GB
- **Content**: High-resolution photographs of excavated artifacts

### LiDAR Data
- **Source**: Airborne laser scanning surveys
- **Format**: GeoTIFF, LAS/LAZ point clouds
- **Size**: ~3GB
- **Content**: High-resolution terrain elevation models

### Geophysical Surveys
- **Source**: Ground-penetrating radar and magnetometry surveys
- **Format**: Binary/ASCII grids, SEG-Y
- **Size**: ~2GB
- **Content**: Subsurface anomaly data

## Data Management

### Caching
Processed data is cached to avoid reprocessing:
- Artifact features: `processed/artifacts/site_X_features.npy`
- Terrain metrics: `processed/terrain/site_X_metrics.npz`
- Geophysical features: `processed/geophysical/site_X_anomalies.npy`

### Storage Guidelines
- Keep raw data unchanged in `raw/` directory
- Store all intermediate results in `processed/`
- Use compression for large arrays (.npz format)
- Include metadata files (.json) alongside data files

### Cleaning Up
To free space:
```bash
# Remove processed data (can be regenerated)
rm -rf processed/*

# Remove raw data (must re-download)
rm -rf raw/*
```

## Data Not Included
This directory is in .gitignore. Download data using:
```python
from src.data_utils import download_archaeological_data
download_archaeological_data('all')
```

Or run the first notebook:
- `notebooks/01_data_preparation.ipynb`
