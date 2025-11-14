# Data Directory

This directory stores cached multi-sensor environmental data for persistent access across Studio Lab sessions.

## Structure

```
data/
├── optical/              # Landsat 8 and Sentinel-2 imagery (~3GB)
│   ├── landsat/
│   │   ├── LC08_*.TIF
│   │   └── metadata.json
│   └── sentinel2/
│       ├── S2A_*.jp2
│       └── metadata.json
│
├── sar/                  # Sentinel-1 SAR data (~4GB)
│   ├── VV_*.tif
│   ├── VH_*.tif
│   └── metadata.json
│
├── lidar/                # LiDAR elevation data (~3GB)
│   ├── elevation.tif
│   ├── intensity.tif
│   ├── canopy_height.tif
│   └── metadata.json
│
└── processed/            # Co-registered and preprocessed data
    ├── aligned_stack_2019.npz
    ├── aligned_stack_2020.npz
    └── ...
```

## Data Not Included in Git

All data files are excluded from version control via `.gitignore`:
- Large raster files (`.tif`, `.jp2`, `.npz`)
- Only metadata files are tracked

## Download Instructions

Run `01_data_preparation.ipynb` to automatically download and cache data:

```python
from src.data_utils import download_multisensor_data

# Downloads ~10GB (one-time, 60 minutes)
download_multisensor_data(
    region='coconino_forest',
    date_range=('2019-01-01', '2024-12-31'),
    output_dir='data/'
)
```

## Data Sources

- **Landsat 8**: AWS Open Data - `s3://landsat-pds/`
- **Sentinel-2**: AWS Open Data - `s3://sentinel-2-l2a/`
- **Sentinel-1**: AWS Open Data - `s3://sentinel-1-l1c/`
- **LiDAR**: OpenTopography - Downloaded via API

## Persistence

Data persists across Studio Lab sessions in your 15GB home directory. No need to re-download!

---

Generated with [Research Jumpstart](https://github.com/research-jumpstart/research-jumpstart)
