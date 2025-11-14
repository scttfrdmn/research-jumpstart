# Ocean and Marine Ecosystem Analysis at Scale

**Tier 1 Flagship Project**

Large-scale ocean data analysis with satellite observations, Argo floats, species tracking, and marine habitat monitoring on AWS.

## Quick Start

The main implementation is in the [`unified-studio/`](unified-studio/) directory.

**[→ View Full Documentation](unified-studio/README.md)**

## Features

- **Satellite ocean observations:** Sea surface temperature, chlorophyll, ocean color from MODIS/VIIRS/Sentinel-3
- **Marine heat wave detection:** Hobday et al. algorithm, track duration/intensity/cumulative impact
- **Argo float analysis:** 4,000+ profiling floats, temperature/salinity profiles, ocean heat content, TEOS-10
- **Coral reef monitoring:** Satellite imagery classification, NOAA CoralTemp bleaching alerts (DHW)
- **Species distribution modeling:** MaxEnt, Random Forest SDMs with environmental predictors
- **Ocean current analysis:** OSCAR/HYCOM visualization, Lagrangian particle tracking, MPA connectivity

## Cost Estimate

**Single research cruise:** $500-1,500
**Regional study (1,000 km²):** $2,000-5,000/month
**Basin-scale (Pacific/Atlantic):** $8,000-20,000/month
**Global ocean monitoring:** $30,000-75,000/month

## Technologies

- **Data Processing:** xarray, netCDF4, Dask for distributed computing
- **Oceanography:** gsw (TEOS-10), cartopy, cmocean colormaps
- **ML:** scikit-learn, Random Forest, MaxEnt for species distribution
- **GIS:** Rasterio, GeoPandas for coral reef mapping
- **AWS:** S3 (netCDF, Zarr), Batch, Lambda, SageMaker, Timestream (buoy data), Athena
- **Data Sources:** NASA Ocean Color, NOAA CoralTemp, Copernicus Marine, Argo, OBIS, GBIF, ERDDAP

## Applications

1. **Marine heat waves:** Detect and track ocean warming events (e.g., 2014-2016 Pacific Blob)
2. **Ocean warming trends:** Argo float analysis showing 0-2000m heat content increase
3. **Coral bleaching:** Satellite monitoring with DHW calculations, 80-90% habitat accuracy
4. **Species distribution:** Model habitat suitability under climate scenarios
5. **Particle tracking:** Oil spill trajectories, MPA connectivity, larval dispersal

## Quick Links

- [Getting Started Guide](unified-studio/README.md#getting-started)
- [Marine Heat Wave Detection](unified-studio/README.md#1-marine-heat-wave-detection-and-analysis)
- [Argo Float Analysis](unified-studio/README.md#2-argo-float-analysis-for-ocean-heat-content)
- [Coral Reef Monitoring](unified-studio/README.md#3-coral-reef-health-monitoring)
