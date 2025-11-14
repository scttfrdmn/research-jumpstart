# Multi-Sensor Ocean Monitoring

**Flagship Project** ⭐ | **Difficulty**: 🟢 Beginner | **Time**: ⏱️ 60-90 min (Tier 0) | ⏱️⏱️ 4-8 hours (Tier 1)

Perform comprehensive ocean monitoring using multi-sensor data fusion without managing terabytes of oceanographic data. Perfect introduction to cloud-based marine science research.

---

## Quick Start Options

### Tier 0: Ocean Species Classification (60-90 min, Free)
**Perfect for:** First-time users, quick demos, species identification learning

Train a CNN to classify marine species from underwater imagery:
- 1.5GB plankton and fish imagery
- 60-75 min training on GPU
- Runs on Google Colab (free)
- **[→ Start Tier 0](tier-0/README.md)**

### Tier 1: Multi-Sensor Ocean Monitoring (4-8 hours, Free)
**Perfect for:** Learning sensor fusion, ensemble methods, persistent workflows

Integrate satellite, Argo float, and acoustic data for comprehensive ocean monitoring:
- 10GB multi-sensor data (satellite ocean color, Argo floats, acoustic)
- 5-6 hours ensemble training
- Requires SageMaker Studio Lab (free, persistent storage)
- **[→ Start Tier 1](tier-1/README.md)**

### Production: Full-Scale Ocean Analysis
**Perfect for:** Research publications, operational monitoring, multi-region analysis

The main production implementation is in the [`unified-studio/`](unified-studio/) directory.

**[→ View Full Documentation](unified-studio/README.md)**

---

## Tier Comparison

| Feature | Tier 0 | Tier 1 | Production |
|---------|--------|--------|------------|
| **Platform** | Colab/Studio Lab | Studio Lab | Unified Studio |
| **Time** | 60-90 min | 4-8 hours | Varies |
| **Cost** | $0 | $0 | $20-40/analysis |
| **Data** | 1.5GB imagery | 10GB multi-sensor | Unlimited S3 |
| **Storage** | None (re-download) | 15GB persistent | Unlimited S3 |
| **Focus** | Species classification | Sensor fusion ensemble | Full ocean monitoring |
| **Sensors** | Image only | 3 types | 10+ types |
| **ML Models** | Single CNN | Ensemble (CNN+LSTM) | Production ensemble |
| **Use Case** | Learning, demos | Research prototyping | Publications, ops |

---

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
