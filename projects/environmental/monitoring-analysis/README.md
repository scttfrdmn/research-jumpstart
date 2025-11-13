# Environmental Monitoring at Scale

**Tier 1 Flagship Project**

Analyze environmental data from satellites, sensors, and monitoring networks using remote sensing and machine learning on AWS.

## Quick Start

The main implementation is in the [`unified-studio/`](unified-studio/) directory.

**[→ View Full Documentation](unified-studio/README.md)**

## Features

- **Satellite data:** Landsat (40+ years), Sentinel-2 (10m), MODIS (daily global)
- **Petabyte scale:** Process years of Earth observation data
- **Applications:** Deforestation, air quality, land cover, climate extremes
- **ML models:** Land cover classification, deforestation detection
- **Sensor networks:** Air quality (EPA, PurpleAir), weather stations

## Cost Estimate

**$1,200-2,500** for regional study (1 year analysis)
**$3,200-4,300/month** for global monitoring

## Technologies

- **Satellites:** Landsat-8/9, Sentinel-2, MODIS, Sentinel-5P
- **Processing:** GDAL, Rasterio, Xarray, Google Earth Engine
- **ML:** U-Net segmentation, Random Forest classification
- **AWS:** S3 (Open Data), Batch, SageMaker, Ground Station
- **Time series:** NDVI trends, phenology, anomaly detection

## Applications

1. **Deforestation:** Track forest loss with Landsat/Sentinel
2. **Air quality:** Monitor PM2.5, ozone, NO2 from sensors
3. **Land cover:** Classify agriculture, urban, water bodies
4. **Climate:** Temperature trends, extreme events
5. **Ocean:** SST, chlorophyll, marine heat waves

## Quick Links

- [Getting Started Guide](unified-studio/README.md#getting-started)
- [Deforestation Detection](unified-studio/README.md#1-deforestation-detection)
- [Land Cover Classification](unified-studio/README.md#2-land-cover-classification)
- [Air Quality Monitoring](unified-studio/README.md#3-air-quality-monitoring-and-prediction)
