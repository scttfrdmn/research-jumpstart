# Precision Agriculture at Scale

**Tier 1 Flagship Project**

Optimize crop yields and resource use with satellite imagery, IoT sensors, and machine learning on AWS.

## Quick Start

The main implementation is in the [`unified-studio/`](unified-studio/) directory.

**[→ View Full Documentation](unified-studio/README.md)**

## Features

- **Satellite data:** Sentinel-2 (10m), Landsat (30m), MODIS (250m-1km)
- **Crop monitoring:** NDVI, EVI, NDRE, NDMI vegetation indices
- **Yield prediction:** XGBoost models using weather, soil, satellite features
- **Disease detection:** CNN models for early pest/disease identification
- **IoT sensors:** Soil moisture, temperature, EC integrated with AWS IoT Core
- **Prescription maps:** Variable rate application for precision equipment

## Cost Estimate

**$50-100/month** for 1,000 hectare farm
**ROI:** 5-15% yield increase, 10-30% input reduction

## Technologies

- **Satellites:** Sentinel-2, Landsat 8/9, MODIS
- **Processing:** GDAL, Rasterio, Xarray, Planetary Computer
- **ML:** XGBoost, TensorFlow, scikit-learn
- **AWS:** S3 (Open Data), Batch, IoT Core, Timestream, SageMaker, Bedrock
- **Indices:** NDVI, EVI, NDRE, NDMI, GNDVI

## Applications

1. **Crop health:** Monitor stress, disease, nutrient deficiency
2. **Yield prediction:** Forecast production weeks before harvest
3. **Irrigation:** Optimize water use with ET models and soil sensors
4. **Variable rate:** Create prescription maps for fertilizer, seed, pesticide
5. **Carbon:** Measure soil carbon sequestration for credits

## Quick Links

- [Getting Started Guide](unified-studio/README.md#getting-started)
- [Crop Health Monitoring](unified-studio/README.md#1-crop-health-monitoring)
- [Yield Prediction](unified-studio/README.md#2-yield-prediction)
- [Disease Detection](unified-studio/README.md#3-disease-detection)
