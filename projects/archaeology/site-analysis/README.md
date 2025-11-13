# Archaeological Site Analysis at Scale

**Tier 1 Flagship Project**

Large-scale archaeological site discovery and analysis with LiDAR, ML artifact classification, and spatial analysis on AWS.

## Quick Start

The main implementation is in the [`unified-studio/`](unified-studio/) directory.

**[→ View Full Documentation](unified-studio/README.md)**

## Features

- **LiDAR site discovery:** Process airborne/terrestrial point clouds to detect archaeological features (mounds, structures, earthworks)
- **Artifact classification:** Deep learning (ResNet, EfficientNet) for pottery, lithics, architecture (85-95% accuracy)
- **Settlement patterns:** Ripley's K, nearest neighbor, Voronoi, viewshed analysis, least-cost paths
- **Radiocarbon dating:** Bayesian chronological modeling with OxCal, phase analysis
- **3D photogrammetry:** OpenDroneMap for Structure from Motion, mesh reconstruction
- **Trade networks:** NetworkX analysis of material flows, geochemical sourcing

## Cost Estimate

**Single site:** $500-800
**Regional (100 sites):** $2,000-4,000/month
**Large-scale (1,000+ sites):** $8,000-15,000/month
**Continental:** $25,000-50,000/month

## Technologies

- **LiDAR:** PDAL, LAStools, CloudCompare for point cloud processing
- **GIS:** PostGIS, QGIS, GeoPandas, Rasterio
- **ML:** PyTorch, SageMaker, ResNet50, EfficientNet
- **Photogrammetry:** OpenDroneMap, Open3D, MeshLab
- **AWS:** S3, EC2 (GPU), Batch, PostGIS on RDS, SageMaker, Lambda, Glue, Athena
- **Databases:** Open Context, tDAR, DINAA, IntChron radiocarbon

## Applications

1. **LiDAR discovery:** Detect Maya sites in jungle, European hillforts
2. **Artifact classification:** Automate pottery/lithic typology
3. **Settlement analysis:** Test clustering, environmental correlates, viewsheds
4. **Chronology:** Bayesian radiocarbon modeling, synchrony testing
5. **3D reconstruction:** Virtual museums, metric analysis from photogrammetry

## Quick Links

- [Getting Started Guide](unified-studio/README.md#getting-started)
- [LiDAR Site Discovery](unified-studio/README.md#1-lidar-based-site-discovery)
- [Artifact Classification](unified-studio/README.md#2-artifact-classification-with-deep-learning)
- [Settlement Patterns](unified-studio/README.md#3-settlement-pattern-analysis)
