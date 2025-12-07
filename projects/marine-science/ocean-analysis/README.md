# Ocean Analysis at Scale

Large-scale marine science research using multi-sensor ocean monitoring for species classification, marine heat wave detection, coral reef monitoring, and ocean health assessment on cloud infrastructure.

## Quick Start by Tier

**New here?** Start with tier-0 (60-90 min, free) to learn ocean species classification.

### 🟢 Tier 0: Ocean Species Classification (60-90 min, FREE)
**[Launch tier-0 project →](tier-0/)**

Train a CNN to classify marine species from underwater imagery:
- ✅ Real underwater imagery (~1.5GB, 50K images from NOAA Fisheries, Kaggle Plankton datasets)
- ✅ CNN species classifier (ResNet/MobileNet transfer learning)
- ✅ 10-15 common species (copepods, diatoms, fish larvae, jellyfish)
- ✅ Data augmentation and class balancing
- ✅ Grad-CAM visualization (model attention maps)
- ✅ Complete in 60-90 minutes
- ✅ No AWS account needed (Colab or Studio Lab)

**Platform**: Google Colab or SageMaker Studio Lab
**Cost**: $0

[View tier-0 README →](tier-0/README.md) | [Open in Colab →](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/marine-science/ocean-analysis/tier-0/ocean-species-classification.ipynb)

---

### 🟡 Tier 1: Multi-Sensor Ocean Monitoring (4-8 hours, FREE)
**[Launch tier-1 project →](tier-1/)**

Comprehensive ocean monitoring with sensor fusion:
- ✅ 10GB multi-sensor data (satellite ocean color, Argo floats, acoustic sensors)
- ✅ Ensemble models (CNN for imagery + LSTM for time series)
- ✅ Sea surface temperature and chlorophyll analysis
- ✅ Multi-modal data fusion techniques
- ✅ Persistent storage and checkpoints (Studio Lab)
- ✅ Still free, no AWS account

**Platform**: SageMaker Studio Lab
**Cost**: $0

[View tier-1 README →](tier-1/README.md)

---

### 🟠 Tier 2: Production Ocean Monitoring (2-3 days, $2K-5K per study)
**[Launch tier-2 project →](tier-2/)**

Research-grade ocean monitoring infrastructure:
- ✅ CloudFormation one-click deployment
- ✅ 100GB+ ocean data on S3 (satellite obs, MODIS/VIIRS/Sentinel-3)
- ✅ Marine heat wave detection (Hobday et al. algorithm, track duration/intensity)
- ✅ Argo float analysis (4,000+ profiling floats, temperature/salinity, TEOS-10)
- ✅ Coral reef monitoring (satellite classification, NOAA CoralTemp bleaching alerts)
- ✅ Species distribution modeling (MaxEnt, Random Forest SDMs)
- ✅ Publication-ready ocean health assessments

**Platform**: AWS with CloudFormation
**Cost**: $500-1,500 per research cruise, $2K-5K/month for regional study

[View tier-2 README →](tier-2/README.md)

---

### 🔴 Tier 3: Global Ocean Platform (Ongoing, $8K-75K/month)
**[Launch tier-3 project →](tier-3/)**

Production platform for oceanographic institutions:
- ✅ Basin-scale or global ocean monitoring (Pacific/Atlantic basins)
- ✅ Real-time ocean current analysis (OSCAR/HYCOM, Lagrangian particle tracking)
- ✅ MPA connectivity analysis and larval dispersal modeling
- ✅ Integration with autonomous vehicles (gliders, profiling floats)
- ✅ Multi-region comparative ocean studies
- ✅ AI-assisted interpretation (Amazon Bedrock for ocean analysis)
- ✅ Team collaboration with versioned datasets

**Platform**: AWS multi-account with enterprise support
**Cost**: $8K-20K/month (basin-scale), $30K-75K/month (global)

[View tier-3 README →](tier-3/README.md)

---

## What You'll Learn

Across all tiers, this project teaches:
- CNN image classification for marine species identification
- Multi-sensor data fusion (satellite, in-situ, acoustic)
- Marine heat wave detection algorithms (Hobday et al.)
- Argo float analysis and ocean heat content calculation
- Coral reef monitoring with satellite imagery
- Species distribution modeling (MaxEnt, Random Forest)
- Distributed oceanographic analysis on cloud infrastructure

## Technologies & Tools

- **Data sources**: NASA Ocean Color (MODIS, VIIRS), NOAA CoralTemp, Copernicus Marine, Argo floats, OBIS (Ocean Biodiversity), GBIF, ERDDAP servers
- **Languages**: Python 3.9+
- **Core libraries**: pandas, numpy, scipy, scikit-learn
- **Oceanography tools**: xarray, netCDF4, Dask (distributed), gsw (TEOS-10 equations), cartopy, cmocean colormaps
- **ML frameworks**: TensorFlow/PyTorch (CNNs), scikit-learn (Random Forest, MaxEnt)
- **GIS**: Rasterio, GeoPandas (coral reef mapping)
- **Cloud services** (tier 2+): S3 (netCDF, Zarr), Batch (distributed processing), Lambda, SageMaker (ML training), Timestream (buoy data), Athena

## Project Structure

```
ocean-analysis/
├── tier-0/              # Species classification (60-90 min, FREE)
│   ├── ocean-species-classification.ipynb
│   ├── README.md
│   └── requirements.txt
├── tier-1/              # Multi-sensor (4-8 hours, FREE)
│   ├── notebooks/
│   ├── src/
│   ├── data/
│   ├── environment.yml
│   └── README.md
├── tier-2/              # Production (2-3 days, $2K-5K/study)
│   ├── cloudformation/
│   ├── notebooks/
│   ├── src/
│   ├── tests/
│   └── README.md
└── tier-3/              # Global platform (ongoing, $8K-75K/mo)
    ├── cloudformation/
    ├── notebooks/
    ├── src/
    ├── infrastructure/
    └── README.md
```

## Progression Path

```
Tier 0           → Tier 1          → Tier 2            → Tier 3
Species Class      Multi-Sensor       Regional Study      Global Monitoring
1.5GB imagery      10GB multi-modal   100GB+ satellite    TB-scale, real-time
60-90 min          4-8 hours          2-3 days            Ongoing
FREE               FREE               $2K-5K/study        $8K-75K/mo
```

You can:
- ✅ Skip tiers if you have AWS experience and large-scale ocean monitoring needs
- ✅ Stop at any tier - tier-1 is great for theses, tier-2 for research cruises
- ✅ Mix and match - use tier-0 for prototyping, tier-2 for publications

[Understanding tiers →](../../../docs/projects/tiers.md)

## Marine Science Applications

- **Marine heat wave detection**: Track ocean warming events (e.g., 2014-2016 Pacific Blob) with duration/intensity metrics
- **Ocean warming trends**: Analyze Argo float data showing 0-2000m heat content increases
- **Coral bleaching monitoring**: Satellite-based DHW (Degree Heating Week) calculations, 80-90% habitat accuracy
- **Species distribution modeling**: Model habitat suitability under climate change scenarios
- **Particle tracking**: Oil spill trajectories, MPA connectivity, larval dispersal pathways
- **Biodiversity assessment**: Automated species identification from underwater imagery

## Related Projects

- **[Climate Science - Ensemble Analysis](../../climate-science/ensemble-analysis/)** - Ocean-atmosphere coupling
- **[Astronomy - Sky Survey](../../astronomy/sky-survey/)** - Similar large-scale image classification
- **[Agriculture - Precision Agriculture](../../agriculture/precision-agriculture/)** - Satellite imagery analysis methods

## Common Use Cases

- **Marine biologists**: Biodiversity surveys, species distribution mapping
- **Oceanographers**: Ocean heat content, circulation patterns, climate impacts
- **Coral reef scientists**: Bleaching monitoring, reef health assessment
- **Fisheries scientists**: Habitat modeling, stock assessment support
- **Climate researchers**: Ocean warming analysis, marine heat waves
- **Conservation planners**: MPA design, connectivity analysis

## Cost Estimates

**Tier 2 Production (Regional Study - 1,000 km²)**:
- **S3 storage** (100GB satellite data, Argo profiles): $2.30/month
- **Lambda** (preprocessing, feature extraction): $30/month
- **SageMaker** (species classification models): ml.p3.2xlarge, 8 hours/month = $80/month
- **AWS Batch** (distributed analysis, marine heat waves): $50/month
- **Timestream** (buoy time series data): $20/month
- **Total**: $2,000-5,000/month for continuous regional monitoring

**Scaling**:
- Research cruise (single deployment): $500-1,500
- Basin-scale (Pacific/Atlantic): $8,000-20,000/month
- Global ocean monitoring: $30,000-75,000/month

**Optimization tips**:
- Use NOAA/Copernicus public data on AWS (no egress fees)
- Cache processed satellite imagery to avoid reprocessing
- Use netCDF4 compression and Zarr format for efficient I/O
- Process Argo profiles in batches for parallelization

## Support

- **Questions**: [GitHub Discussions](https://github.com/scttfrdmn/research-jumpstart/discussions)
- **Issues**: [GitHub Issues](https://github.com/scttfrdmn/research-jumpstart/issues)
- **Office Hours**: [Every Tuesday](../../../docs/community/office-hours.md)

## Citation

If you use this project in your research, please cite:

```bibtex
@software{research_jumpstart_ocean_analysis,
  title = {Ocean Analysis at Scale: Research Jumpstart},
  author = {Research Jumpstart Community},
  year = {2025},
  url = {https://github.com/scttfrdmn/research-jumpstart},
  note = {Accessed: [date]}
}
```

Also cite the appropriate data sources:
- **NASA Ocean Color**: https://oceancolor.gsfc.nasa.gov/
- **Argo**: http://www.argo.ucsd.edu/
- **NOAA CoralTemp**: https://coralreefwatch.noaa.gov/

## License

Apache 2.0 - See [LICENSE](../../../LICENSE) for details.

---

*Part of [Research Jumpstart](https://github.com/scttfrdmn/research-jumpstart) - Pre-built research workflows for cloud computing*
