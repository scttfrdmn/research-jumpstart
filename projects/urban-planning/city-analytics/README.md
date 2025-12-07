# City Analytics at Scale

Large-scale urban analytics using satellite imagery, mobility data, and machine learning. Predict urban growth, analyze transportation patterns, optimize infrastructure planning, and model city dynamics on cloud infrastructure.

## Quick Start by Tier

**New here?** Start with tier-0 (60-90 min, free) to learn urban growth prediction from satellite imagery.

### 🟢 Tier 0: Urban Growth Prediction (60-90 min, FREE)
**[Launch tier-0 project →](tier-0/)**

Predict urban expansion patterns from satellite imagery:
- ✅ Real satellite data (~1.5GB, Landsat 8/9 time series, 2000-2024, 30m resolution)
- ✅ CNN for urban growth prediction (transfer learning with ImageNet)
- ✅ Urban indices (NDBI - Normalized Difference Built-up Index, NDVI)
- ✅ Multi-temporal change detection (20-year urban expansion analysis)
- ✅ Growth forecasts (2025-2035 urban expansion projections)
- ✅ Complete in 60-90 minutes
- ✅ No AWS account needed (Colab or Studio Lab)

**Platform**: Google Colab or SageMaker Studio Lab
**Cost**: $0

[View tier-0 README →](tier-0/README.md) | [Open in Colab →](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/urban-planning/city-analytics/tier-0/urban-growth-prediction.ipynb)

---

### 🟡 Tier 1: Multi-City Urban Analytics (4-8 hours, FREE)
**[Launch tier-1 project →](tier-1/)**

Comprehensive urban analysis across multiple cities:
- ✅ 10GB multi-city data (5-6 metropolitan areas, satellite + mobility)
- ✅ Ensemble models (CNN, Random Forest, Gradient Boosting for growth prediction)
- ✅ Transportation network analysis (OpenStreetMap, traffic patterns)
- ✅ Comparative urban studies across cities
- ✅ Persistent storage for long training runs (Studio Lab)
- ✅ Still free, no AWS account

**Platform**: SageMaker Studio Lab
**Cost**: $0

[View tier-1 README →](tier-1/README.md)

---

### 🟠 Tier 2: Production Urban Analytics (2-3 days, $150-300/month)
**[Launch tier-2 project →](tier-2/)**

Research-grade urban analytics infrastructure:
- ✅ CloudFormation one-click deployment
- ✅ 100GB+ satellite imagery and mobility data on S3
- ✅ Distributed preprocessing with Lambda (imagery, OSM, census data)
- ✅ SageMaker for large-scale urban modeling
- ✅ Real-time growth monitoring with Sentinel-2/Landsat (weekly updates)
- ✅ Transportation optimization with graph analytics
- ✅ Publication-ready maps and visualizations

**Platform**: AWS with CloudFormation
**Cost**: $150-300/month for continuous monitoring

[View tier-2 README →](tier-2/README.md)

---

### 🔴 Tier 3: Enterprise Urban Platform (Ongoing, $3K-8K/month)
**[Launch tier-3 project →](tier-3/)**

Production platform for urban planning agencies:
- ✅ State/nation-wide deployment (50+ cities, 2TB+ imagery)
- ✅ Real-time dashboards for planners and policymakers (QuickSight)
- ✅ Distributed Dask clusters for massive geospatial analysis
- ✅ Integration with city systems (GIS, permitting, zoning databases)
- ✅ Traffic simulation and congestion modeling
- ✅ AI-assisted planning recommendations (Amazon Bedrock)
- ✅ Team collaboration with versioned analyses

**Platform**: AWS multi-account with enterprise support
**Cost**: $3K-8K/month

[View tier-3 README →](tier-3/README.md)

---

## What You'll Learn

Across all tiers, this project teaches:
- Convolutional Neural Networks for satellite imagery analysis
- Multi-temporal change detection and urban growth modeling
- Urban indices (NDBI, NDVI) for built environment analysis
- Transportation network analysis with graph algorithms
- Distributed geospatial processing on cloud infrastructure
- Mobility pattern analysis and traffic optimization

## Technologies & Tools

- **Data sources**: Landsat 8/9, Sentinel-2 (AWS Open Data), OpenStreetMap, Census TIGER, SafeGraph mobility data
- **Languages**: Python 3.9+
- **Core libraries**: pandas, numpy, scipy, scikit-learn
- **Geospatial**: rasterio, geopandas, shapely, GDAL, OSMnx (network analysis)
- **ML frameworks**: TensorFlow/PyTorch (CNNs), scikit-learn, XGBoost
- **Cloud services** (tier 2+): S3 (satellite data), Lambda (preprocessing), SageMaker (training), Batch (distributed analysis), QuickSight (dashboards)

## Project Structure

```
city-analytics/
├── tier-0/              # Urban growth (60-90 min, FREE)
│   ├── urban-growth-prediction.ipynb
│   ├── README.md
│   └── requirements.txt
├── tier-1/              # Multi-city (4-8 hours, FREE)
│   ├── notebooks/
│   ├── src/
│   ├── data/
│   ├── environment.yml
│   └── README.md
├── tier-2/              # Production (2-3 days, $150-300/mo)
│   ├── cloudformation/
│   ├── notebooks/
│   ├── src/
│   ├── tests/
│   └── README.md
└── tier-3/              # Enterprise platform (ongoing, $3K-8K/mo)
    ├── cloudformation/
    ├── notebooks/
    ├── src/
    ├── infrastructure/
    └── README.md
```

## Progression Path

```
Tier 0           → Tier 1          → Tier 2            → Tier 3
Urban Growth       Multi-City         Production          Enterprise
Single city        5-6 cities         100GB+ data         State-wide
60-90 min          4-8 hours          2-3 days            Ongoing
FREE               FREE               $150-300/mo         $3K-8K/mo
```

You can:
- ✅ Skip tiers if you have AWS experience and city-scale analysis needs
- ✅ Stop at any tier - tier-1 is great for academic research, tier-2 for planning agencies
- ✅ Mix and match - use tier-0 for prototyping, tier-2 for operational planning

[Understanding tiers →](../../../docs/projects/tiers.md)

## Urban Planning Applications

- **Urban growth modeling**: Predict expansion patterns 10-20 years ahead with CNN/ensemble models
- **Transportation optimization**: Analyze traffic patterns, optimize routes, simulate interventions
- **Infrastructure planning**: Site selection for schools, hospitals, transit based on growth forecasts
- **Gentrification analysis**: Track neighborhood change through satellite and census data
- **Climate adaptation**: Identify heat island effects, flood risk zones, green space needs
- **Zoning optimization**: Data-driven zoning recommendations based on growth and mobility patterns

## Related Projects

- **[Climate Science - Satellite Imagery](../../climate-science/ensemble-analysis/)** - Similar remote sensing techniques
- **[Agriculture - Precision Agriculture](../../agriculture/precision-agriculture/)** - Satellite imagery analysis
- **[Economics - Time Series Forecasting](../../economics/time-series-forecasting/)** - Growth prediction methods

## Common Use Cases

- **City planning departments**: 20-year comprehensive plans, infrastructure siting
- **Regional agencies**: Multi-city comparative analysis, regional growth coordination
- **Developers**: Market analysis, site selection, feasibility studies
- **Researchers**: Publish urban studies papers, test planning theories
- **Transportation agencies**: Traffic optimization, transit planning, congestion reduction
- **Climate adaptation**: Urban heat island mitigation, green infrastructure planning

## Cost Estimates

**Tier 2 Production (Single City, Continuous Monitoring)**:
- **S3 storage** (100GB imagery + OSM): $2.30/month
- **Lambda** (weekly satellite processing): $30/month
- **SageMaker training** (monthly growth model updates): ml.p3.2xlarge, 4 hours = $15/month
- **Batch** (distributed geospatial analysis): $40/month
- **QuickSight** (dashboards, 5 users): $45/month
- **Total**: $150-300/month for automated urban monitoring

**Scaling**:
- 5-10 cities: $500-800/month
- State-wide (50+ cities): $3K-8K/month

**Optimization tips**:
- Use spot instances for SageMaker and Batch (60-70% savings)
- Cache processed imagery to reduce Lambda compute
- Use Sentinel-2 COGs (Cloud-Optimized GeoTIFFs) for faster access
- Archive historical imagery to S3 Glacier ($0.004/GB/month)

## Support

- **Questions**: [GitHub Discussions](https://github.com/scttfrdmn/research-jumpstart/discussions)
- **Issues**: [GitHub Issues](https://github.com/scttfrdmn/research-jumpstart/issues)
- **Office Hours**: [Every Tuesday](../../../docs/community/office-hours.md)

## Citation

If you use this project in your research, please cite:

```bibtex
@software{research_jumpstart_city_analytics,
  title = {City Analytics at Scale: Research Jumpstart},
  author = {Research Jumpstart Community},
  year = {2025},
  url = {https://github.com/scttfrdmn/research-jumpstart},
  note = {Accessed: [date]}
}
```

Also cite the appropriate data sources:
- **Landsat**: USGS Earth Explorer, https://earthexplorer.usgs.gov/
- **Sentinel-2**: Copernicus Open Access Hub, ESA
- **OpenStreetMap**: https://www.openstreetmap.org/

## License

Apache 2.0 - See [LICENSE](../../../LICENSE) for details.

---

*Part of [Research Jumpstart](https://github.com/scttfrdmn/research-jumpstart) - Pre-built research workflows for cloud computing*
