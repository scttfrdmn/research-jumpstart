# Archaeological Site Analysis at Scale

Large-scale archaeological site discovery and analysis using LiDAR, machine learning for artifact classification, spatial pattern analysis, and Bayesian chronological modeling on AWS.

## Quick Start by Tier

**New here?** Start with tier-0 (60-90 min, free) to learn artifact classification with deep learning.

### 🟢 Tier 0: Artifact Classification (60-90 min, FREE)
**[Launch tier-0 project →](tier-0/)**

Classify archaeological artifacts using convolutional neural networks:
- ✅ Real artifact imagery (~1.5GB, 5,000 images from museum collections)
- ✅ Transfer learning with ResNet for pottery, tools, ornaments
- ✅ Handle class imbalance and data augmentation techniques
- ✅ Expert-validated classifications
- ✅ Complete in 60-90 minutes
- ✅ No AWS account needed (Colab or Studio Lab)

**Platform**: Google Colab or SageMaker Studio Lab
**Cost**: $0

[View tier-0 README →](tier-0/README.md) | [Open in Colab →](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/archaeology/site-analysis/tier-0/artifact-quick-demo.ipynb)

---

### 🟡 Tier 1: Multi-Site Pattern Analysis (4-8 hours, FREE)
**[Launch tier-1 project →](tier-1/)**

Analyze settlement patterns and spatial relationships:
- ✅ 10GB multi-site dataset (artifacts, site locations, environmental data)
- ✅ Spatial statistics: Ripley's K, nearest neighbor, Voronoi tessellation
- ✅ Environmental correlates: viewshed analysis, least-cost path modeling
- ✅ Predictive site location modeling
- ✅ Persistent storage for archaeological databases (Studio Lab)
- ✅ Still free, no AWS account

**Platform**: SageMaker Studio Lab
**Cost**: $0

[View tier-1 README →](tier-1/README.md)

---

### 🟠 Tier 2: Regional Survey Processing (2-3 days, $500-800 per site)
**[Launch tier-2 project →](tier-2/)**

Research-grade archaeological analysis infrastructure:
- ✅ CloudFormation one-click deployment
- ✅ LiDAR point cloud processing (PDAL, ground classification, DTM generation)
- ✅ Automated site detection with computer vision (mounds, structures, earthworks)
- ✅ Distributed artifact classification with AWS Batch
- ✅ 100GB+ LiDAR and artifact image archives on S3
- ✅ Bayesian radiocarbon dating with OxCal integration
- ✅ 3D photogrammetry pipeline (OpenDroneMap, Structure from Motion)
- ✅ Publication-ready outputs and visualizations

**Platform**: AWS with CloudFormation
**Cost**: $500-800 per site, $2K-4K/month for 100 sites

[View tier-2 README →](tier-2/README.md)

---

### 🔴 Tier 3: Continental-Scale Archaeological Platform (Ongoing, $8K-15K/month)
**[Launch tier-3 project →](tier-3/)**

Production platform for archaeological research teams:
- ✅ Multi-project collaboration with shared databases (Open Context, tDAR, DINAA)
- ✅ AI-assisted artifact interpretation (Amazon Bedrock)
- ✅ Petabyte-scale LiDAR processing for landscape archaeology
- ✅ Real-time trade network analysis (material flows, geochemical sourcing)
- ✅ Automated chronological modeling (IntChron radiocarbon database integration)
- ✅ Team workflows with versioned datasets
- ✅ Integration with museum collection management systems

**Platform**: AWS multi-account with enterprise support
**Cost**: $8K-15K/month for 1,000+ sites

[View tier-3 README →](tier-3/README.md)

---

## What You'll Learn

Across all tiers, this project teaches:
- LiDAR point cloud processing for site discovery (PDAL, ground classification, DTM generation)
- Deep learning for artifact classification (ResNet, EfficientNet, transfer learning)
- Spatial statistics for settlement patterns (Ripley's K, nearest neighbor, Voronoi)
- Bayesian chronological modeling (OxCal, phase analysis, synchrony testing)
- 3D photogrammetry (Structure from Motion, mesh reconstruction, web visualization)
- Trade network analysis (NetworkX, geochemical sourcing, centrality metrics)

## Technologies & Tools

- **Data sources**: Open Context, tDAR, DINAA, IntChron radiocarbon, OpenTopography LiDAR, Sentinel-2
- **Languages**: Python 3.9+
- **Core libraries**: pandas, numpy, scipy, scikit-learn, networkx
- **Archaeology tools**: PyOxCal, geostat-framework, archaeopy, scikit-spatial
- **Geospatial**: PDAL, GDAL, geopandas, rasterio, shapely, PostGIS
- **ML frameworks**: PyTorch, TensorFlow (ResNet50, EfficientNet), SageMaker
- **3D processing**: OpenDroneMap, Open3D, MeshLab, CloudCompare
- **Cloud services** (tier 2+): S3, EC2 (GPU for ML, g4dn.xlarge), Batch, RDS (PostGIS), SageMaker, Lambda, Glue, Athena

## Project Structure

```
site-analysis/
├── tier-0/              # Artifact classification (60-90 min, FREE)
│   ├── artifact-quick-demo.ipynb
│   ├── README.md
│   └── requirements.txt
├── tier-1/              # Pattern analysis (4-8 hours, FREE)
│   ├── notebooks/
│   ├── src/
│   ├── data/
│   ├── environment.yml
│   └── README.md
├── tier-2/              # Regional survey (2-3 days, $500-800/site)
│   ├── cloudformation/
│   ├── notebooks/
│   ├── src/
│   ├── tests/
│   └── README.md
└── tier-3/              # Continental scale (ongoing, $8K-15K/mo)
    ├── cloudformation/
    ├── notebooks/
    ├── src/
    ├── infrastructure/
    └── README.md
```

## Progression Path

```
Tier 0           → Tier 1          → Tier 2            → Tier 3
Artifact           Settlement         Regional            Continental
Classification     Patterns           Survey              Platform
60-90 min          4-8 hours          2-3 days            Ongoing
FREE               FREE               $500-800/site       $8K-15K/mo
```

You can:
- ✅ Skip tiers if you have AWS experience and regional survey needs
- ✅ Stop at any tier - tier-0 and tier-1 are great for teaching and pilot studies
- ✅ Mix and match - use tier-0 for method prototyping, tier-2 for dissertation research

[Understanding tiers →](../../../docs/projects/tiers.md)

## Archaeological Applications

- **LiDAR site discovery**: Detect Maya settlements in jungle canopy, European hillforts, mounds, earthworks
- **Artifact typology**: Automate pottery classification, lithic technology identification (85-95% accuracy)
- **Settlement patterns**: Test clustering hypotheses, environmental correlates, viewshed analysis
- **Chronological modeling**: Bayesian radiocarbon dating with OxCal, phase modeling, synchrony testing
- **3D reconstruction**: Virtual museums, metric analysis from photogrammetry, web visualization (glTF)
- **Trade networks**: Material flow analysis, geochemical sourcing (obsidian, ceramics), centrality metrics

## Related Projects

- **[Digital Humanities - Text Analysis](../../digital-humanities/text-corpus-analysis/)** - Similar ML classification techniques
- **[Climate Science - Satellite Imagery](../../climate-science/ensemble-analysis/)** - Remote sensing methods for landscape archaeology
- **[Social Science - Network Analysis](../../social-science/network-analysis/)** - Trade network and interaction analysis

## Common Use Cases

- **Academic research**: Dissertation projects, multi-site comparative studies, chronological synthesis
- **Cultural resource management**: Predictive modeling for site location, impact assessments
- **Museum collections**: Automate artifact classification for large digitization projects
- **Landscape archaeology**: LiDAR-based survey for site discovery in forested/agricultural areas
- **Public archaeology**: Virtual reconstructions for education and heritage tourism
- **Settlement pattern studies**: Test models of social organization, environmental adaptation

## Cost Estimates

**Tier 2 Regional Survey**:
- **Single site**: $500-800 (LiDAR processing, artifact classification, spatial analysis)
- **Regional survey (100 sites)**: $2K-4K/month (amortized over survey duration)
- **Large-scale (1,000+ sites)**: $8K-15K/month (continental-scale landscape archaeology)

**Data Requirements**:
- LiDAR: 1-10GB per site (airborne or terrestrial point clouds)
- Artifact images: 10,000+ photos at 5-10MB each (50-100GB)
- GIS data: Site locations, environmental layers, DEM (5-20GB)
- Radiocarbon dates: IntChron, context associations (1-5GB)

## Support

- **Questions**: [GitHub Discussions](https://github.com/scttfrdmn/research-jumpstart/discussions)
- **Issues**: [GitHub Issues](https://github.com/scttfrdmn/research-jumpstart/issues)
- **Office Hours**: [Every Tuesday](../../../docs/community/office-hours.md)

## Citation

If you use this project in your research, please cite:

```bibtex
@software{research_jumpstart_archaeological_analysis,
  title = {Archaeological Site Analysis at Scale: Research Jumpstart},
  author = {Research Jumpstart Community},
  year = {2025},
  url = {https://github.com/scttfrdmn/research-jumpstart},
  note = {Accessed: [date]}
}
```

Also cite the appropriate data sources:
- **Open Context**: https://opencontext.org
- **tDAR**: https://www.tdar.org
- **DINAA**: http://ux.opencontext.org/archaeology-site-data/
- **IntChron**: https://intchron.org
- **OpenTopography**: https://opentopography.org

## License

Apache 2.0 - See [LICENSE](../../../LICENSE) for details.

---

*Part of [Research Jumpstart](https://github.com/scttfrdmn/research-jumpstart) - Pre-built research workflows for cloud computing*
