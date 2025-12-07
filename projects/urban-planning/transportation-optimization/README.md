# Transportation Network Optimization at Scale

Large-scale transportation network optimization with real-time traffic prediction, transit planning, mobility analysis, and equity assessment on cloud infrastructure.

## Quick Start by Tier

**New here?** Start with tier-0 (60-90 min, free) to learn transportation network optimization fundamentals.

### 🟢 Tier 0: Transportation Network Optimization (60-90 min, FREE)
**[Launch tier-0 project →](tier-0/)**

Optimize urban transportation networks using graph algorithms:
- ✅ Synthetic city transportation network (nodes, edges, traffic flows)
- ✅ Network flow optimization (shortest paths, traffic assignment)
- ✅ Bottleneck identification (congestion points, capacity analysis)
- ✅ Accessibility analysis (job accessibility, transit coverage metrics)
- ✅ Route optimization and network efficiency
- ✅ Complete in 60-90 minutes
- ✅ No AWS account needed (Colab or Studio Lab)

**Platform**: Google Colab or SageMaker Studio Lab
**Cost**: $0

[View tier-0 README →](tier-0/README.md) | [Open in Colab →](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/urban-planning/transportation-optimization/tier-0/transportation-optimization.ipynb)

---

### 🟡 Tier 1: Multi-City Transportation Analysis (4-8 hours, FREE)
**[Launch tier-1 project →](tier-1/)**

Comprehensive transportation analysis across multiple cities:
- ✅ 10GB+ transportation data (5-6 metro areas: OSM, GTFS, traffic sensors)
- ✅ Advanced graph analytics (centrality metrics, community detection)
- ✅ Transit network optimization (route planning, frequency optimization)
- ✅ Comparative city analysis (benchmark transit systems)
- ✅ Persistent storage for large networks (Studio Lab)
- ✅ Still free, no AWS account

**Platform**: SageMaker Studio Lab
**Cost**: $0

[View tier-1 README →](tier-1/README.md)

---

### 🟠 Tier 2: Production Transportation Platform (2-3 days, varies by city size)
**[Launch tier-2 project →](tier-2/)**

Research-grade transportation optimization infrastructure:
- ✅ CloudFormation one-click deployment
- ✅ 100GB+ transportation data on S3 (OpenStreetMap, GTFS, traffic sensors)
- ✅ Real-time traffic prediction (GCN + LSTM models, 15-60 min forecasts, MAE 5-10 mph)
- ✅ Transit optimization (route planning, coverage analysis, frequency optimization)
- ✅ Graph analysis at scale (NetworkX, igraph, Neptune for large networks)
- ✅ Equity analysis (transit access by demographics, job accessibility gaps)
- ✅ Scenario planning with SUMO simulation

**Platform**: AWS with CloudFormation
**Cost**: $2K-5K/month (1M pop city), $8K-15K/month (5M pop metro), $25K-50K/month (multi-city)

[View tier-2 README →](tier-2/README.md)

---

### 🔴 Tier 3: National Transportation Platform (Ongoing, $100K-250K/month)
**[Launch tier-3 project →](tier-3/)**

Production platform for transportation agencies:
- ✅ National-scale network analysis (50+ cities, 20M+ population)
- ✅ Real-time traffic prediction and congestion management
- ✅ Multi-modal integration (transit, bike-share, ride-hail, autonomous vehicles)
- ✅ AI-assisted planning (Amazon Bedrock for infrastructure recommendations)
- ✅ Integration with city systems (traffic signals, transit AVL, smart cards)
- ✅ Federal-level policy analysis and scenario modeling
- ✅ Team collaboration with versioned transportation models

**Platform**: AWS multi-account with enterprise support
**Cost**: $100K-250K/month

[View tier-3 README →](tier-3/README.md)

---

## What You'll Learn

Across all tiers, this project teaches:
- Graph algorithms for transportation networks (shortest paths, traffic assignment)
- Real-time traffic prediction with GCN + LSTM models
- Transit network optimization (coverage, frequency, route planning)
- Mobility pattern analysis (origin-destination matrices, mode share)
- Transportation equity analysis (accessibility by demographics)
- Distributed graph analytics on cloud infrastructure

## Technologies & Tools

- **Data sources**: OpenStreetMap (OSM), GTFS (transit schedules), traffic sensors, GPS traces, smart card data
- **Languages**: Python 3.9+
- **Core libraries**: pandas, numpy, scipy, scikit-learn
- **Graph analysis**: NetworkX, OSMnx, igraph, Neptune (AWS graph database)
- **Geospatial**: PostGIS, GeoPandas, Shapely
- **ML frameworks**: PyTorch Geometric (GCN for traffic prediction), LSTM, XGBoost
- **Simulation**: SUMO (Simulation of Urban MObility)
- **Cloud services** (tier 2+): Kinesis (real-time traffic), PostGIS on RDS, Neptune (graph DB), SageMaker (ML), Batch, Timestream (sensor data), QuickSight (dashboards)

## Project Structure

```
transportation-optimization/
├── tier-0/              # Network optimization (60-90 min, FREE)
│   ├── transportation-optimization.ipynb
│   ├── README.md
│   └── requirements.txt
├── tier-1/              # Multi-city (4-8 hours, FREE)
│   ├── notebooks/
│   ├── src/
│   ├── data/
│   ├── environment.yml
│   └── README.md
├── tier-2/              # Production (2-3 days, $2K-15K/mo)
│   ├── cloudformation/
│   ├── notebooks/
│   ├── src/
│   ├── tests/
│   └── README.md
└── tier-3/              # National platform (ongoing, $100K-250K/mo)
    ├── cloudformation/
    ├── notebooks/
    ├── src/
    ├── infrastructure/
    └── README.md
```

## Progression Path

```
Tier 0           → Tier 1          → Tier 2            → Tier 3
Network Opt        Multi-City         Production          National
Synthetic data     5-6 metro areas    Real-time city      50+ cities
60-90 min          4-8 hours          2-3 days            Ongoing
FREE               FREE               $2K-15K/mo          $100K-250K/mo
```

You can:
- ✅ Skip tiers if you have AWS experience and large-scale transportation analysis needs
- ✅ Stop at any tier - tier-1 is great for academic studies, tier-2 for city planning depts
- ✅ Mix and match - use tier-0 for prototyping, tier-2 for operational planning

[Understanding tiers →](../../../docs/projects/tiers.md)

## Transportation Applications

- **Transit optimization**: Maximize coverage and efficiency with data-driven route planning
- **Traffic prediction**: Real-time congestion forecasting with GCN+LSTM (15-60 min ahead, MAE 5-10 mph)
- **Mobility analysis**: Understand travel patterns, mode choices, trip generation
- **Equity analysis**: Identify transit deserts, measure job accessibility by demographics
- **Infrastructure planning**: Model scenarios for new transit lines, BRT, bike lanes
- **Congestion mitigation**: Optimize signal timing, pricing, demand management

## Related Projects

- **[City Analytics](../city-analytics/)** - Urban growth prediction and planning
- **[Social Science - Network Analysis](../../social-science/network-analysis/)** - Graph theory methods
- **[Economics - Time Series Forecasting](../../economics/time-series-forecasting/)** - Similar prediction techniques

## Common Use Cases

- **City planners**: Transit route optimization, corridor studies, scenario planning
- **Transportation agencies**: Real-time operations, service planning, capital investments
- **Mobility researchers**: Analyze travel behavior, mode choice, accessibility
- **Equity advocates**: Document transit gaps, measure accessibility by race/income
- **Private sector**: Ride-hail optimization, autonomous vehicle routing
- **Traffic engineers**: Congestion management, signal optimization, ITS planning

## Cost Estimates

**Tier 2 Production**:
- **Single city (1M pop)**: $2,000-5,000/month
- **Metro area (5M pop)**: $8,000-15,000/month
- **Multi-city (20M pop)**: $25,000-50,000/month
- **National network**: $100,000-250,000/month

**Breakdown (Metro Area - 5M Pop)**:
- **PostGIS on RDS** (network database, db.r5.2xlarge): $2,500/month
- **Neptune** (graph DB for large networks, db.r5.large): $350/month
- **Kinesis** (real-time traffic data): $1,500/month
- **SageMaker** (GCN+LSTM traffic prediction, daily training): $2,000/month
- **Timestream** (sensor time series): $500/month
- **QuickSight** (dashboards, 20 users): $180/month
- **Batch** (SUMO simulations, scenario modeling): $1,000/month
- **Total**: $8,000-15,000/month for metro-scale transportation platform

**Optimization tips**:
- Use spot instances for Batch simulations (60-70% savings)
- Cache network graphs and preprocessed GTFS data
- Use PostGIS spatial indexes for fast queries
- Process traffic data in batches for efficiency

## Support

- **Questions**: [GitHub Discussions](https://github.com/scttfrdmn/research-jumpstart/discussions)
- **Issues**: [GitHub Issues](https://github.com/scttfrdmn/research-jumpstart/issues)
- **Office Hours**: [Every Tuesday](../../../docs/community/office-hours.md)

## Citation

If you use this project in your research, please cite:

```bibtex
@software{research_jumpstart_transportation,
  title = {Transportation Network Optimization at Scale: Research Jumpstart},
  author = {Research Jumpstart Community},
  year = {2025},
  url = {https://github.com/scttfrdmn/research-jumpstart},
  note = {Accessed: [date]}
}
```

Also cite the appropriate data sources:
- **OpenStreetMap**: https://www.openstreetmap.org/
- **GTFS**: General Transit Feed Specification

## License

Apache 2.0 - See [LICENSE](../../../LICENSE) for details.

---

*Part of [Research Jumpstart](https://github.com/scttfrdmn/research-jumpstart) - Pre-built research workflows for cloud computing*
