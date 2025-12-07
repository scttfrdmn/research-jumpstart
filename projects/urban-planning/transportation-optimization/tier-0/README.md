# Transportation Network Optimization and Routing

**Duration:** 60-90 minutes
**Platform:** Google Colab or SageMaker Studio Lab
**Cost:** $0 (no AWS account needed)
**Data:** OpenStreetMap (~300MB), synthetic traffic data

## Research Goal

Apply graph algorithms and optimization techniques to urban transportation networks including shortest path routing, traffic flow optimization, transit accessibility analysis, and vehicle routing problems using real street network data from OpenStreetMap.

## Quick Start

### Run in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/urban-planning/transportation-optimization/tier-0/transport-optimization.ipynb)

### Run in SageMaker Studio Lab
[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/urban-planning/transportation-optimization/tier-0/transport-optimization.ipynb)

## What You'll Build

1. **Download street network** (OSM for medium city, ~300MB, 10-15 min)
2. **Network analysis** (centrality, connectivity, accessibility metrics)
3. **Shortest path routing** (Dijkstra, A*, multiple routes)
4. **Traffic assignment** (user equilibrium, 20-30 minutes)
5. **Transit accessibility** (isochrone maps, 15-minute city analysis)
6. **Vehicle routing** (TSP, VRP for delivery optimization)

## Dataset

**OpenStreetMap Street Network**
- Area: Medium city (~50 km² urban area)
- Nodes: ~50,000 intersections
- Edges: ~100,000 road segments
- Attributes: Road type, speed limits, one-way restrictions, lanes
- Size: ~300MB for network + metadata
- Format: OSM XML, converted to graph (NetworkX, OSMnx)

**Supplementary Data:**
- **Synthetic traffic:** Hourly volumes, travel times
- **Transit stops:** Bus/metro station locations
- **POIs:** Schools, hospitals, grocery stores (from OSM)
- **Demographics:** Population density (census blocks)

## Colab Considerations

This notebook works on Colab but you'll notice:
- **10-15 minute download** (large OSM files, no caching)
- **Medium city only** (metropolitan areas crash due to memory)
- **Simple traffic models** (static assignment vs. dynamic simulation)
- **Limited optimization** (small VRP instances, 20-30 vehicles)
- **No real-time data** (historical/synthetic only)

These limitations prevent realistic transportation planning applications.

## What's Included

- Single Jupyter notebook (`transport-optimization.ipynb`)
- OSMnx for street network download and analysis
- NetworkX for graph algorithms
- Shortest path routing (Dijkstra, A*, bidirectional)
- Traffic assignment (Frank-Wolfe algorithm)
- Accessibility analysis (isochrones, gravity models)
- Vehicle routing problem (VRP) solver
- Interactive maps (Folium)

## Key Methods

- **Graph Algorithms:** Dijkstra, A*, bidirectional search
- **Network Centrality:** Betweenness, closeness, PageRank
- **Traffic Assignment:** User equilibrium (Wardrop principle), Frank-Wolfe
- **Accessibility Metrics:** Hansen accessibility, cumulative opportunities
- **Routing Optimization:** TSP, VRP with capacity constraints
- **Spatial Analysis:** Isochrones, service areas, coverage
- **Network Design:** Facility location, transit planning

## Optimization Outputs

1. **Optimal Routes:** Shortest/fastest paths with turn-by-turn directions
2. **Traffic Flows:** Link volumes and speeds under equilibrium
3. **Accessibility Maps:** 5/10/15-minute travel time isochrones
4. **VRP Solutions:** Optimized delivery routes (20-30 vehicles)
5. **Network Metrics:** Bottlenecks, critical links, redundancy
6. **Transit Coverage:** Population within 400m of transit stops

## Next Steps

**Need metropolitan-scale optimization?** This project handles medium cities:

- **Tier 1:** [Large-scale transportation modeling](../tier-1/) on Studio Lab
  - Metropolitan areas (1M+ population, 500K+ edges)
  - Dynamic traffic assignment (DTA)
  - Multi-modal network optimization
  - Large VRP instances (1000+ vehicles)
  - Persistent storage for iterative planning

- **Tier 2:** [AWS-integrated smart mobility](../tier-2/) with cloud computing
  - Real-time traffic data integration
  - Distributed graph processing with EMR
  - S3 for historical travel time matrices
  - Lambda for routing API services
  - SageMaker for demand prediction

- **Tier 3:** [Production transportation system](../tier-3/) with CloudFormation
  - Region-wide optimization (multi-city, 10M+ population)
  - Real-time traffic routing with AWS Location Service
  - Agent-based traffic simulation (AWS Batch, 1M+ agents)
  - Integrated transit planning dashboards
  - Fleet management and optimization at scale
  - Integration with connected vehicle systems

## Requirements

Pre-installed in Colab and Studio Lab:
- Python 3.9+
- OSMnx (street network analysis)
- NetworkX (graph algorithms)
- geopandas, shapely (spatial data)
- scipy.optimize (optimization algorithms)
- folium (interactive maps)
- matplotlib, seaborn

**Data Download:** Requires 300MB storage and 10-15 minute OSM download
