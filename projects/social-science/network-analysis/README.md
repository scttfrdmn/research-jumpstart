# Social Network Analysis at Scale

Large-scale social network analysis using graph theory, community detection, influence measurement, and network dynamics for studying social structures on cloud infrastructure.

## Quick Start by Tier

**New here?** Start with tier-0 (60-90 min, free) to learn social network analysis fundamentals.

### 🟢 Tier 0: Social Network Analysis (60-90 min, FREE)
**[Launch tier-0 project →](tier-0/)**

Analyze social network structures and dynamics:
- ✅ Synthetic social network (500 nodes, 2000 edges, friendship/collaboration network)
- ✅ Centrality metrics (degree, betweenness, eigenvector, closeness centrality)
- ✅ Community detection (Louvain algorithm, modularity optimization)
- ✅ Network visualization (graph layouts, community coloring)
- ✅ Key player identification (influencers, bridges, hubs)
- ✅ Complete in 60-90 minutes
- ✅ No AWS account needed (Colab or Studio Lab)

**Platform**: Google Colab or SageMaker Studio Lab
**Cost**: $0

[View tier-0 README →](tier-0/README.md) | [Open in Colab →](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/social-science/network-analysis/tier-0/social-network-quick-demo.ipynb)

---

### 🟡 Tier 1: Large-Scale Network Analysis (4-8 hours, FREE)
**[Launch tier-1 project →](tier-1/)**

Comprehensive network analysis with real social data:
- ✅ 10GB+ social network data (100K+ nodes, 1M+ edges from Twitter, Facebook, co-authorship)
- ✅ Advanced community detection (Infomap, Label Propagation, hierarchical methods)
- ✅ Influence propagation models (Independent Cascade, Linear Threshold)
- ✅ Temporal network dynamics (evolution over time, tie formation/dissolution)
- ✅ Persistent storage for large graphs (Studio Lab)
- ✅ Still free, no AWS account

**Platform**: SageMaker Studio Lab
**Cost**: $0

[View tier-1 README →](tier-1/README.md)

---

### 🟠 Tier 2: Production Network Analysis (2-3 days, $300-800/month)
**[Launch tier-2 project →](tier-2/)**

Research-grade social network analysis infrastructure:
- ✅ CloudFormation one-click deployment
- ✅ 100GB+ network data on S3 (millions of nodes, billions of edges)
- ✅ Graph database with AWS Neptune (sub-second queries on massive graphs)
- ✅ Distributed graph processing with EMR + GraphFrames
- ✅ SageMaker for graph neural networks (node classification, link prediction)
- ✅ Real-time influence tracking and community evolution
- ✅ Publication-ready network visualizations and metrics

**Platform**: AWS with CloudFormation
**Cost**: $300-800/month for continuous network analysis

[View tier-2 README →](tier-2/README.md)

---

### 🔴 Tier 3: Enterprise Social Network Platform (Ongoing, $5K-25K/month)
**[Launch tier-3 project →](tier-3/)**

Production platform for social network research:
- ✅ Billion-node networks (Twitter, Facebook, LinkedIn scale)
- ✅ Real-time network monitoring and anomaly detection
- ✅ Advanced influence modeling (cascading behavior, viral spread)
- ✅ Multi-network integration (cross-platform user linking)
- ✅ AI-assisted interpretation (Amazon Bedrock for network insights)
- ✅ Privacy-preserving analytics (differential privacy, k-anonymity)
- ✅ Team collaboration with versioned network snapshots

**Platform**: AWS multi-account with enterprise support
**Cost**: $5K-25K/month

[View tier-3 README →](tier-3/README.md)

---

## What You'll Learn

Across all tiers, this project teaches:
- Graph theory fundamentals (centrality, clustering, paths)
- Community detection algorithms (Louvain, Infomap, modularity)
- Influence and information diffusion modeling
- Network visualization and interpretation
- Temporal network dynamics and evolution
- Distributed graph processing on cloud infrastructure

## Technologies & Tools

- **Data sources**: Twitter API, Facebook Graph API, academic collaboration networks, Wikipedia links, Reddit networks
- **Languages**: Python 3.9+
- **Core libraries**: pandas, numpy, scipy, scikit-learn
- **Graph analysis**: NetworkX, igraph, graph-tool, PyTorch Geometric
- **Community detection**: python-louvain, cdlib, Infomap
- **Visualization**: matplotlib, seaborn, plotly, pyvis (interactive networks), Gephi
- **Cloud services** (tier 2+): Neptune (graph database), EMR (GraphFrames), SageMaker (graph neural networks), S3 (graph storage)

## Project Structure

```
network-analysis/
├── tier-0/              # Network fundamentals (60-90 min, FREE)
│   ├── social-network-quick-demo.ipynb
│   ├── README.md
│   └── requirements.txt
├── tier-1/              # Large-scale (4-8 hours, FREE)
│   ├── notebooks/
│   ├── src/
│   ├── data/
│   ├── environment.yml
│   └── README.md
├── tier-2/              # Production (2-3 days, $300-800/mo)
│   ├── cloudformation/
│   ├── notebooks/
│   ├── src/
│   ├── tests/
│   └── README.md
└── tier-3/              # Enterprise platform (ongoing, $5K-25K/mo)
    ├── cloudformation/
    ├── notebooks/
    ├── src/
    ├── infrastructure/
    └── README.md
```

## Progression Path

```
Tier 0           → Tier 1          → Tier 2            → Tier 3
Fundamentals       Large-Scale        Production          Enterprise
500 nodes          100K+ nodes        Millions            Billions
60-90 min          4-8 hours          2-3 days            Ongoing
FREE               FREE               $300-800/mo         $5K-25K/mo
```

You can:
- ✅ Skip tiers if you have AWS experience and billion-node network needs
- ✅ Stop at any tier - tier-1 is great for dissertations, tier-2 for funded research
- ✅ Mix and match - use tier-0 for teaching, tier-2 for publications

[Understanding tiers →](../../../docs/projects/tiers.md)

## Social Network Applications

- **Community detection**: Identify tightly-knit groups, measure polarization (modularity 0.3-0.7)
- **Influence measurement**: Rank users by centrality, identify opinion leaders and bridges
- **Information diffusion**: Model viral spread, predict cascade size, optimize seeding
- **Network evolution**: Track tie formation, community dynamics, structural change over time
- **Link prediction**: Predict future connections, recommend collaborators/friends
- **Anomaly detection**: Identify bot networks, coordinated campaigns, unusual patterns

## Related Projects

- **[Social Media Analysis](../social-media-analysis/)** - Sentiment and content analysis
- **[Urban Planning - Transportation](../../urban-planning/transportation-optimization/)** - Graph algorithms
- **[Public Health - Epidemiology](../../public-health/epidemiology/)** - Contact tracing networks

## Common Use Cases

- **Sociologists**: Study social structure, group formation, inequality, tie strength
- **Network scientists**: Test theories of network formation, influence, dynamics
- **Marketing researchers**: Identify influencers, optimize viral campaigns, measure reach
- **Political scientists**: Analyze political networks, polarization, echo chambers
- **Organizations**: Map collaboration networks, optimize communication, identify silos
- **Security analysts**: Detect coordinated campaigns, bot networks, manipulation

## Cost Estimates

**Tier 2 Production (Million-Node Networks)**:
- **S3 storage** (100GB network data): $2.30/month
- **Neptune** (graph database, db.r5.2xlarge): $1,000/month
- **EMR** (distributed graph processing, monthly analysis): $200/month
- **SageMaker** (graph neural networks): ml.p3.2xlarge, 20 hours/month = $150/month
- **Lambda** (preprocessing, API): $30/month
- **Total**: $300-800/month for continuous network analysis

**Scaling**:
- 100M nodes: $2,000-5,000/month
- 1B+ nodes: $5,000-25,000/month

**Optimization tips**:
- Use Neptune read replicas for query-intensive workloads
- Cache community detection results (expensive to recompute)
- Use spot instances for EMR processing (60-70% savings)
- Store static snapshots on S3, dynamic queries on Neptune

## Support

- **Questions**: [GitHub Discussions](https://github.com/scttfrdmn/research-jumpstart/discussions)
- **Issues**: [GitHub Issues](https://github.com/scttfrdmn/research-jumpstart/issues)
- **Office Hours**: [Every Tuesday](../../../docs/community/office-hours.md)

## Citation

If you use this project in your research, please cite:

```bibtex
@software{research_jumpstart_network_analysis,
  title = {Social Network Analysis at Scale: Research Jumpstart},
  author = {Research Jumpstart Community},
  year = {2025},
  url = {https://github.com/scttfrdmn/research-jumpstart},
  note = {Accessed: [date]}
}
```

## License

Apache 2.0 - See [LICENSE](../../../LICENSE) for details.

---

*Part of [Research Jumpstart](https://github.com/scttfrdmn/research-jumpstart) - Pre-built research workflows for cloud computing*
