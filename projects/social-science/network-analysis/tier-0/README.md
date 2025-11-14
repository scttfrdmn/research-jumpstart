# Social Network Analysis with Deep Learning

**Duration:** 60-90 minutes
**Platform:** Google Colab or SageMaker Studio Lab
**Cost:** $0 (no AWS account needed)
**Data:** ~1.5GB social network graph (Twitter/Reddit)

## Research Goal

Train a Graph Neural Network (GNN) to detect communities and predict influence patterns in large-scale social networks. Analyze information diffusion, identify influential nodes, and understand network dynamics using deep learning.

## Quick Start

### Run in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/social-science/network-analysis/tier-0/social-network-quick-demo.ipynb)

### Run in SageMaker Studio Lab
[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/social-science/network-analysis/tier-0/social-network-quick-demo.ipynb)

## What You'll Build

1. **Download social network data** (~1.5GB Twitter/Reddit graph, takes 15-20 min)
2. **Preprocess network** (build adjacency matrices, node features)
3. **Train Graph Neural Network** (60-75 minutes on GPU)
4. **Community detection** (identify clusters, measure modularity)
5. **Influence prediction** (rank nodes by influence scores)

## Dataset

**Twitter/Reddit Social Network Graph**
- Nodes: 500K users
- Edges: 5M interactions (retweets, replies, mentions)
- Features: User metadata (followers, posts, engagement)
- Period: 2023 (1 month of activity)
- Format: Edge list + node features
- Size: ~1.5GB compressed JSON/CSV
- Source: Publicly available social network datasets

## Colab Considerations

This notebook works on Colab but you'll notice:
- **20-minute download** at session start (no persistence)
- **60-75 minute training** (close to timeout limit)
- **Re-download required** if session disconnects
- **~11GB RAM usage** (near Colab's limit)

These limitations become important for real research workflows.

## What's Included

- Single Jupyter notebook (`social-network-quick-demo.ipynb`)
- Network data loading utilities
- Graph Neural Network architecture (GraphSAGE)
- Community detection algorithms
- Influence prediction pipeline
- Network visualization tools

## Key Methods

- **Graph Neural Networks:** Learn node embeddings from graph structure
- **Community Detection:** Louvain algorithm, modularity optimization
- **Influence Prediction:** PageRank, node centrality measures
- **Information Diffusion:** Cascade models, viral spread patterns
- **Network Sampling:** Efficient subgraph extraction

## Next Steps

**Experiencing limitations?** This project pushes Colab to its limits:

- **Tier 1:** [Multi-platform ensemble analysis](../tier-1/) on Studio Lab
  - Cache 10GB of data (Twitter, Reddit, Facebook graphs)
  - Train ensemble models (5-6 hours continuous)
  - Temporal dynamics analysis
  - No session timeouts

- **Tier 2:** [AWS-integrated workflows](../tier-2/) with S3 and Neptune
  - Store 100GB+ social graphs on S3
  - Graph database with Neptune
  - Real-time influence tracking

- **Tier 3:** [Production-scale analysis](../tier-3/) with full CloudFormation
  - Multi-platform integration (10+ networks)
  - Streaming data ingestion
  - Distributed graph processing

## Requirements

Pre-installed in Colab and Studio Lab:
- Python 3.9+, PyTorch
- NetworkX, PyTorch Geometric
- scikit-learn, pandas
- matplotlib, seaborn

**Note:** First run downloads 1.5GB of data (15-20 minutes)
