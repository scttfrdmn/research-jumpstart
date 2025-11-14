# Multi-Platform Social Dynamics Ensemble

**Duration:** 5-6 hours
**Platform:** SageMaker Studio Lab ONLY
**Cost:** $0 (free Studio Lab account, no AWS charges)
**Data:** ~10GB multi-platform social network data

## Research Goal

Perform cross-platform social network analysis using ensemble Graph Neural Networks. Train models on Twitter, Reddit, Facebook, and other platforms to predict influence patterns, detect coordinated behavior, and analyze temporal dynamics of information diffusion.

**This project requires Studio Lab** - it won't work on Colab Free due to:
- **10GB dataset** (Colab has no persistent storage)
- **5-6 hour continuous training** (Colab disconnects after 90 min idle)
- **Model checkpointing essential** (must resume from checkpoints)
- **Environment persistence** (complex dependency stack with PyTorch Geometric)

## What This Enables

Real research that isn't possible on Colab:

### 🔬 Dataset Persistence
- Download 10GB of multi-platform data **once**
- Access instantly in all future sessions
- No 20-minute re-downloads every session
- Cache intermediate graph embeddings

### ⚡ Long-Running Training
- Train 5-6 GNN models (40-60 min each)
- Total compute: 5-6 hours continuous
- Automatic checkpointing every epoch
- Resume from checkpoint if needed

### 🧪 Reproducible Environments
- Conda environment with PyTorch Geometric
- Persists between sessions
- No reinstalling dependencies
- Team members use identical setup

### 📊 Iterative Analysis
- Save ensemble predictions
- Build on previous runs
- Refine models incrementally
- Collaborative notebook development

## What You'll Build

Multi-notebook research workflow:

1. **Data Acquisition** (60 min)
   - Download Twitter, Reddit, Facebook graphs (~10GB total)
   - Cache in persistent storage
   - Build unified graph representations
   - Generate cross-platform features

2. **Ensemble GNN Training** (5-6 hours)
   - Train GraphSAGE for each platform
   - Graph Attention Networks for influence prediction
   - Temporal Graph Networks for dynamics
   - Checkpoint every epoch
   - Parallel training workflows

3. **Cross-Platform Analysis** (60 min)
   - Compare influence patterns across platforms
   - Detect coordinated behavior
   - Temporal dynamics analysis
   - Information cascade tracking

4. **Results Visualization** (45 min)
   - Interactive network visualizations
   - Temporal evolution plots
   - Cross-platform comparison dashboards
   - Publication-ready figures

## Datasets

**Multi-Platform Social Network Collection**
- **Platforms:** Twitter, Reddit, Facebook, Mastodon, Discord
- **Nodes:** 5M+ users across platforms
- **Edges:** 50M+ interactions
- **Features:** User metadata, temporal info, content features
- **Period:** 2023 (6 months of activity)
- **Format:** Edge lists + node features (JSON/CSV/pickle)
- **Total size:** ~10GB compressed
- **Storage:** Cached in Studio Lab's 15GB persistent storage

## Setup

### 1. Create Studio Lab Account
1. Go to [SageMaker Studio Lab](https://studiolab.sagemaker.aws)
2. Request free account (approval in 1-2 days)
3. No credit card or AWS account needed

### 2. Import Project
```bash
# In Studio Lab terminal
git clone https://github.com/YOUR_USERNAME/research-jumpstart.git
cd research-jumpstart/projects/social-science/network-analysis/tier-1
```

### 3. Install Dependencies
```bash
# Create conda environment (persists between sessions!)
conda env create -f environment.yml
conda activate social-studio-lab

# Or use pip
pip install -r requirements.txt
```

### 4. Run Notebooks in Order
1. `01_data_preparation.ipynb` - Download and cache multi-platform data
2. `02_graph_construction.ipynb` - Build unified graph representations
3. `03_gnn_training.ipynb` - Train ensemble Graph Neural Networks
4. `04_cross_platform_analysis.ipynb` - Analyze influence patterns
5. `05_temporal_dynamics.ipynb` - Study network evolution over time
6. `06_visualization_dashboard.ipynb` - Create interactive visualizations

## Key Features

### Persistence Example
```python
# Save GNN model checkpoint (persists between sessions!)
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'saved_models/gnn_twitter_epoch_10.pt')

# Load in next session
checkpoint = torch.load('saved_models/gnn_twitter_epoch_10.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

### Longer Computations
```python
# Run intensive GNN training that would timeout on Colab
# Studio Lab: 12-hour GPU sessions, 4-hour CPU sessions
for epoch in range(50):
    train_loss = train_epoch(model, data)
    save_checkpoint(model, epoch)  # Resume-able!
```

### Shared Utilities
```python
# Import from project modules (better code organization)
from src.data_utils import load_twitter_graph, load_reddit_graph
from src.gnn_models import GraphSAGE, GAT, TemporalGNN
from src.visualization import plot_influence_network
```

## Project Structure

```
tier-1/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment
│
├── notebooks/
│   ├── 01_data_preparation.ipynb      # Download and cache data
│   ├── 02_graph_construction.ipynb    # Build graph representations
│   ├── 03_gnn_training.ipynb          # Train GNN models
│   ├── 04_cross_platform_analysis.ipynb  # Cross-platform analysis
│   ├── 05_temporal_dynamics.ipynb     # Temporal analysis
│   └── 06_visualization_dashboard.ipynb  # Interactive visualizations
│
├── src/
│   ├── __init__.py
│   ├── data_utils.py                  # Data loading utilities
│   ├── graph_utils.py                 # Graph construction
│   ├── gnn_models.py                  # GNN architectures
│   ├── analysis.py                    # Analysis functions
│   └── visualization.py               # Plotting functions
│
├── data/                              # Persistent data storage (gitignored)
│   ├── raw/                          # Downloaded datasets
│   ├── processed/                    # Graph representations
│   └── README.md                     # Data documentation
│
└── saved_models/                      # Model checkpoints (gitignored)
    └── README.md                      # Model documentation
```

## Why This Needs Studio Lab

| Requirement | Colab Free | Studio Lab |
|-------------|------------|------------|
| **10GB dataset** | ❌ No storage | ✅ 15GB persistent |
| **5-6 hour training** | ❌ 90 min limit | ✅ 12 hour sessions |
| **GNN frameworks** | ❌ Complex install | ✅ Conda persists |
| **Checkpointing** | ❌ Lost on disconnect | ✅ Persists forever |
| **Environment setup** | ❌ Reinstall each time | ✅ One-time setup |
| **Resume analysis** | ❌ Start from scratch | ✅ Pick up where you left off |
| **Team sharing** | ❌ Copy/paste notebooks | ✅ Git integration |

**Bottom line:** This research workflow is not viable on Colab Free.

## Time Estimate

**First Run:**
- Setup: 20 minutes (one-time)
- Data download: 60 minutes (one-time, ~10GB)
- Environment setup: 15 minutes (one-time, PyTorch Geometric)
- GNN training: 5-6 hours
- Analysis: 2-3 hours
- **Total: 8-10 hours**

**Subsequent Runs:**
- Data: Instant (cached)
- Environment: Instant (persisted)
- Training: 5-6 hours (or resume from checkpoint)
- **Total: 5-6 hours**

You can pause and resume at any time!

## Research Applications

This workflow enables research in:

- **Influence Propagation:** How information spreads across platforms
- **Coordinated Behavior:** Detecting bot networks and manipulation
- **Cross-Platform Dynamics:** How communities span multiple networks
- **Temporal Evolution:** How networks change over time
- **Polarization Studies:** Echo chambers and filter bubbles
- **Misinformation Tracking:** How false information spreads

## Key Methods

- **Graph Neural Networks:** GraphSAGE, GAT, GCN architectures
- **Temporal Graph Networks:** Capture time-evolving dynamics
- **Community Detection:** Louvain, label propagation, spectral clustering
- **Influence Prediction:** PageRank, centrality measures, learned embeddings
- **Cascade Analysis:** Independent cascade model, linear threshold model
- **Transfer Learning:** Train on one platform, apply to others

## Next Steps

After mastering Studio Lab:

- **Tier 2:** AWS Neptune for graph database, real-time analysis - $5-15
- **Tier 3:** Production infrastructure with streaming ingestion - $50-500/month

## Resources

- [SageMaker Studio Lab Documentation](https://studiolab.sagemaker.aws/docs)
- [PyTorch Geometric Tutorials](https://pytorch-geometric.readthedocs.io/)
- [Stanford SNAP Datasets](http://snap.stanford.edu/data/)
- [Network Science Book](http://networksciencebook.com/)

## Troubleshooting

### Environment Issues
```bash
# Reset environment
conda env remove -n social-studio-lab
conda env create -f environment.yml
```

### Storage Full
```bash
# Check usage
du -sh data/ saved_models/

# Clean old files
rm -rf data/raw/old_*.pkl
rm -rf saved_models/checkpoint_epoch_*.pt
```

### GPU Out of Memory
```python
# Reduce batch size
batch_size = 512  # Try 256 or 128

# Use gradient accumulation
# Accumulate gradients over 2 steps
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss = loss / 2  # Normalize by accumulation steps
    loss.backward()

    if (i + 1) % 2 == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Session Timeout
Data and checkpoints persist! Just restart and continue where you left off.

---

**Built for SageMaker Studio Lab with [Claude Code](https://claude.com/claude-code)**
