# Crystal Structure Property Prediction with GNN

**Duration:** 60-90 minutes
**Platform:** Google Colab or SageMaker Studio Lab
**Cost:** $0 (no AWS account needed)
**Data:** ~1.5GB Materials Project database

## Research Goal

Train a graph neural network to predict material properties (band gap, formation energy) from crystal structures using the Materials Project database. Learn to represent materials as graphs and use deep learning for materials discovery.

## Quick Start

### Run in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/materials/computational-materials/tier-0/crystal-property-prediction.ipynb)

### Run in SageMaker Studio Lab
[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/materials/computational-materials/tier-0/crystal-property-prediction.ipynb)

## What You'll Build

1. **Download Materials Project data** (~1.5GB, takes 15-20 min)
2. **Convert crystal structures to graphs** (atoms as nodes, bonds as edges)
3. **Train GNN for band gap prediction** (60-75 minutes on GPU)
4. **Evaluate model performance** (MAE, R², predicted vs actual)
5. **Predict properties for new materials** (discover novel semiconductors)

## Dataset

**Materials Project Database**
- Source: Materials Project (materialsproject.org)
- Materials: ~50,000 inorganic crystals
- Properties: Band gap, formation energy, structure
- Size: ~1.5GB JSON/CSV files
- Access: Public API (no authentication needed)

## Colab Considerations

This notebook works on Colab but you'll notice:
- **15-20 minute download** at session start (no persistence)
- **60-75 minute training** (close to timeout limit)
- **Re-download required** if session disconnects
- **~11GB RAM usage** (near Colab's limit)

These limitations become important for real research workflows.

## What's Included

- Single Jupyter notebook (`crystal-property-prediction.ipynb`)
- Materials Project data access utilities
- Crystal Graph Convolutional Neural Network (CGCNN)
- Training and evaluation pipeline
- Property prediction for novel materials

## Key Methods

- **Graph Neural Networks:** Represent crystals as graphs
- **Crystal graphs:** Atoms are nodes, bonds are edges
- **Message passing:** Aggregate neighbor information
- **Property prediction:** Regression on graph embeddings
- **Transfer learning:** Pre-trained representations

## Next Steps

**Experiencing limitations?** This project pushes Colab to its limits:

- **Tier 1:** [Multi-database materials discovery](../tier-1/) on Studio Lab
  - Cache 10GB of data (download once, use forever)
  - Ensemble GNN models from multiple databases (5-6 hours)
  - High-throughput screening (1,000+ materials)
  - Persistent environments and checkpoints
  - No session timeouts

- **Tier 2:** [AWS-integrated workflows](../tier-2/) with S3 and SageMaker
  - Store 100GB+ materials data on S3
  - Distributed training with SageMaker
  - Managed hyperparameter tuning

- **Tier 3:** [Production-scale discovery](../tier-3/) with full CloudFormation
  - Million+ materials from multiple databases
  - DFT validation on AWS Batch
  - Real-time discovery pipeline

## Requirements

Pre-installed in Colab and Studio Lab:
- Python 3.9+, PyTorch/TensorFlow
- pymatgen, torch-geometric
- pandas, numpy, scikit-learn
- matplotlib, seaborn

**Note:** First run downloads 1.5GB of data (15-20 minutes)
