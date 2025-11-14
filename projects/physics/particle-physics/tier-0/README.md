# High-Energy Physics Particle Classification

**Duration:** 60-90 minutes
**Platform:** Google Colab or SageMaker Studio Lab
**Cost:** $0 (no AWS account needed)
**Data:** ~1.5GB LHC event data

## Research Goal

Train a convolutional neural network for jet tagging in high-energy physics collision events. Classify particle jets from simulated LHC (Large Hadron Collider) data to distinguish between signal (e.g., top quarks, Higgs bosons) and background processes.

## Quick Start

### Run in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/physics/particle-physics/tier-0/particle-classification.ipynb)

### Run in SageMaker Studio Lab
[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/physics/particle-physics/tier-0/particle-classification.ipynb)

## What You'll Build

1. **Download LHC event data** (~1.5GB simulated collision events, takes 15-20 min)
2. **Preprocess particle features** (kinematic variables, jet reconstruction)
3. **Train CNN for jet tagging** (60-75 minutes on GPU)
4. **Evaluate classifier performance** (ROC curves, signal efficiency)
5. **Analyze physics signatures** (invariant mass distributions, b-tagging)

## Dataset

**Simulated LHC Collision Events**
- Data: Top quark pair production (signal) vs QCD jets (background)
- Format: ROOT files or HDF5 (converted for Python)
- Events: ~500,000 collision events
- Features per jet: pT, eta, phi, mass, b-tagging scores, substructure variables
- Size: ~1.5GB
- Source: Open data from CERN or simulated using Pythia/Delphes

## Colab Considerations

This notebook works on Colab but you'll notice:
- **15-20 minute download** at session start (no persistence)
- **60-75 minute training** (close to timeout limit)
- **Re-download required** if session disconnects
- **~10GB RAM usage** (near Colab's limit)

These limitations become important for real research workflows.

## What's Included

- Single Jupyter notebook (`particle-classification.ipynb`)
- LHC data access utilities
- CNN architecture for jet classification
- Training and evaluation pipeline
- Physics-aware visualization (jet images, mass distributions)

## Key Methods

- **Jet tagging:** Classify particle jets by origin (top, W/Z bosons, QCD)
- **Deep learning:** Convolutional neural networks on jet images
- **Feature engineering:** High-level physics variables (mass, pT, substructure)
- **Signal vs background:** Optimize classifier for discovery sensitivity
- **ROC analysis:** Evaluate trade-offs between signal efficiency and background rejection

## Next Steps

**Experiencing limitations?** This project pushes Colab to its limits:

- **Tier 1:** [Multi-detector ensemble reconstruction](../tier-1/) on Studio Lab
  - Cache 8-12GB from multiple detectors (download once, use forever)
  - Train ensemble models (5-6 hours continuous)
  - Persistent environments and checkpoints
  - No session timeouts

- **Tier 2:** [AWS-integrated workflows](../tier-2/) with S3 and SageMaker
  - Store 100GB+ collision data on S3
  - Distributed preprocessing with Lambda
  - Managed training jobs with hyperparameter tuning

- **Tier 3:** [Production-scale analysis](../tier-3/) with full CloudFormation
  - Process real LHC data (petabyte scale)
  - Distributed computing on batch systems
  - Integration with CERN computing infrastructure

## Requirements

Pre-installed in Colab and Studio Lab:
- Python 3.9+, TensorFlow/PyTorch
- NumPy, pandas, scikit-learn
- matplotlib, seaborn
- Optional: uproot (for ROOT file handling)

**Note:** First run downloads 1.5GB of data (15-20 minutes)
