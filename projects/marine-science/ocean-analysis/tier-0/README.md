# Ocean Species Classification from Underwater Imagery

**Duration:** 60-90 minutes
**Platform:** Google Colab or SageMaker Studio Lab
**Cost:** $0 (no AWS account needed)
**Data:** ~1.5GB underwater imagery (plankton, fish)

## Research Goal

Train a convolutional neural network to classify marine species from underwater imagery, focusing on plankton and fish identification for biodiversity monitoring and ocean health assessment.

## Quick Start

### Run in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/marine-science/ocean-analysis/tier-0/ocean-species-classification.ipynb)

### Run in SageMaker Studio Lab
[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/marine-science/ocean-analysis/tier-0/ocean-species-classification.ipynb)

## What You'll Build

1. **Download underwater imagery** (~1.5GB from public datasets, takes 15-20 min)
2. **Preprocess images** (resizing, normalization, augmentation)
3. **Train CNN classifier** (60-75 minutes on GPU)
4. **Evaluate performance** (accuracy, confusion matrix, F1-score)
5. **Visualize predictions** (sample classifications, attention maps)

## Dataset

**NOAA Plankton & Fish Imagery**
- Source: Combined from NOAA Fisheries, Kaggle Plankton datasets
- Species: 10-15 common categories (copepods, diatoms, fish larvae, jellyfish, etc.)
- Images: ~50,000 underwater photos
- Resolution: Variable (128x128 to 512x512 pixels)
- Size: ~1.5GB compressed JPG files
- Format: Labeled image directories by species

## Colab Considerations

This notebook works on Colab but you'll notice:
- **15-20 minute download** at session start (no persistence)
- **60-75 minute training** (close to timeout limit)
- **Re-download required** if session disconnects
- **~10GB RAM usage** (near Colab's limit)

These limitations become important for real research workflows.

## What's Included

- Single Jupyter notebook (`ocean-species-classification.ipynb`)
- Image data downloading utilities
- CNN architecture (ResNet18/MobileNet transfer learning)
- Training and evaluation pipeline
- Species prediction visualization

## Key Methods

- **Transfer learning:** Fine-tune pre-trained ImageNet models
- **Data augmentation:** Rotation, flipping, color jittering
- **Class balancing:** Handle imbalanced species distributions
- **Confusion matrix:** Identify commonly confused species
- **Grad-CAM visualization:** Understand model attention

## Next Steps

**Experiencing limitations?** This project pushes Colab to its limits:

- **Tier 1:** [Multi-sensor ocean monitoring](../tier-1/) on Studio Lab
  - Cache 10GB of multi-modal data (download once, use forever)
  - Train ensemble models (5-6 hours continuous)
  - Persistent environments and checkpoints
  - No session timeouts

- **Tier 2:** [AWS-integrated workflows](../tier-2/) with S3 and SageMaker
  - Store 100GB+ ocean data on S3
  - Distributed preprocessing with Lambda
  - Managed training jobs with hyperparameter tuning

- **Tier 3:** [Production-scale analysis](../tier-3/) with full CloudFormation
  - Real-time species detection from autonomous vehicles
  - Distributed inference clusters
  - Integration with ocean monitoring systems

## Requirements

Pre-installed in Colab and Studio Lab:
- Python 3.9+, PyTorch or TensorFlow
- torchvision, pillow, scikit-learn
- matplotlib, seaborn, numpy

**Note:** First run downloads 1.5GB of data (15-20 minutes)
