# Artifact Classification with Deep Learning

**Duration:** 60-90 minutes
**Platform:** Google Colab or SageMaker Studio Lab
**Cost:** $0 (no AWS account needed)
**Data:** ~1.5GB artifact imagery

## Research Goal

Train a deep learning model to classify archaeological artifacts (pottery, tools, ornaments) from images using convolutional neural networks (CNNs). Learn to handle imbalanced datasets and transfer learning techniques common in archaeological research.

## Quick Start

### Run in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/archaeology/site-analysis/tier-0/artifact-classification.ipynb)

### Run in SageMaker Studio Lab
[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/archaeology/site-analysis/tier-0/artifact-classification.ipynb)

## What You'll Build

1. **Download artifact imagery** (~1.5GB from archaeological databases, takes 15-20 min)
2. **Preprocess images** (normalization, augmentation, train/test split)
3. **Train CNN classifier** (60-75 minutes on GPU)
4. **Evaluate model performance** (accuracy, precision, recall, confusion matrix)
5. **Generate predictions** (classify new artifact images)

## Dataset

**Archaeological Artifact Image Database**
- Categories: Pottery, Stone Tools, Metal Objects, Ornaments, Bone Tools
- Period: Mixed (Neolithic to Medieval)
- Images: ~5,000 high-resolution artifact photos
- Size: ~1.5GB JPEG files
- Source: Public archaeological databases and museum collections
- Annotation: Expert-validated classifications

## Colab Considerations

This notebook works on Colab but you'll notice:
- **20-minute download** at session start (no persistence)
- **60-75 minute training** (close to timeout limit)
- **Re-download required** if session disconnects
- **~10GB RAM usage** (near Colab's limit)

These limitations become important for real research workflows.

## What's Included

- Single Jupyter notebook (`artifact-classification.ipynb`)
- Image preprocessing utilities
- CNN architecture (ResNet-based transfer learning)
- Training and evaluation pipeline
- Visualization of predictions and confusion matrices

## Key Methods

- **Convolutional Neural Networks:** Deep learning for image classification
- **Transfer learning:** Pre-trained models (ResNet, VGG) fine-tuned on artifacts
- **Data augmentation:** Rotation, flipping, color jittering to increase dataset size
- **Class imbalance handling:** Weighted loss functions for rare artifact types
- **Uncertainty quantification:** Prediction confidence scores

## Next Steps

**Experiencing limitations?** This project pushes Colab to its limits:

- **Tier 1:** [Multi-site archaeological ensemble analysis](../tier-1/) on Studio Lab
  - Cache 8-12GB of data (download once, use forever)
  - Train ensemble models (4-8 hours continuous)
  - Persistent environments and checkpoints
  - No session timeouts

- **Tier 2:** [AWS-integrated workflows](../tier-2/) with S3 and SageMaker
  - Store 50GB+ artifact imagery on S3
  - Distributed preprocessing with Lambda
  - Managed training jobs
  - Model versioning and deployment

- **Tier 3:** [Production-scale analysis](../tier-3/) with full CloudFormation
  - Multi-site datasets (100GB+)
  - Distributed processing with SageMaker
  - Real-time artifact identification API
  - Integration with archaeological databases

## Requirements

Pre-installed in Colab and Studio Lab:
- Python 3.9+, TensorFlow/PyTorch
- torchvision, PIL, opencv-python
- scikit-learn, scipy, matplotlib

**Note:** First run downloads 1.5GB of data (15-20 minutes)
