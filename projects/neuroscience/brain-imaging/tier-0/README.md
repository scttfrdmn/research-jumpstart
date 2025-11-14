# fMRI Brain State Classification

**Duration:** 60-90 minutes
**Platform:** Google Colab or SageMaker Studio Lab
**Cost:** $0 (no AWS account needed)
**Data:** ~1.5GB preprocessed fMRI data

## Research Goal

Train a 3D convolutional neural network to decode brain states from functional MRI (fMRI) scans. Classify cognitive states (e.g., working memory, resting state, visual processing) from blood oxygen level-dependent (BOLD) signal patterns across the brain.

## Quick Start

### Run in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/neuroscience/brain-imaging/tier-0/fmri-brain-states.ipynb)

### Run in SageMaker Studio Lab
[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/neuroscience/brain-imaging/tier-0/fmri-brain-states.ipynb)

## What You'll Build

1. **Download preprocessed fMRI data** (~1.5GB from HCP/ABIDE, takes 15-20 min)
2. **Preprocess 3D brain volumes** (spatial normalization, smoothing)
3. **Train 3D CNN for state decoding** (60-75 minutes on GPU)
4. **Evaluate classification accuracy** (confusion matrix, ROC curves)
5. **Visualize brain activation patterns** (glass brain plots, saliency maps)

## Dataset

**Human Connectome Project (HCP) Task fMRI Subset**
- Participants: 20 subjects
- Tasks: Working memory, motor, language, social cognition
- Modality: BOLD fMRI (preprocessed)
- Resolution: 2mm isotropic (91x109x91 voxels)
- Size: ~1.5GB compressed NIfTI files
- Source: HCP S1200 Release (subset)

**Alternative:** ABIDE (Autism Brain Imaging Data Exchange) resting-state fMRI

## Colab Considerations

This notebook works on Colab but you'll notice:
- **20-minute download** at session start (no persistence)
- **60-75 minute training** (close to timeout limit)
- **Re-download required** if session disconnects
- **~11GB RAM usage** (near Colab's limit with 3D convolutions)

These limitations become important for real research workflows.

## What's Included

- Single Jupyter notebook (`fmri-brain-states.ipynb`)
- fMRI data loading utilities (NIfTI format)
- 3D CNN architecture for brain decoding
- Training and evaluation pipeline
- Brain visualization tools (Nilearn)

## Key Methods

- **3D Convolutional Neural Networks:** Learn spatial patterns in brain volumes
- **BOLD signal analysis:** Model hemodynamic response
- **Spatial smoothing:** Reduce noise while preserving structure
- **Brain decoding:** Classify cognitive states from activity patterns
- **Saliency mapping:** Identify discriminative brain regions

## Next Steps

**Experiencing limitations?** This project pushes Colab to its limits:

- **Tier 1:** [Multi-subject ensemble connectivity analysis](../tier-1/) on Studio Lab
  - Cache 10GB multi-cohort data (download once, use forever)
  - Train ensemble decoders across subjects (5-6 hours continuous)
  - Functional connectivity analysis with persistent results
  - No session timeouts

- **Tier 2:** [AWS-integrated workflows](../tier-2/) with S3 and SageMaker
  - Store 100GB+ neuroimaging data on S3
  - Distributed preprocessing with AWS Batch
  - Managed training jobs with hyperparameter tuning

- **Tier 3:** [Production-scale analysis](../tier-3/) with full CloudFormation
  - Multi-site datasets (1TB+)
  - Distributed processing clusters
  - Real-time brain decoding pipelines

## Requirements

Pre-installed in Colab and Studio Lab:
- Python 3.9+, TensorFlow/PyTorch
- NiBabel, Nilearn, scipy
- scikit-learn, matplotlib

**Note:** First run downloads 1.5GB of data (15-20 minutes)
