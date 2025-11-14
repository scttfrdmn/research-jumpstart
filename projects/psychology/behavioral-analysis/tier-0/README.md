# EEG-Based Emotion Classification

**Duration:** 60-90 minutes
**Platform:** Google Colab or SageMaker Studio Lab
**Cost:** $0 (no AWS account needed)
**Data:** ~1.5GB EEG signals from emotion recognition study

## Research Goal

Train a CNN to classify emotional states (happiness, sadness, anger, fear, neutral) from raw EEG signals. Use multi-channel time-series data to predict affective states with >75% accuracy.

## Quick Start

### Run in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/psychology/behavioral-analysis/tier-0/eeg-emotion-classification.ipynb)

### Run in SageMaker Studio Lab
[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/psychology/behavioral-analysis/tier-0/eeg-emotion-classification.ipynb)

## What You'll Build

1. **Download EEG data** (~1.5GB from DEAP dataset, takes 15-20 min)
2. **Preprocess signals** (filtering, artifact removal, normalization)
3. **Train CNN classifier** (60-75 minutes on GPU)
4. **Evaluate performance** (accuracy, confusion matrix, per-class metrics)
5. **Analyze patterns** (visualize learned features, channel importance)

## Dataset

**DEAP: Database for Emotion Analysis using Physiological Signals**
- Source: Multi-modal emotional stimulus database
- Participants: 32 subjects
- Channels: 32 EEG channels + 8 peripheral physiological signals
- Emotions: 5 classes (happiness, sadness, anger, fear, neutral)
- Trials: 40 music videos per participant (1-minute clips)
- Sampling rate: 128 Hz
- Size: ~1.5GB preprocessed data
- Format: Python pickle files (.dat)

## Colab Considerations

This notebook works on Colab but you'll notice:
- **20-minute download** at session start (no persistence)
- **60-75 minute training** (close to timeout limit)
- **Re-download required** if session disconnects
- **~11GB RAM usage** (near Colab's limit)

These limitations become important for real research workflows.

## What's Included

- Single Jupyter notebook (`eeg-emotion-classification.ipynb`)
- DEAP dataset access utilities
- CNN architecture for EEG signal classification
- Signal preprocessing pipeline
- Training and evaluation code
- Visualization of learned features

## Key Methods

- **EEG signal processing:** Bandpass filtering (0.5-45 Hz)
- **Artifact removal:** ICA-based eye blink and movement correction
- **Feature extraction:** Raw signals + spectral power bands
- **Deep learning:** 1D CNN for temporal pattern recognition
- **Class balancing:** Weighted loss for imbalanced emotions
- **Cross-validation:** Subject-independent evaluation

## Next Steps

**Experiencing limitations?** This project pushes Colab to its limits:

- **Tier 1:** [Multi-modal affect recognition](../tier-1/) on Studio Lab
  - Cache 10GB of multi-modal data (EEG, face, physiology)
  - Train ensemble models (5-6 hours continuous)
  - Cross-modal fusion requiring persistence
  - No session timeouts

- **Tier 2:** [AWS-integrated workflows](../tier-2/) with S3 and SageMaker
  - Store 100GB+ psychological datasets on S3
  - Distributed preprocessing with Lambda
  - Managed training jobs with hyperparameter tuning

- **Tier 3:** [Production-scale analysis](../tier-3/) with full CloudFormation
  - Real-time emotion recognition API
  - Multi-study meta-analysis (1000+ participants)
  - Distributed model training
  - Clinical deployment infrastructure

## Requirements

Pre-installed in Colab and Studio Lab:
- Python 3.9+, TensorFlow/PyTorch
- NumPy, SciPy, Pandas
- scikit-learn, MNE-Python (EEG library)
- Matplotlib, Seaborn

**Note:** First run downloads 1.5GB of data (15-20 minutes)
