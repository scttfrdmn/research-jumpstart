# Chest X-ray Disease Classification with Deep Learning

**Duration:** 60-90 minutes
**Platform:** Google Colab or SageMaker Studio Lab
**Cost:** $0 (no AWS account needed)
**Data:** ~1.5GB NIH ChestX-ray14 subset

## Research Goal

Train a ResNet-based deep learning model to classify 14 thoracic diseases from chest X-ray images using the NIH ChestX-ray14 dataset. Learn multi-label classification for medical imaging.

## Quick Start

### Run in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/medical/disease-prediction/tier-0/chest-xray-classification.ipynb)

### Run in SageMaker Studio Lab
[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/medical/disease-prediction/tier-0/chest-xray-classification.ipynb)

## What You'll Build

1. **Download NIH ChestX-ray14 subset** (~1.5GB, 5,000 images, takes 15-20 min)
2. **Preprocess medical images** (resizing, normalization, augmentation)
3. **Train ResNet-18 classifier** (60-75 minutes on GPU)
4. **Evaluate disease predictions** (AUC-ROC, sensitivity, specificity)
5. **Visualize attention maps** (GradCAM for interpretability)

## Dataset

**NIH ChestX-ray14 (Curated Subset)**
- Source: National Institutes of Health Clinical Center
- Images: 5,000 chest X-rays (subset of 112,120 full dataset)
- Diseases: 14 classes (Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, Hernia)
- Format: Grayscale PNG, 1024x1024 pixels
- Labels: Multi-label (patients can have multiple diseases)
- Size: ~1.5GB compressed
- Access: Public domain dataset

## Colab Considerations

This notebook works on Colab but you'll notice:
- **15-20 minute download** at session start (no persistence)
- **60-75 minute training** (close to timeout limit)
- **Re-download required** if session disconnects
- **~11GB RAM usage** (near Colab's limit)

These limitations become important for real research workflows.

## What's Included

- Single Jupyter notebook (`chest-xray-classification.ipynb`)
- NIH dataset download utilities
- ResNet-18 architecture for multi-label classification
- Training and evaluation pipeline
- GradCAM visualization for model interpretability
- Clinical metrics (AUC-ROC, sensitivity, specificity)

## Key Methods

- **Multi-label classification:** Predict multiple diseases per image
- **Transfer learning:** Pre-trained ImageNet weights
- **Data augmentation:** Rotation, flipping, brightness adjustment
- **Class imbalance handling:** Weighted loss functions
- **Interpretability:** GradCAM attention maps show where model looks

## Clinical Context

Understanding automated disease detection in chest X-rays:
- **Multi-label challenge:** Patients often have multiple conditions
- **Class imbalance:** Some diseases are rare (e.g., Hernia: 0.2%)
- **High-resolution images:** 1024x1024 pixels contain fine details
- **Interpretability crucial:** Clinicians need to understand model decisions
- **Performance metrics:** AUC-ROC standard in medical imaging

## Next Steps

**Experiencing limitations?** This project pushes Colab to its limits:

- **Tier 1:** [Multi-modal medical imaging ensemble](../tier-1/) on Studio Lab
  - Cache 10GB of multi-modal data (X-ray, CT, MRI) - download once
  - Train ensemble classifiers (5-6 hours continuous)
  - Persistent environments and model checkpoints
  - No session timeouts

- **Tier 2:** [AWS-integrated workflows](../tier-2/) with S3 and SageMaker
  - Store 100GB+ medical imaging datasets on S3
  - Distributed preprocessing with AWS Batch
  - Managed training jobs with hyperparameter tuning
  - Model registry and deployment

- **Tier 3:** [Production clinical AI](../tier-3/) with full CloudFormation
  - Multi-hospital dataset federation (TB-scale)
  - Real-time inference endpoints
  - HIPAA-compliant infrastructure
  - Continuous model monitoring

## Requirements

Pre-installed in Colab and Studio Lab:
- Python 3.9+, PyTorch or TensorFlow
- torchvision, PIL, opencv
- scikit-learn, scipy
- matplotlib, seaborn

**Note:** First run downloads 1.5GB of data (15-20 minutes)

## Ethical Considerations

This is an educational project using publicly available, de-identified data:
- **Not for clinical use:** This model is not FDA-approved
- **Educational purposes only:** Do not use for patient diagnosis
- **Data privacy:** Dataset is fully de-identified
- **Algorithmic bias:** Model may perform differently across demographics
- **Human oversight:** AI assists but never replaces clinical judgment

## Resources

- [NIH ChestX-ray14 Dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- [Original Paper](https://arxiv.org/abs/1705.02315) - Wang et al., 2017
- [Medical Imaging AI Best Practices](https://pubs.rsna.org/doi/10.1148/radiol.2020192224)
