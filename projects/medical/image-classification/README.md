# Medical Image Classification - Tier 2 Complete

**Duration:** 2-3 days | **Platform:** SageMaker/Unified Studio | **Cost:** $10-20

Deep learning for medical image classification with CNN architectures, transfer learning, and clinical evaluation.

## Overview

Production-ready medical imaging AI using convolutional neural networks (CNNs) for disease detection. Covers data preprocessing, model training, evaluation with clinical metrics, and deployment considerations.

## What You'll Learn

- Medical image preprocessing and augmentation
- CNN architectures (ResNet, EfficientNet, Vision Transformers)
- Transfer learning from ImageNet
- Class imbalance handling
- Clinical performance metrics (sensitivity, specificity, AUC)
- Model interpretability (Grad-CAM)
- Regulatory considerations (FDA, CE Mark)

## Dataset

### AWS Open Data Registry (Recommended)

Access large-scale medical imaging datasets from AWS for free:

**NIH Chest X-ray14** (s3://nih-chest-xrays)
- 112,120 frontal-view chest X-rays
- 14 disease categories
- Public access, no credentials required
- 45 GB total size

**The Cancer Imaging Archive (TCIA)** (s3://imaging.nci.nih.gov)
- Cancer imaging from clinical trials
- Multiple cancer types and organs
- CT, MRI, PET modalities
- Registration required for some collections

**Medical Segmentation Decathlon** (s3://medicalsegmentation)
- 10 organ segmentation tasks
- Brain, heart, liver, lung, prostate, etc.
- CT and MRI modalities
- Ground truth segmentations included

```python
# Access AWS Open Data
from scripts.aws_data_access import download_sample_images, download_nih_metadata

# Download NIH chest X-rays with pneumonia
download_sample_images(
    output_dir='data/chest_xrays',
    n_samples=100,
    disease_filter='Pneumonia'
)

# Get metadata with disease labels
metadata = download_nih_metadata('data/nih_metadata.csv')
```

See `scripts/aws_data_access.py` for complete examples.

### Sample Dataset

**Chest X-Ray Pneumonia Detection**
- 5,000 chest X-ray images
- Classes: Normal, Pneumonia
- DICOM and PNG formats
- Split: 70% train, 15% val, 15% test

## Methods Covered

- Data augmentation for medical images
- Transfer learning with pre-trained CNNs
- Focal loss for class imbalance
- ROC/AUC analysis
- Grad-CAM visualization
- Cross-validation
- Confidence calibration

## Cost: $10-20 (GPU required)
