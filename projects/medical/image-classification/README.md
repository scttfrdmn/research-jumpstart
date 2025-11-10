# Medical Image Classification (X-ray/CT/MRI)

**Difficulty**: 🟡 Intermediate | **Time**: ⏱️ 3-4 hours (Studio Lab)

Train deep learning models on medical imaging data to classify conditions, detect abnormalities, and assist diagnostic workflows.

## Status

**Studio Lab**: 🚧 Lightweight quickstart (in development)
**Unified Studio**: ⏳ Planned

## Quick Start (Studio Lab)

```bash
git clone https://github.com/research-jumpstart/research-jumpstart.git
cd research-jumpstart/projects/medical/image-classification/studio-lab
conda env create -f environment.yml
conda activate medical-imaging
jupyter notebook quickstart.ipynb
```

## What You'll Learn

- Load and preprocess medical images (DICOM, PNG)
- Image augmentation techniques
- Transfer learning with pre-trained CNNs
- Model training and evaluation
- Grad-CAM visualization for explainability
- Clinical performance metrics (sensitivity, specificity)

## Key Analyses

1. **Data Preparation**
   - DICOM file handling
   - Image normalization and resizing
   - Train/validation/test splits
   - Data augmentation (rotation, flip, zoom)

2. **Model Training**
   - Transfer learning (ResNet, EfficientNet)
   - Fine-tuning strategies
   - Class imbalance handling
   - Training with early stopping

3. **Evaluation**
   - Accuracy, precision, recall, F1-score
   - ROC curves and AUC
   - Confusion matrices
   - Clinical metrics (sensitivity, specificity, PPV, NPV)

4. **Explainability**
   - Grad-CAM heatmaps
   - Attention visualization
   - Model interpretation

## Sample Data

Public chest X-ray dataset:
- Normal vs Pneumonia classification
- 100 sample images
- Pre-split train/val/test sets
- Educational/demonstration purposes

## Ethical Considerations

- HIPAA compliance (production)
- Patient privacy and de-identification
- Model bias and fairness
- Clinical validation requirements
- Regulatory approval (FDA clearance)

## Cost

**Studio Lab**: Free forever (small dataset)
**Unified Studio**: ~$25-50 per model training (GPU instances)

## Use Cases

- Pneumonia detection from chest X-rays
- Brain tumor classification from MRI
- Diabetic retinopathy screening
- Bone fracture detection
- COVID-19 screening

## Resources

- [NIH Chest X-ray Dataset](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community)
- [Grad-CAM Paper](https://arxiv.org/abs/1610.02391)
- [Medical Imaging with Deep Learning](https://www.mdpi.com/journal/diagnostics/special_issues/Deep_Learning_Medical_Imaging)

*Last updated: 2025-11-09*
