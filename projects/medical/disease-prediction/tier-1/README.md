# Multi-Modal Medical Imaging Ensemble

**Duration:** 4-8 hours
**Platform:** SageMaker Studio Lab ONLY
**Cost:** $0 (free Studio Lab account, no AWS charges)
**Data:** ~10GB multi-modal medical imaging datasets

## Research Goal

Build an ensemble learning system that combines multiple imaging modalities (chest X-rays, CT scans, MRI) to improve disease prediction accuracy. Train multiple deep learning models with persistent storage and checkpointing for production-ready medical AI.

**This project requires Studio Lab** - it won't work on Colab Free due to:
- **10GB multi-modal dataset** (Colab has no persistent storage)
- **5-6 hour continuous training** (Colab disconnects after 90 min idle)
- **Model checkpointing essential** (must resume from checkpoints)
- **Complex environment** (multiple medical imaging libraries)

## What This Enables

Real medical AI research that isn't possible on Colab:

### 🔬 Dataset Persistence
- Download 10GB of multi-modal data **once** (X-ray, CT, MRI)
- Access instantly in all future sessions
- No 20-minute re-downloads every session
- Cache preprocessed images and features

### ⚡ Long-Running Training
- Train 5-6 model ensemble (30-60 min each)
- Total compute: 5-6 hours continuous
- Automatic checkpointing every epoch
- Resume from checkpoint if interrupted

### 🧪 Reproducible Environments
- Conda environment with medical imaging stack
- Persists between sessions
- No reinstalling PyTorch, SimpleITK, nibabel
- Team members use identical setup

### 📊 Iterative Analysis
- Save ensemble predictions
- Build on previous experiments
- Refine models incrementally
- Collaborative development

## What You'll Build

Multi-notebook research workflow:

1. **Data Acquisition & Preprocessing** (60 min)
   - Download 10GB from 3 modalities
   - NIH ChestX-ray14: 8,000 X-rays (4GB)
   - LIDC-IDRI CT scans: 500 cases (3GB)
   - BraTS MRI scans: 300 cases (3GB)
   - Preprocess, normalize, and cache

2. **Model Training** (5-6 hours)
   - Train ResNet-50 on X-rays (90 min)
   - Train 3D CNN on CT scans (120 min)
   - Train U-Net on MRI scans (90 min)
   - Train ensemble meta-learner (45 min)
   - Checkpoint every epoch

3. **Ensemble Integration** (45 min)
   - Combine predictions from all modalities
   - Weighted voting and stacking
   - Cross-validation
   - Uncertainty quantification

4. **Clinical Evaluation** (60 min)
   - Per-modality and ensemble performance
   - ROC curves, confusion matrices
   - Sensitivity, specificity, PPV, NPV
   - GradCAM interpretability
   - Publication-ready figures

## Datasets

### Chest X-rays: NIH ChestX-ray14
- **Size:** 8,000 images (4GB subset)
- **Diseases:** 14 thoracic pathologies
- **Format:** PNG, 1024x1024 grayscale
- **Source:** NIH Clinical Center

### CT Scans: LIDC-IDRI
- **Size:** 500 cases (3GB subset)
- **Target:** Lung nodule detection
- **Format:** DICOM, 512x512x~150 slices
- **Source:** Lung Image Database Consortium

### MRI Scans: BraTS
- **Size:** 300 cases (3GB subset)
- **Target:** Brain tumor segmentation
- **Format:** NIfTI, 240x240x155 volumes
- **Modalities:** T1, T1ce, T2, FLAIR

**Total:** ~10GB cached in Studio Lab's 15GB persistent storage

## Setup

### 1. Create Studio Lab Account
1. Go to [SageMaker Studio Lab](https://studiolab.sagemaker.aws)
2. Request free account (approval in 1-2 days)
3. No credit card or AWS account needed

### 2. Import Project
```bash
# In Studio Lab terminal
git clone https://github.com/YOUR_USERNAME/research-jumpstart.git
cd research-jumpstart/projects/medical/disease-prediction/tier-1
```

### 3. Install Dependencies
```bash
# Create conda environment (persists between sessions!)
conda env create -f environment.yml
conda activate medical-imaging-lab

# Or use pip
pip install -r requirements.txt
```

### 4. Run Notebooks in Order
1. `01_data_preparation.ipynb` - Download and cache all datasets
2. `02_xray_model.ipynb` - Train X-ray classifier
3. `03_ct_model.ipynb` - Train CT nodule detector
4. `04_mri_model.ipynb` - Train MRI segmentation model
5. `05_ensemble_integration.ipynb` - Combine models
6. `06_clinical_evaluation.ipynb` - Comprehensive evaluation

## Key Features

### Persistence Example
```python
# Save model checkpoint (persists between sessions!)
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'auc_score': auc_score,
}, 'saved_models/xray_resnet50_epoch10.pth')

# Resume in next session
checkpoint = torch.load('saved_models/xray_resnet50_epoch10.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

### Multi-Modal Processing
```python
# Import from project modules
from src.data_utils import load_xray, load_ct_scan, load_mri
from src.models import XRayClassifier, CTNoduleDetector, MRISegmenter
from src.ensemble import EnsemblePredictor

# Load different modalities
xray = load_xray('data/xrays/patient_001.png')
ct_scan = load_ct_scan('data/ct/patient_001.dcm')
mri_scan = load_mri('data/mri/patient_001.nii.gz')

# Ensemble prediction
ensemble = EnsemblePredictor(models=[xray_model, ct_model, mri_model])
combined_prediction = ensemble.predict([xray, ct_scan, mri_scan])
```

## Project Structure

```
tier-1/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment
│
├── notebooks/
│   ├── 01_data_preparation.ipynb      # Download all datasets
│   ├── 02_xray_model.ipynb           # X-ray classifier training
│   ├── 03_ct_model.ipynb             # CT nodule detector training
│   ├── 04_mri_model.ipynb            # MRI segmentation training
│   ├── 05_ensemble_integration.ipynb # Combine models
│   └── 06_clinical_evaluation.ipynb  # Final evaluation
│
├── src/
│   ├── __init__.py
│   ├── data_utils.py                 # Data loading for all modalities
│   ├── preprocessing.py              # Image preprocessing
│   ├── models.py                     # Model architectures
│   ├── ensemble.py                   # Ensemble methods
│   ├── evaluation.py                 # Clinical metrics
│   └── visualization.py              # GradCAM, plots
│
├── data/                             # Persistent data storage (gitignored)
│   ├── xrays/                       # NIH ChestX-ray14 (4GB)
│   ├── ct/                          # LIDC-IDRI CT scans (3GB)
│   ├── mri/                         # BraTS MRI scans (3GB)
│   ├── processed/                   # Preprocessed data
│   └── README.md                    # Data documentation
│
└── saved_models/                     # Model checkpoints (gitignored)
    ├── xray_resnet50.pth
    ├── ct_3dcnn.pth
    ├── mri_unet.pth
    ├── ensemble_meta.pth
    └── README.md                     # Model documentation
```

## Why This Needs Studio Lab

| Requirement | Colab Free | Studio Lab |
|-------------|------------|------------|
| **10GB multi-modal data** | ❌ No storage | ✅ 15GB persistent |
| **5-6 hour training** | ❌ 90 min limit | ✅ 12 hour sessions |
| **Model checkpoints** | ❌ Lost on disconnect | ✅ Persists forever |
| **Medical imaging stack** | ❌ Reinstall each time | ✅ Conda persists |
| **Resume experiments** | ❌ Start from scratch | ✅ Pick up where you left off |
| **DICOM/NIfTI support** | ❌ Complex setup | ✅ Pre-configured environment |

**Bottom line:** Multi-modal medical AI research is not viable on Colab Free.

## Time Estimate

**First Run:**
- Setup: 15 minutes (one-time)
- Data download: 60 minutes (one-time, ~10GB)
- Environment setup: 20 minutes (one-time)
- Model training: 5-6 hours
- Evaluation: 1-2 hours
- **Total: 7-9 hours**

**Subsequent Runs:**
- Data: Instant (cached)
- Environment: Instant (persisted)
- Training: 5-6 hours (or resume from checkpoint)
- **Total: 5-6 hours**

You can pause and resume at any time!

## Clinical Performance Targets

### Individual Models
- **X-ray classifier:** Mean AUC > 0.85 across 14 diseases
- **CT nodule detector:** Sensitivity > 0.90, FP/scan < 4
- **MRI segmenter:** Dice coefficient > 0.85

### Ensemble Model
- **Target:** 5-10% improvement over single modality
- **Expected:** Mean AUC > 0.90 across all tasks
- **Interpretability:** Per-modality contribution scores

## Advanced Features

### Multi-Modal Fusion Strategies
1. **Early fusion:** Concatenate features from all modalities
2. **Late fusion:** Average predictions from each model
3. **Learned fusion:** Meta-learner weights each modality
4. **Attention fusion:** Dynamic weighting based on input

### Clinical Validation
- **Cross-validation:** 5-fold stratified CV
- **External validation:** Test on held-out hospital data
- **Subgroup analysis:** Performance by age, gender, ethnicity
- **Confidence calibration:** Ensure predicted probabilities are accurate

### Model Interpretability
- **GradCAM:** Attention maps for each modality
- **Feature importance:** Which modality contributes most
- **Uncertainty quantification:** Confidence intervals
- **Failure analysis:** Where and why models fail

## Ethical Considerations

### Data Privacy
- All datasets are publicly available and de-identified
- No patient information included
- Complies with HIPAA de-identification standards

### Algorithmic Bias
- Evaluate performance across demographic groups
- Test for disparate impact
- Document known limitations
- Report confidence intervals

### Clinical Deployment
- This is educational/research use only
- Not FDA-approved for clinical practice
- Requires extensive validation before deployment
- Human oversight essential for all predictions

## Next Steps

After mastering multi-modal medical AI:

- **Tier 2:** AWS integration (S3, SageMaker, Batch) - $20-50
- **Tier 3:** Production HIPAA-compliant infrastructure - $200-500/month

## Resources

### Medical Imaging Datasets
- [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- [LIDC-IDRI CT Database](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)
- [BraTS MRI Dataset](https://www.med.upenn.edu/cbica/brats/)

### Medical AI Guidelines
- [FDA AI/ML Guidance](https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-software-medical-device)
- [RSNA AI Best Practices](https://pubs.rsna.org/doi/10.1148/radiol.2020192224)
- [WHO Ethics Guidelines](https://www.who.int/publications/i/item/9789240029200)

### Technical Resources
- [Medical Imaging with Deep Learning](https://arxiv.org/abs/1702.05747)
- [PyTorch Medical Imaging](https://pytorch.org/vision/stable/models.html)
- [MONAI Framework](https://monai.io/) - Medical Open Network for AI

## Troubleshooting

### Environment Issues
```bash
# Reset environment
conda env remove -n medical-imaging-lab
conda env create -f environment.yml
```

### Storage Full
```bash
# Check usage
du -sh data/ saved_models/

# Clean old checkpoints
rm -rf saved_models/xray_resnet50_epoch[1-9].pth

# Archive old experiments
tar -czf archived_data.tar.gz data/processed/old_*
```

### DICOM/NIfTI Issues
```bash
# Install additional libraries if needed
pip install pydicom SimpleITK nibabel
```

### GPU Memory Issues
```python
# Reduce batch size in training scripts
batch_size = 16  # Instead of 32
```

### Session Timeout
Data and models persist! Just restart and continue where you left off.

---

**🤖 Built for medical AI research with [Claude Code](https://claude.com/claude-code)**
