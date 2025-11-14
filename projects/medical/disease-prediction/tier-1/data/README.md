# Medical Imaging Data Directory

This directory stores all medical imaging datasets used in the multi-modal ensemble project.

## Directory Structure

```
data/
├── xrays/              # NIH ChestX-ray14 dataset (~4GB)
│   ├── images/         # PNG chest X-ray images
│   └── metadata.csv    # Disease labels and patient info
│
├── ct/                 # LIDC-IDRI CT scans (~3GB)
│   ├── scans/          # DICOM CT series
│   └── annotations.csv # Nodule annotations
│
├── mri/                # BraTS MRI dataset (~3GB)
│   ├── volumes/        # NIfTI brain MRI volumes
│   └── segmentations/  # Ground truth tumor masks
│
├── processed/          # Preprocessed data cache
│   ├── xray_features/  # Extracted X-ray features
│   ├── ct_patches/     # CT patches for training
│   └── mri_slices/     # MRI 2D slices
│
└── splits/             # Train/val/test splits
    ├── xray_train.csv
    ├── xray_val.csv
    ├── xray_test.csv
    ├── ct_train.csv
    ├── ct_val.csv
    ├── ct_test.csv
    ├── mri_train.csv
    ├── mri_val.csv
    └── mri_test.csv
```

## Datasets

### 1. NIH ChestX-ray14
- **Source:** https://nihcc.app.box.com/v/ChestXray-NIHCC
- **Size:** 8,000 images (subset), ~4GB
- **Format:** PNG, 1024x1024 grayscale
- **Diseases:** 14 thoracic pathologies
- **License:** Public domain (CC0)

### 2. LIDC-IDRI CT Scans
- **Source:** https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI
- **Size:** 500 cases (subset), ~3GB
- **Format:** DICOM, 512x512x~150 slices
- **Target:** Lung nodule detection and classification
- **License:** Creative Commons Attribution 3.0

### 3. BraTS MRI Scans
- **Source:** https://www.med.upenn.edu/cbica/brats/
- **Size:** 300 cases (subset), ~3GB
- **Format:** NIfTI (.nii.gz), 240x240x155 volumes
- **Modalities:** T1, T1ce, T2, FLAIR
- **Target:** Brain tumor segmentation
- **License:** Attribution-NonCommercial 4.0

## Download Instructions

### Method 1: Automated Script (Recommended)
```bash
# Run from tier-1 directory
python src/download_data.py --all

# Or download specific datasets
python src/download_data.py --xray
python src/download_data.py --ct
python src/download_data.py --mri
```

### Method 2: Manual Download

#### NIH ChestX-ray14
```bash
cd data/xrays
wget https://nihcc.app.box.com/shared/static/subset.zip
unzip subset.zip
rm subset.zip
```

#### LIDC-IDRI CT
1. Visit: https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI
2. Download NBIA Data Retriever
3. Download manifest file
4. Extract to `data/ct/scans/`

#### BraTS MRI
1. Register at: https://www.med.upenn.edu/cbica/brats/
2. Download training data
3. Extract to `data/mri/volumes/`

## Data Preprocessing

All preprocessing is handled automatically by the notebooks:

1. **`01_data_preparation.ipynb`** - Downloads and preprocesses all datasets
   - Resizing and normalization
   - Data augmentation
   - Train/val/test splitting
   - Feature extraction and caching

## Storage Requirements

- **Total raw data:** ~10GB
- **Processed data:** ~2GB
- **Total:** ~12GB (fits in Studio Lab's 15GB storage)

## Data Organization

### Patient IDs
- X-ray: `xray_patient_XXXXX`
- CT: `ct_patient_XXXXX`
- MRI: `mri_patient_XXXXX`

### File Naming Conventions
```
# X-rays
xrays/images/00000001_001.png

# CT scans
ct/scans/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.*.dcm

# MRI volumes
mri/volumes/BraTS20_Training_001/BraTS20_Training_001_t1.nii.gz
```

## Privacy and Ethics

- All datasets are publicly available and fully de-identified
- No patient identifiable information (PII) included
- Complies with HIPAA de-identification standards
- For educational and research use only
- Not for clinical diagnosis or treatment

## Gitignore

This directory is excluded from git (see `.gitignore`):
- Large binary files not suitable for version control
- Data downloaded locally on each machine
- Processed data regenerated from raw data

## Troubleshooting

### Disk Space Issues
```bash
# Check current usage
du -sh data/*

# Remove processed cache (will regenerate)
rm -rf data/processed/*

# Remove specific dataset
rm -rf data/ct/*
```

### Corrupted Downloads
```bash
# Verify file integrity
python src/verify_data.py

# Re-download corrupted files
python src/download_data.py --verify --redownload
```

### Permission Issues
```bash
# Fix permissions
chmod -R u+w data/
```

## Citation

If you use these datasets in research, please cite:

**NIH ChestX-ray14:**
```
Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: Hospital-scale
Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and
Localization of Common Thorax Diseases. IEEE CVPR 2017.
```

**LIDC-IDRI:**
```
Armato III SG, et al. The Lung Image Database Consortium (LIDC) and Image
Database Resource Initiative (IDRI): A Completed Reference Database of Lung
Nodules on CT Scans. Medical Physics 2011; 38(2):915-931.
```

**BraTS:**
```
Menze BH, et al. The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS).
IEEE Transactions on Medical Imaging 2015; 34(10):1993-2024.
```

---

**Last updated:** 2025-11-13
