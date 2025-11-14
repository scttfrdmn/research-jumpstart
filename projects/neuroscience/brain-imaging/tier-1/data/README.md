# Neuroimaging Data Storage

This directory stores downloaded and processed fMRI datasets. Data persists between SageMaker Studio Lab sessions!

## Directory Structure

```
data/
├── raw/                    # Original downloaded NIfTI files
│   ├── HCP/               # Human Connectome Project
│   │   ├── sub-01/
│   │   └── sub-02/
│   └── ABIDE/             # Autism Brain Imaging Data Exchange
│       ├── sub-01/
│       └── sub-02/
│
└── processed/              # Processed connectivity matrices and features
    ├── connectivity/       # Functional connectivity matrices
    │   ├── sub-01_conn.npy
    │   └── sub-02_conn.npy
    └── timeseries/        # Extracted ROI time series
        ├── sub-01_ts.npy
        └── sub-02_ts.npy
```

## Datasets

### Human Connectome Project (HCP)
- **Source:** https://www.humanconnectome.org/
- **Subjects:** 30 (subset from S1200 release)
- **Tasks:** Working memory, motor, language, social
- **Resting-state:** 4 runs × 15 minutes
- **Resolution:** 2mm isotropic MNI space
- **Size:** ~6GB
- **Variables:** BOLD fMRI (preprocessed with ICA-FIX)

### ABIDE (Autism Brain Imaging Data Exchange)
- **Source:** http://fcon_1000.projects.nitrc.org/indi/abide/
- **Subjects:** 25 (12 control, 13 ASD)
- **Modality:** Resting-state fMRI
- **Duration:** 5-10 minutes per subject
- **Resolution:** 3mm isotropic MNI space
- **Size:** ~4GB
- **Variables:** BOLD fMRI (preprocessed with CPAC)

## Usage

Data is automatically downloaded and cached on first use:

```python
from src.data_utils import load_hcp_subject, load_abide_subject

# First run: downloads and caches
subj_data = load_hcp_subject('sub-01')  # Downloads ~200MB

# Subsequent runs: uses cache
subj_data = load_hcp_subject('sub-01')  # Instant!

# Force re-download
subj_data = load_hcp_subject('sub-01', force_download=True)
```

## Connectivity Matrices

Connectivity matrices are saved as NumPy arrays:

```python
# Load pre-computed connectivity
import numpy as np
conn_matrix = np.load('data/processed/connectivity/sub-01_conn.npy')

# Shape: (n_rois, n_rois) - typically (200, 200) or (400, 400)
print(f"Connectivity matrix shape: {conn_matrix.shape}")
```

## Parcellations

Brain parcellations used for ROI extraction:

- **Schaefer 200:** 200 cortical regions (7 networks)
- **Schaefer 400:** 400 cortical regions (7 networks)
- **AAL:** 116 anatomical regions
- **Power 264:** 264 functional nodes

## Storage Management

Check current usage:
```bash
du -sh data/
du -sh data/raw/
du -sh data/processed/
```

Clean old files:
```bash
rm -rf data/raw/*_backup/
rm -rf data/processed/*_old.npy
```

Free up space by removing raw data (keep processed):
```bash
# Only if you don't need to reprocess!
rm -rf data/raw/
```

## Persistence

- **Persistent:** This directory survives Studio Lab session restarts
- **15GB Limit:** Studio Lab provides 15GB persistent storage
- **Shared:** All notebooks in this project share this data directory
- **Version Control:** .gitignore excludes data/ from git

## Processing Pipeline

1. **Raw fMRI** (NIfTI files) → `data/raw/`
2. **ROI extraction** → Time series → `data/processed/timeseries/`
3. **Connectivity** → Correlation matrices → `data/processed/connectivity/`
4. **Features** → For ML models → Used directly in training

## Notes

- NIfTI files are compressed (.nii.gz) to save space
- Connectivity matrices stored in NumPy format for fast loading
- Processed files optimized for analysis
- Keep raw data for reproducibility (if space permits)
- Consider external storage (AWS S3) for Tier 2/3 projects
