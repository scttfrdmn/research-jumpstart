# Multi-Subject Ensemble Connectivity Analysis

**Duration:** 5-6 hours
**Platform:** SageMaker Studio Lab ONLY
**Cost:** $0 (free Studio Lab account, no AWS charges)
**Data:** ~10GB multi-cohort fMRI data

## Research Goal

Perform multi-subject functional connectivity analysis and train ensemble brain decoders across diverse populations. Analyze functional brain networks, compute group-level connectivity patterns, and build robust classification models that generalize across subjects and acquisition sites.

**This project requires Studio Lab** - it won't work on Colab Free due to:
- **10GB dataset** (Colab has no persistent storage)
- **5-6 hour continuous training** (Colab disconnects after 90 min idle)
- **Model checkpointing essential** (must resume from checkpoints)
- **Environment persistence** (complex neuroimaging stack)

## What This Enables

Real research that isn't possible on Colab:

### Dataset Persistence
- Download 10GB of multi-cohort fMRI data **once**
- Access instantly in all future sessions
- No 20-minute re-downloads every session
- Cache intermediate processing results

### Long-Running Training
- Train ensemble decoders across 50+ subjects (20-30 min each)
- Total compute: 5-6 hours continuous
- Automatic checkpointing every epoch
- Resume from checkpoint if needed

### Reproducible Environments
- Conda environment with 25+ packages
- Persists between sessions
- No reinstalling dependencies
- Team members use identical setup

### Iterative Analysis
- Save connectivity matrices and models
- Build on previous runs
- Refine analyses incrementally
- Collaborative notebook development

## What You'll Build

Multi-notebook research workflow:

1. **Data Acquisition** (45-60 min)
   - Download HCP and ABIDE datasets (~10GB total)
   - Cache in persistent storage
   - Quality control and preprocessing
   - Generate training/validation splits

2. **Functional Connectivity Analysis** (90 min)
   - Extract time series from brain parcellations
   - Compute correlation-based connectivity matrices
   - Graph theory metrics (modularity, efficiency, etc.)
   - Group-level network analysis

3. **Ensemble Decoder Training** (5-6 hours)
   - Train 3D CNN decoders for each subject
   - Transfer learning across subjects
   - Checkpoint every epoch
   - Cross-subject generalization testing

4. **Results Analysis** (45 min)
   - Compare connectivity patterns across groups
   - Evaluate decoder performance
   - Brain network visualizations
   - Publication-ready figures

## Datasets

**Multi-Cohort fMRI Collection**
- **HCP (Human Connectome Project):** 30 subjects, task + resting-state
- **ABIDE (Autism Brain Imaging Data Exchange):** 25 subjects (control + ASD)
- **Modality:** Preprocessed BOLD fMRI
- **Resolution:** 2mm isotropic (MNI space)
- **Total size:** ~10GB compressed NIfTI
- **Storage:** Cached in Studio Lab's 15GB persistent storage

## Setup

### 1. Create Studio Lab Account
1. Go to [SageMaker Studio Lab](https://studiolab.sagemaker.aws)
2. Request free account (approval in 1-2 days)
3. No credit card or AWS account needed

### 2. Import Project
```bash
# In Studio Lab terminal
git clone https://github.com/YOUR_USERNAME/research-jumpstart.git
cd research-jumpstart/projects/neuroscience/brain-imaging/tier-1
```

### 3. Install Dependencies
```bash
# Create conda environment (persists between sessions!)
conda env create -f environment.yml
conda activate neuro-studio-lab

# Or use pip
pip install -r requirements.txt
```

### 4. Run Notebooks in Order
1. `01_data_preparation.ipynb` - Download and cache datasets
2. `02_connectivity_analysis.ipynb` - Functional connectivity mapping
3. `03_ensemble_decoders.ipynb` - Train multi-subject classifiers
4. `04_visualization_results.ipynb` - Generate brain network visualizations

## Key Features

### Persistence Example
```python
# Save connectivity matrices (persists between sessions!)
np.save('data/processed/connectivity_matrix_sub01.npy', conn_matrix)

# Load in next session
conn_matrix = np.load('data/processed/connectivity_matrix_sub01.npy')
```

### Longer Computations
```python
# Run intensive computations that would timeout on Colab
# Studio Lab: 12-hour GPU sessions, 4-hour CPU sessions
for subject in subjects:
    model = train_3d_cnn(subject_data[subject])
    save_checkpoint(model, f'models/decoder_{subject}.h5')
```

### Shared Utilities
```python
# Import from project modules (better code organization)
from src.connectivity import compute_correlation_matrix, graph_metrics
from src.visualization import plot_connectome, plot_brain_network
from src.models import build_3d_cnn, ensemble_predict
```

## Project Structure

```
tier-1/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment
│
├── notebooks/
│   ├── 01_data_preparation.ipynb      # Download and cache data
│   ├── 02_connectivity_analysis.ipynb # Functional connectivity
│   ├── 03_ensemble_decoders.ipynb     # Train classifiers
│   └── 04_visualization_results.ipynb # Brain visualizations
│
├── src/
│   ├── __init__.py
│   ├── data_utils.py                  # Data loading utilities
│   ├── connectivity.py                # Connectivity analysis
│   ├── models.py                      # Deep learning models
│   └── visualization.py               # Brain plotting functions
│
├── data/                              # Persistent data storage (gitignored)
│   ├── raw/                          # Downloaded fMRI data
│   ├── processed/                    # Connectivity matrices
│   └── README.md                     # Data documentation
│
└── saved_models/                      # Model checkpoints (gitignored)
    └── README.md                      # Model documentation
```

## Why This Needs Studio Lab

| Requirement | Colab Free | Studio Lab |
|-------------|------------|------------|
| **10GB dataset** | No storage | 15GB persistent |
| **5-6 hour training** | 90 min limit | 12 hour sessions |
| **Checkpointing** | Lost on disconnect | Persists forever |
| **Environment setup** | Reinstall each time | Conda persists |
| **Resume analysis** | Start from scratch | Pick up where you left off |
| **Team sharing** | Copy/paste notebooks | Git integration |

**Bottom line:** This research workflow is not viable on Colab Free.

## Time Estimate

**First Run:**
- Setup: 15 minutes (one-time)
- Data download: 45-60 minutes (one-time, ~10GB)
- Environment setup: 15 minutes (one-time)
- Connectivity analysis: 90 minutes
- Model training: 5-6 hours
- Visualization: 45 minutes
- **Total: 8-10 hours**

**Subsequent Runs:**
- Data: Instant (cached)
- Environment: Instant (persisted)
- Analysis: 90 minutes
- Training: 5-6 hours (or resume from checkpoint)
- **Total: 6-8 hours**

You can pause and resume at any time!

## Key Methods

- **Functional Connectivity:** Correlation-based brain network analysis
- **Graph Theory:** Network modularity, efficiency, small-worldness
- **3D Convolutional Networks:** Spatial pattern learning in brain volumes
- **Ensemble Learning:** Aggregate predictions across subjects
- **Transfer Learning:** Pre-train on one cohort, fine-tune on another
- **Cross-Site Validation:** Test generalization across scanners

## Research Applications

This workflow enables real neuroscience research:

1. **Clinical Prediction:** Classify neurological/psychiatric disorders
2. **Brain Development:** Track connectivity changes across lifespan
3. **Cognitive Decoding:** Predict behavior from brain networks
4. **Biomarker Discovery:** Identify diagnostic network features
5. **Multi-Site Studies:** Harmonize data from different scanners

## Next Steps

After mastering Studio Lab:

- **Tier 2:** Introduction to AWS services (S3, Batch, SageMaker) - $5-15
- **Tier 3:** Production infrastructure with CloudFormation - $50-500/month

## Resources

- [SageMaker Studio Lab Documentation](https://studiolab.sagemaker.aws/)
- [Nilearn Documentation](https://nilearn.github.io/)
- [Human Connectome Project](https://www.humanconnectome.org/)
- [ABIDE Dataset](http://fcon_1000.projects.nitrc.org/indi/abide/)

## Troubleshooting

### Environment Issues
```bash
# Reset environment
conda env remove -n neuro-studio-lab
conda env create -f environment.yml
```

### Storage Full
```bash
# Check usage
du -sh data/ saved_models/

# Clean old files
rm -rf data/raw/*_backup.nii.gz
```

### Session Timeout
Data persists! Just restart and continue where you left off.

---

**Built for SageMaker Studio Lab with [Claude Code](https://claude.com/claude-code)**
