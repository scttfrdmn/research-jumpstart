# Multi-Modal Affect Recognition Ensemble

**Duration:** 4-8 hours
**Platform:** SageMaker Studio Lab ONLY
**Cost:** $0 (free Studio Lab account, no AWS charges)
**Data:** ~10GB multi-modal affective computing dataset

## Research Goal

Perform multi-modal emotion recognition by fusing EEG brain signals, facial expressions, and physiological responses (heart rate, skin conductance, respiration). Train ensemble models across modalities to achieve state-of-the-art affect recognition with uncertainty quantification.

**This project requires Studio Lab** - it won't work on Colab Free due to:
- **10GB multi-modal dataset** (Colab has no persistent storage)
- **5-6 hour continuous training** (Colab disconnects after 90 min idle)
- **Cross-modal fusion requiring checkpointing** (must resume from checkpoints)
- **Environment persistence** (complex dependency stack with MNE, OpenCV, etc.)

## What This Enables

Real research that isn't possible on Colab:

### Research Dataset Persistence
- Download 10GB of multi-modal affective data **once**
- Access instantly in all future sessions
- No 20-minute re-downloads every session
- Cache intermediate fusion features

### Long-Running Training
- Train 5 modality-specific models (45-60 min each)
- Cross-modal fusion ensemble (2-3 hours)
- Total compute: 5-8 hours continuous
- Automatic checkpointing every epoch
- Resume from checkpoint if needed

### Reproducible Environments
- Conda environment with specialized packages (MNE, OpenCV, dlib)
- Persists between sessions
- No reinstalling dependencies
- Team members use identical setup

### Iterative Analysis
- Save fusion model results
- Build on previous experiments
- Refine ensemble incrementally
- Collaborative notebook development

## What You'll Build

Multi-notebook research workflow:

1. **Data Acquisition** (60 min)
   - Download multi-modal MAHNOB-HCI dataset (~10GB)
   - EEG signals (32 channels, 256 Hz)
   - Video recordings (facial expressions, 24 fps)
   - Physiological signals (ECG, GSR, respiration, temperature)
   - Cache in persistent storage
   - Synchronize modalities across time

2. **Modality-Specific Feature Extraction** (2-3 hours)
   - EEG: Spectral power bands, connectivity features
   - Facial: Action Units (AUs) via OpenFace, emotion embeddings
   - Physiological: HRV, SCL/SCR, respiratory rate
   - Generate fusion-ready feature matrices
   - Save intermediate representations

3. **Ensemble Model Training** (5-6 hours)
   - Train CNN for EEG signals (60-75 min)
   - Train CNN for facial video (60-75 min)
   - Train LSTM for physiological time series (45-60 min)
   - Early fusion model (raw signal concatenation, 90 min)
   - Late fusion ensemble (decision-level fusion, 60 min)
   - Checkpoint every modality

4. **Cross-Modal Fusion Analysis** (1-2 hours)
   - Compare single-modality vs multi-modal performance
   - Analyze modality importance and complementarity
   - Generate ensemble predictions with uncertainty
   - Subject-independent cross-validation
   - Publication-ready figures

## Datasets

**MAHNOB-HCI-Style Multi-Modal Affective Database**
- **Source:** Multi-modal emotion elicitation database
- **Participants:** 27 subjects
- **Stimuli:** 20 emotional video clips (film excerpts)
- **Duration:** 30-120 seconds per trial
- **EEG:** 32 channels, 256 Hz, 10-20 system
- **Video:** Frontal face recording, 1920x1080, 24 fps
- **Physiological:**
  - ECG (electrocardiogram): 256 Hz
  - GSR (galvanic skin response): 32 Hz
  - Respiration: 32 Hz
  - Skin temperature: 32 Hz
- **Annotations:**
  - Valence: -10 (negative) to +10 (positive)
  - Arousal: -10 (calm) to +10 (excited)
  - Discrete emotions: 9 categories
- **Total size:** ~10GB (preprocessed)
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
cd research-jumpstart/projects/psychology/behavioral-analysis/tier-1
```

### 3. Install Dependencies
```bash
# Create conda environment (persists between sessions!)
conda env create -f environment.yml
conda activate psychology-studio-lab

# Or use pip
pip install -r requirements.txt
```

### 4. Run Notebooks in Order
1. `01_data_acquisition.ipynb` - Download and synchronize multi-modal data
2. `02_feature_extraction.ipynb` - Extract features from each modality
3. `03_single_modality_models.ipynb` - Train individual modality classifiers
4. `04_fusion_ensemble.ipynb` - Build and evaluate fusion models

## Key Features

### Persistence Example
```python
# Save modality-specific features (persists between sessions!)
np.save('data/processed/eeg_features.npy', eeg_features)
np.save('data/processed/facial_features.npy', facial_features)
np.save('data/processed/physio_features.npy', physio_features)

# Load in next session
eeg_features = np.load('data/processed/eeg_features.npy')
facial_features = np.load('data/processed/facial_features.npy')
physio_features = np.load('data/processed/physio_features.npy')
```

### Longer Computations
```python
# Run intensive computations that would timeout on Colab
# Studio Lab: 12-hour GPU sessions, 4-hour CPU sessions
for modality in ['eeg', 'facial', 'physio']:
    model = train_modality_specific_model(modality, epochs=100)
    model.save(f'saved_models/{modality}_model.h5')
```

### Shared Utilities
```python
# Import from project modules (better code organization)
from src.eeg_utils import preprocess_eeg, extract_spectral_features
from src.facial_utils import extract_action_units, detect_facial_landmarks
from src.physio_utils import compute_hrv, extract_scr_features
from src.fusion import early_fusion, late_fusion, hybrid_fusion
```

## Project Structure

```
tier-1/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment
│
├── notebooks/
│   ├── 01_data_acquisition.ipynb      # Download and sync multi-modal data
│   ├── 02_feature_extraction.ipynb    # Extract modality-specific features
│   ├── 03_single_modality_models.ipynb # Train individual classifiers
│   └── 04_fusion_ensemble.ipynb       # Build fusion models
│
├── src/
│   ├── __init__.py
│   ├── eeg_utils.py                   # EEG processing utilities
│   ├── facial_utils.py                # Facial expression analysis
│   ├── physio_utils.py                # Physiological signal processing
│   ├── fusion.py                      # Multi-modal fusion strategies
│   └── visualization.py               # Plotting functions
│
├── data/                              # Persistent data storage (gitignored)
│   ├── raw/                          # Downloaded multi-modal datasets
│   ├── processed/                    # Extracted features
│   └── README.md                     # Data documentation
│
└── saved_models/                      # Model checkpoints (gitignored)
    └── README.md                      # Model documentation
```

## Why This Needs Studio Lab

| Requirement | Colab Free | Studio Lab |
|-------------|------------|------------|
| **10GB multi-modal dataset** | No storage | 15GB persistent |
| **5-8 hour training** | 90 min limit | 12 hour sessions |
| **Checkpointing** | Lost on disconnect | Persists forever |
| **Environment setup** | Reinstall each time | Conda persists |
| **Resume analysis** | Start from scratch | Pick up where you left off |
| **Team sharing** | Copy/paste notebooks | Git integration |

**Bottom line:** Multi-modal fusion research is not viable on Colab Free.

## Time Estimate

**First Run:**
- Setup: 15 minutes (one-time)
- Data download: 60 minutes (one-time, ~10GB)
- Environment setup: 15 minutes (one-time)
- Feature extraction: 2-3 hours
- Model training: 5-6 hours
- Fusion analysis: 1-2 hours
- **Total: 9-12 hours**

**Subsequent Runs:**
- Data: Instant (cached)
- Features: Instant (cached)
- Environment: Instant (persisted)
- Training: 5-6 hours (or resume from checkpoint)
- **Total: 5-6 hours**

You can pause and resume at any time!

## Fusion Strategies

### 1. Early Fusion
Concatenate raw signals or low-level features before classification:
- Input: [EEG signals | Facial pixels | Physio signals]
- Model: Single deep network processing all modalities
- Pros: Learns cross-modal interactions at low level
- Cons: High dimensionality, computationally expensive

### 2. Late Fusion
Train separate models per modality, combine predictions:
- Models: EEG CNN | Facial CNN | Physio LSTM
- Fusion: Weighted average or meta-classifier
- Pros: Modular, interpretable, handles missing modalities
- Cons: Limited cross-modal interaction learning

### 3. Hybrid Fusion
Combine feature-level and decision-level fusion:
- Extract modality-specific features
- Concatenate mid-level representations
- Final classifier with attention mechanism
- Pros: Best of both worlds
- Cons: More complex architecture

## Next Steps

After mastering multi-modal affect recognition:

- **Tier 2:** AWS integration (S3, Lambda, SageMaker) - $5-15
- **Tier 3:** Production deployment with real-time API - $50-500/month

## Resources

- [SageMaker Studio Lab Documentation](https://studiolab.sagemaker.aws/docs)
- [MNE-Python EEG Analysis](https://mne.tools/)
- [OpenFace Facial Analysis](https://github.com/TadasBaltrusaitis/OpenFace)
- [Multi-Modal Affective Computing Review](https://arxiv.org/abs/2107.12790)

## Troubleshooting

### Environment Issues
```bash
# Reset environment
conda env remove -n psychology-studio-lab
conda env create -f environment.yml
```

### Storage Full
```bash
# Check usage
du -sh data/ saved_models/

# Clean old files
rm -rf data/raw/old_*.dat
rm -rf saved_models/checkpoint_*.h5
```

### Session Timeout
Data and models persist! Just restart and continue where you left off.

---

**Built for SageMaker Studio Lab with [Claude Code](https://claude.com/claude-code)**
