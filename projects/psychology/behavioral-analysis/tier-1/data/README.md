# Multi-Modal Affective Data Storage

This directory stores downloaded and processed multi-modal affective computing datasets. Data persists between SageMaker Studio Lab sessions!

## Directory Structure

```
data/
├── raw/                    # Original downloaded files
│   ├── eeg/               # EEG signals (DEAP-style, .dat files)
│   ├── facial/            # Facial video recordings (.mp4, .avi)
│   ├── physiological/     # ECG, GSR, respiration, temperature (.csv)
│   └── annotations/       # Emotion labels and metadata (.csv)
│
└── processed/              # Extracted features and preprocessed data
    ├── eeg_features.npy   # Extracted EEG spectral and connectivity features
    ├── facial_features.npy # Facial Action Units and emotion embeddings
    ├── physio_features.npy # HRV, SCR, respiratory features
    ├── labels.npy          # Emotion labels (valence, arousal, discrete)
    └── synchronized.h5     # Time-synchronized multi-modal dataset
```

## Datasets

### Multi-Modal Affective Dataset (MAHNOB-HCI Style)

**EEG Signals**
- **Source:** 32-channel EEG headset (10-20 system)
- **Sampling rate:** 256 Hz
- **Duration:** 30-120 seconds per trial
- **Size:** ~4GB (preprocessed)
- **Format:** Python pickle (.dat) or HDF5 (.h5)
- **Variables:** Raw EEG signals + extracted spectral features

**Facial Videos**
- **Source:** Frontal face recording during emotion elicitation
- **Resolution:** 1920x1080 or 1280x720
- **Frame rate:** 24 fps
- **Duration:** 30-120 seconds per trial
- **Size:** ~4GB (compressed videos)
- **Format:** MP4, AVI
- **Extracted:** Facial landmarks, Action Units, emotion probabilities

**Physiological Signals**
- **ECG (Electrocardiogram):** 256 Hz, ~500KB per trial
- **GSR (Galvanic Skin Response):** 32 Hz, ~100KB per trial
- **Respiration:** 32 Hz, ~100KB per trial
- **Skin Temperature:** 32 Hz, ~100KB per trial
- **Total size:** ~2GB
- **Format:** CSV or HDF5

**Annotations**
- **Valence:** -10 (negative) to +10 (positive)
- **Arousal:** -10 (calm) to +10 (excited)
- **Discrete emotions:** happiness, sadness, anger, fear, disgust, surprise, neutral, contempt, boredom
- **Format:** CSV with trial IDs and timestamps

### Total Dataset Size
- **Raw data:** ~10GB
- **Processed features:** ~1GB
- **Total:** ~11GB (fits within Studio Lab's 15GB limit)

## Usage

Data is automatically downloaded and cached on first use:

```python
from src.eeg_utils import preprocess_eeg, extract_spectral_features
from src.facial_utils import extract_facial_features
from src.physio_utils import extract_all_physiological_features

# First run: downloads and caches
eeg_data = load_eeg_trial(participant=1, trial=1)  # Downloads ~200MB
facial_video = load_facial_video(participant=1, trial=1)  # Downloads ~150MB
physio_data = load_physiological_data(participant=1, trial=1)  # Downloads ~5MB

# Subsequent runs: uses cache
eeg_data = load_eeg_trial(participant=1, trial=1)  # Instant!

# Extract and save features (persists between sessions)
eeg_features = extract_spectral_features(eeg_data)
np.save('data/processed/eeg_features_p1_t1.npy', eeg_features)

# Load processed features in next session
eeg_features = np.load('data/processed/eeg_features_p1_t1.npy')
```

## Storage Management

Check current usage:
```bash
du -sh data/
du -sh data/raw/
du -sh data/processed/
```

Clean old or temporary files:
```bash
# Remove temporary extraction files
rm -rf data/processed/temp_*.npy

# Remove specific participant data
rm -rf data/raw/eeg/participant_*.dat
```

List largest files:
```bash
find data/ -type f -exec du -h {} + | sort -rh | head -20
```

## Synchronization

Multi-modal data needs temporal alignment:

```python
# Synchronize modalities using timestamps
from src.data_utils import synchronize_modalities

synchronized_data = synchronize_modalities(
    eeg_data, facial_data, physio_data,
    eeg_timestamps, facial_timestamps, physio_timestamps,
    target_sampling_rate=32  # Hz (downsampled for alignment)
)

# Save synchronized dataset
import h5py
with h5py.File('data/processed/synchronized.h5', 'w') as f:
    f.create_dataset('eeg', data=synchronized_data['eeg'])
    f.create_dataset('facial', data=synchronized_data['facial'])
    f.create_dataset('physio', data=synchronized_data['physio'])
    f.create_dataset('labels', data=synchronized_data['labels'])
```

## Persistence

✅ **Persistent:** This directory survives Studio Lab session restarts
✅ **15GB Limit:** Studio Lab provides 15GB persistent storage
✅ **Shared:** All notebooks in this project share this data directory
✅ **Version controlled:** .gitignore excludes data/ from git

## Data Privacy

**Important:** If using real participant data:
- Ensure proper IRB approval and consent
- Anonymize participant identifiers
- Do not share raw data publicly
- Follow GDPR/HIPAA guidelines if applicable
- This demo uses synthetic data for illustration

## Notes

- Features are stored in NumPy (.npy) or HDF5 (.h5) format for fast loading
- Raw files preserved for reproducibility
- Processed files optimized for model training
- Temporal synchronization is critical for multi-modal fusion
- Consider using compression for large datasets (e.g., gzip for HDF5)

## References

- DEAP Dataset: https://www.eecs.qmul.ac.uk/mmv/datasets/deap/
- MAHNOB-HCI: https://mahnob-db.eu/hci-tagging/
- AMIGOS Dataset: http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/
- Facial Action Coding System (FACS): https://www.cs.cmu.edu/~face/facs.htm
