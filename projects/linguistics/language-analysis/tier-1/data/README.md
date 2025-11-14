# Linguistic Data Storage

This directory stores downloaded and processed dialect corpora. Data persists between SageMaker Studio Lab sessions!

## Directory Structure

```
data/
├── raw/                    # Original downloaded files
│   ├── english_dialects/
│   ├── spanish_dialects/
│   ├── mandarin_dialects/
│   ├── arabic_dialects/
│   └── german_dialects/
│
└── processed/              # Cleaned and processed files
    ├── features/           # Extracted linguistic features
    ├── annotations/        # Dialect labels
    └── splits/             # Train/validation/test splits
```

## Datasets

### English Dialects
- **Source:** IDEA (International Dialects of English Archive)
- **URL:** https://www.dialectsarchive.com/
- **Dialects:** US (General American, Southern, New York, etc.), UK (RP, Cockney, etc.), Australian
- **Size:** ~2GB
- **Modalities:** Speech audio + transcriptions

### Spanish Dialects
- **Source:** COSER (Corpus Oral y Sonoro del Español Rural)
- **Dialects:** Peninsular, Latin American varieties (Mexican, Argentine, etc.)
- **Size:** ~2GB
- **Modalities:** Speech audio + transcriptions

### Mandarin Dialects
- **Source:** Public dialectology corpora
- **Dialects:** Standard Mandarin, regional varieties (Beijing, Shanghai, etc.)
- **Size:** ~2GB
- **Modalities:** Speech audio + transcriptions

### Arabic Dialects
- **Source:** Multi-Dialect Arabic Parallel Corpus
- **Dialects:** MSA, Egyptian, Levantine, Gulf, Maghrebi
- **Size:** ~2GB
- **Modalities:** Speech audio + text

### German Dialects
- **Source:** Public German dialectology resources
- **Dialects:** Standard German, regional varieties (Bavarian, Saxon, etc.)
- **Size:** ~2GB
- **Modalities:** Speech audio + transcriptions

## Usage

Data is automatically downloaded and cached on first use:

```python
from src.data_utils import load_dialect_corpus

# First run: downloads and caches
corpus = load_dialect_corpus('english')  # Downloads ~2GB

# Subsequent runs: uses cache
corpus = load_dialect_corpus('english')  # Instant!

# Force re-download
corpus = load_dialect_corpus('english', force_download=True)
```

## Storage Management

Check current usage:
```bash
du -sh data/
```

Clean old files:
```bash
rm -rf data/raw/*.old
rm -rf data/processed/*.backup
```

## Persistence

- **Persistent:** This directory survives Studio Lab session restarts
- **15GB Limit:** Studio Lab provides 15GB persistent storage
- **Shared:** All notebooks in this project share this data directory

## Notes

- Audio files stored in WAV format for compatibility
- Text files in UTF-8 encoding
- Features extracted to NumPy arrays
- Raw files preserved for reproducibility
- Processed files optimized for analysis
- .gitignore excludes data/ from version control
