# Multi-Language Dialectology Ensemble

**Duration:** 4-8 hours
**Platform:** SageMaker Studio Lab ONLY
**Cost:** $0 (free Studio Lab account, no AWS charges)
**Data:** ~10GB cross-linguistic dialect corpus (50+ dialects)

## Research Goal

Perform large-scale comparative dialectology across multiple languages using ensemble deep learning models. Train specialized classifiers for 50+ dialects across language families, quantify cross-linguistic variation patterns, and build a unified framework for understanding dialect diversity.

**This project requires Studio Lab** - it won't work on Colab Free due to:
- **10GB dataset** (Colab has no persistent storage)
- **5-6 hour continuous training** (Colab disconnects after 90 min idle)
- **Model checkpointing essential** (must resume from checkpoints)
- **Environment persistence** (complex dependency stack)

## What This Enables

Real research that isn't possible on Colab:

### Dataset Persistence
- Download 10GB of cross-linguistic dialect data **once**
- Access instantly in all future sessions
- No 20-minute re-downloads every session
- Cache intermediate processing results

### Long-Running Training
- Train ensemble of 10+ dialect models (30-40 min each)
- Total compute: 5-6 hours continuous
- Automatic checkpointing every epoch
- Resume from checkpoint if needed

### Reproducible Environments
- Conda environment with 20+ packages
- Persists between sessions
- No reinstalling dependencies
- Team members use identical setup

### Iterative Analysis
- Save ensemble analysis results
- Build on previous runs
- Refine models incrementally
- Collaborative notebook development

## What You'll Build

Multi-notebook research workflow:

1. **Data Acquisition** (60 min)
   - Download 50+ dialect corpora (~10GB total)
   - Cache in persistent storage
   - Preprocess and align features
   - Generate training/validation splits

2. **Ensemble Model Training** (5-6 hours)
   - Train transformer classifier for each dialect
   - Transfer learning from multilingual base models
   - Checkpoint every epoch
   - Parallel training workflows

3. **Cross-Linguistic Analysis** (60 min)
   - Compare dialectal variation patterns
   - Quantify phonological/lexical distances
   - Identify universal vs. language-specific features
   - Visualize dialect space embeddings

4. **Results Analysis** (45 min)
   - Compare model performance across languages
   - Identify challenging dialect pairs
   - Quantify classification confidence
   - Publication-ready visualizations

## Datasets

**Cross-Linguistic Dialect Corpus**
- **Languages:** 5+ (English, Spanish, Mandarin, Arabic, German)
- **Dialects:** 50+ regional/social varieties
- **Modalities:** Speech audio + text transcriptions
- **Features:** Phonetic, lexical, syntactic, prosodic
- **Period:** Contemporary data (2010-2024)
- **Resolution:** Utterance-level annotations
- **Total size:** ~10GB audio + text files
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
cd research-jumpstart/projects/linguistics/language-analysis/tier-1
```

### 3. Install Dependencies
```bash
# Create conda environment (persists between sessions!)
conda env create -f environment.yml
conda activate linguistics-studio-lab

# Or use pip
pip install -r requirements.txt
```

### 4. Run Notebooks in Order
1. `01_data_preparation.ipynb` - Download and cache datasets
2. `02_feature_extraction.ipynb` - Extract linguistic features
3. `03_ensemble_training.ipynb` - Build and save classification models
4. `04_cross_linguistic_analysis.ipynb` - Comparative analysis

## Key Features

### Persistence Example
```python
# Save model checkpoint (persists between sessions!)
model.save('saved_models/dialect_classifier_v1.pkl')

# Load in next session
model = load_model('saved_models/dialect_classifier_v1.pkl')
```

### Longer Computations
```python
# Run intensive computations that would timeout on Colab
# Studio Lab: 12-hour GPU sessions, 4-hour CPU sessions
results = train_ensemble_classifiers(n_models=50, epochs=10)
```

### Shared Utilities
```python
# Import from project modules (better code organization)
from src.data_utils import load_dialect_corpus, extract_features
from src.visualization import plot_dialect_space, create_confusion_matrix
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
│   ├── 02_feature_extraction.ipynb    # Linguistic feature extraction
│   ├── 03_ensemble_training.ipynb     # Train classification models
│   └── 04_cross_linguistic_analysis.ipynb  # Comparative dialectology
│
├── src/
│   ├── __init__.py
│   ├── data_utils.py                  # Data loading utilities
│   ├── feature_extraction.py          # Feature extraction functions
│   ├── analysis.py                    # Analysis functions
│   └── visualization.py               # Plotting functions
│
├── data/                              # Persistent data storage (gitignored)
│   ├── raw/                          # Downloaded datasets
│   ├── processed/                    # Cleaned data
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
- Data download: 60 minutes (one-time, ~10GB)
- Environment setup: 10 minutes (one-time)
- Model training: 5-6 hours
- Analysis: 1-2 hours
- **Total: 8-9 hours**

**Subsequent Runs:**
- Data: Instant (cached)
- Environment: Instant (persisted)
- Training: 5-6 hours (or resume from checkpoint)
- **Total: 5-6 hours**

You can pause and resume at any time!

## Next Steps

After mastering Studio Lab:

- **Tier 2:** Introduction to AWS services (S3, Lambda, Athena) - $5-15
- **Tier 3:** Production infrastructure with CloudFormation - $50-500/month

## Resources

- [SageMaker Studio Lab Documentation](https://studiolab.sagemaker.aws/docs)
- [Getting Started Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lab.html)
- [Community Forum](https://github.com/aws/studio-lab-examples)

## Troubleshooting

### Environment Issues
```bash
# Reset environment
conda env remove -n linguistics-studio-lab
conda env create -f environment.yml
```

### Storage Full
```bash
# Check usage
du -sh data/ saved_models/

# Clean old files
rm -rf data/raw/old_*.wav
```

### Session Timeout
Data persists! Just restart and continue where you left off.

---

**Built for SageMaker Studio Lab with [Claude Code](https://claude.com/claude-code)**
