# Multi-Institution Learning Outcomes Ensemble

**Duration:** 4-8 hours
**Platform:** SageMaker Studio Lab ONLY
**Cost:** $0 (free Studio Lab account, no AWS charges)
**Data:** ~10GB multi-institution learning data

## Research Goal

Perform cross-institutional analysis of learning outcomes using ensemble machine learning models. Train models on data from multiple universities to predict learning pathways, transfer patterns, and competency development across diverse educational contexts.

**This project requires Studio Lab** - it won't work on Colab Free due to:
- **10GB dataset** (Colab has no persistent storage)
- **5-6 hour continuous training** (Colab disconnects after 90 min idle)
- **Model checkpointing essential** (must resume from checkpoints)
- **Environment persistence** (complex dependency stack)

## What This Enables

Real research that isn't possible on Colab:

### 🔬 Dataset Persistence
- Download 10GB of multi-institution data **once**
- Access instantly in all future sessions
- No 20-minute re-downloads every session
- Cache intermediate processing results

### ⚡ Long-Running Training
- Train ensemble models across institutions (5-6 hours)
- Total compute: 5-6 hours continuous
- Automatic checkpointing every epoch
- Resume from checkpoint if needed

### 🧪 Reproducible Environments
- Conda environment with 20+ packages
- Persists between sessions
- No reinstalling dependencies
- Team members use identical setup

### 📊 Iterative Analysis
- Save ensemble analysis results
- Build on previous runs
- Refine models incrementally
- Collaborative notebook development

## What You'll Build

Multi-notebook research workflow:

1. **Data Acquisition** (45 min)
   - Download data from 5+ institutions (~10GB total)
   - Cache in persistent storage
   - Harmonize schemas across institutions
   - Generate training/validation splits

2. **Ensemble Model Training** (5-6 hours)
   - Train LSTM/Transformer for each institution
   - Transfer learning across institutions
   - Ensemble model aggregation
   - Checkpoint every epoch
   - Parallel training workflows

3. **Learning Pathway Analysis** (45 min)
   - Generate probabilistic learning pathways
   - Cross-institutional transfer patterns
   - Competency development trajectories
   - Intervention timing optimization

4. **Results Analysis** (30 min)
   - Compare institutional effectiveness
   - Identify common success factors
   - Quantify institutional differences
   - Publication-ready figures

## Datasets

**Multi-Institution Learning Outcomes**
- **Institutions:** 5+ universities (public, private, community colleges)
- **Students:** ~100,000 learners across institutions
- **Programs:** Multiple disciplines (STEM, liberal arts, professional)
- **Variables:** Grades, course sequences, time-to-degree, demographics, engagement
- **Period:** 5 years of longitudinal data (2018-2023)
- **Total size:** ~10GB CSV/Parquet files
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
cd research-jumpstart/projects/education/learning-analytics/tier-1
```

### 3. Install Dependencies
```bash
# Create conda environment (persists between sessions!)
conda env create -f environment.yml
conda activate education-analytics

# Or use pip
pip install -r requirements.txt
```

### 4. Run Notebooks in Order
1. `01_data_preparation.ipynb` - Download and harmonize institutional data
2. `02_exploratory_analysis.ipynb` - Cross-institutional EDA
3. `03_ensemble_training.ipynb` - Train and ensemble models
4. `04_pathway_analysis.ipynb` - Learning pathway prediction
5. `05_intervention_optimization.ipynb` - Optimize intervention strategies

## Key Features

### Persistence Example
```python
# Save model checkpoint (persists between sessions!)
model.save('saved_models/learning_pathway_ensemble_v1.h5')

# Load in next session
model = load_model('saved_models/learning_pathway_ensemble_v1.h5')
```

### Longer Computations
```python
# Run intensive computations that would timeout on Colab
# Studio Lab: 12-hour GPU sessions, 4-hour CPU sessions
results = train_multi_institution_ensemble(
    n_institutions=5,
    n_epochs=50,
    cross_validation_folds=10
)
```

### Shared Utilities
```python
# Import from project modules (better code organization)
from src.data_utils import load_institutional_data, harmonize_schemas
from src.ensemble import train_ensemble_models, cross_institutional_transfer
from src.visualization import create_pathway_diagram, plot_intervention_timing
```

## Project Structure

```
tier-1/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment
│
├── notebooks/
│   ├── 01_data_preparation.ipynb      # Download and harmonize data
│   ├── 02_exploratory_analysis.ipynb  # Cross-institutional EDA
│   ├── 03_ensemble_training.ipynb     # Train ensemble models
│   ├── 04_pathway_analysis.ipynb      # Learning pathway prediction
│   └── 05_intervention_optimization.ipynb # Optimize interventions
│
├── src/
│   ├── __init__.py
│   ├── data_utils.py                  # Data loading and harmonization
│   ├── ensemble.py                    # Ensemble model utilities
│   ├── pathway_analysis.py            # Learning pathway functions
│   └── visualization.py               # Plotting functions
│
├── data/                              # Persistent data storage (gitignored)
│   ├── raw/                          # Downloaded institutional datasets
│   ├── processed/                    # Harmonized data
│   └── README.md                     # Data documentation
│
└── saved_models/                      # Model checkpoints (gitignored)
    └── README.md                      # Model documentation
```

## Why This Needs Studio Lab

| Requirement | Colab Free | Studio Lab |
|-------------|------------|------------|
| **10GB dataset** | ❌ No storage | ✅ 15GB persistent |
| **5-6 hour training** | ❌ 90 min limit | ✅ 12 hour sessions |
| **Checkpointing** | ❌ Lost on disconnect | ✅ Persists forever |
| **Environment setup** | ❌ Reinstall each time | ✅ Conda persists |
| **Resume analysis** | ❌ Start from scratch | ✅ Pick up where you left off |
| **Team sharing** | ❌ Copy/paste notebooks | ✅ Git integration |

**Bottom line:** This research workflow is not viable on Colab Free.

## Time Estimate

**First Run:**
- Setup: 15 minutes (one-time)
- Data download: 45 minutes (one-time, ~10GB)
- Environment setup: 10 minutes (one-time)
- Model training: 5-6 hours
- Analysis: 2-3 hours
- **Total: 8-10 hours**

**Subsequent Runs:**
- Data: Instant (cached)
- Environment: Instant (persisted)
- Training: 5-6 hours (or resume from checkpoint)
- **Total: 5-6 hours**

You can pause and resume at any time!

## Key Research Questions

This project enables you to answer:

1. **Transfer Learning:** Do models trained on one institution generalize to others?
2. **Success Factors:** What learning patterns predict success across institutions?
3. **Intervention Timing:** When should interventions be deployed for maximum effect?
4. **Equity Analysis:** How do outcomes vary by student demographics across institutions?
5. **Pathway Diversity:** What alternative pathways lead to similar outcomes?

## Methodology

### Cross-Institutional Data Harmonization
```python
# Harmonize different institutional schemas
def harmonize_schemas(inst_datasets):
    """
    Map different variable names and scales to common schema
    Handle missing data across institutions
    Ensure temporal alignment
    """
    pass
```

### Ensemble Model Architecture
- **Base models:** Institution-specific LSTM/Transformers
- **Transfer learning:** Fine-tune on target institution
- **Ensemble aggregation:** Weighted voting, stacking
- **Uncertainty quantification:** Confidence intervals per prediction

### Learning Pathway Prediction
- **Sequence modeling:** Predict course sequences and outcomes
- **Counterfactual analysis:** What-if scenarios for interventions
- **Temporal dynamics:** How pathways evolve over semesters

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
conda env remove -n education-analytics
conda env create -f environment.yml
```

### Storage Full
```bash
# Check usage
du -sh data/ saved_models/

# Clean old files
rm -rf data/raw/old_*.csv
```

### Session Timeout
Data persists! Just restart and continue where you left off.

---

**🤖 Built for SageMaker Studio Lab with [Claude Code](https://claude.com/claude-code)**
