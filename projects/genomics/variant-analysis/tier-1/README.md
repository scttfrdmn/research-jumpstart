# Multi-Cohort Genomic Variant Analysis with Ensemble Callers

**Duration:** 4-8 hours
**Platform:** SageMaker Studio Lab ONLY
**Cost:** $0 (free Studio Lab account, no AWS charges)
**Data:** ~10GB multi-cohort genomic data (1000 Genomes subset)

## Research Goal

Perform population-scale variant calling using an ensemble of deep learning models trained on multiple individuals from the 1000 Genomes Project. Build consensus variant calls across diverse genetic backgrounds, quantify inter-model agreement, and identify population-specific variants.

**This project requires Studio Lab** - it won't work on Colab Free due to:
- **10GB dataset** (multiple BAM files, Colab has no persistent storage)
- **5-6 hour continuous training** (ensemble of variant callers, Colab disconnects after 90 min idle)
- **Model checkpointing essential** (must resume from checkpoints)
- **Environment persistence** (complex bioinformatics dependencies)

## What This Enables

Real research that isn't possible on Colab:

### 🔬 Dataset Persistence
- Download 10GB of multi-sample BAM files **once**
- Access instantly in all future sessions
- No 20-minute re-downloads every session
- Cache intermediate pileup tensors and features

### ⚡ Long-Running Training
- Train 6-8 variant caller models (45-60 min each)
- Total compute: 5-6 hours continuous
- Automatic checkpointing every epoch
- Resume from checkpoint if needed

### 🧪 Reproducible Environments
- Conda environment with 15+ bioinformatics packages
- Persists between sessions
- No reinstalling samtools, bcftools, etc.
- Team members use identical setup

### 📊 Iterative Analysis
- Save ensemble predictions
- Build consensus calls incrementally
- Compare population-specific variants
- Collaborative notebook development

## What You'll Build

Multi-notebook research workflow:

1. **Data Acquisition** (45-60 min)
   - Download 8-10 individual BAM files (~10GB total)
   - Cache in persistent storage
   - Index BAM files
   - Download reference genome and truth sets

2. **Feature Engineering** (60-90 min)
   - Generate pileup tensors for all samples
   - Extract variant features (quality, depth, strand bias)
   - Create training/validation splits
   - Cache processed features

3. **Ensemble Model Training** (5-6 hours)
   - Train CNN variant caller for each sample
   - Train gradient boosting models
   - Train random forest classifiers
   - Checkpoint every model
   - Parallel training workflows

4. **Consensus Calling** (45 min)
   - Combine predictions from all models
   - Weighted voting by model confidence
   - Population-specific variant filtering
   - Generate high-confidence VCF files

5. **Results Analysis** (60 min)
   - Compare with GATK/DeepVariant benchmarks
   - Precision/recall/F1 by variant type
   - Population stratification analysis
   - Publication-ready figures

## Datasets

**1000 Genomes Project Multi-Cohort Subset**
- **Samples:** 8-10 individuals from diverse populations
  - CEU (European): NA12878, NA12891, NA12892
  - YRI (African): NA19238, NA19239, NA19240
  - CHB (East Asian): NA18525, NA18526
  - MXL (Mexican): NA19648, NA19649
- **Region:** Chromosome 20 (full chromosome or large subset)
- **Coverage:** 30x whole-genome sequencing
- **Format:** BAM (aligned reads) + BAI (index)
- **Reference:** GRCh37/hg19
- **Total size:** ~10GB (8-10 BAM files @ 1-1.2GB each)
- **Storage:** Cached in Studio Lab's 15GB persistent storage
- **Truth sets:** GIAB high-confidence calls (NA12878)

**Population Diversity:**
- European, African, East Asian, Admixed American ancestry
- Enables population-specific variant analysis
- Tests model generalization across genetic backgrounds

## Setup

### 1. Create Studio Lab Account
1. Go to [SageMaker Studio Lab](https://studiolab.sagemaker.aws)
2. Request free account (approval in 1-2 days)
3. No credit card or AWS account needed

### 2. Import Project
```bash
# In Studio Lab terminal
git clone https://github.com/YOUR_USERNAME/research-jumpstart.git
cd research-jumpstart/projects/genomics/variant-analysis/tier-1
```

### 3. Install Dependencies
```bash
# Create conda environment (persists between sessions!)
conda env create -f environment.yml
conda activate genomics-studio-lab

# Or use pip
pip install -r requirements.txt
```

### 4. Run Notebooks in Order
1. `01_data_preparation.ipynb` - Download and cache BAM files
2. `02_feature_extraction.ipynb` - Generate pileup tensors and features
3. `03_ensemble_training.ipynb` - Train multiple variant caller models
4. `04_consensus_calling.ipynb` - Combine predictions and generate VCF
5. `05_population_analysis.ipynb` - Analyze population-specific variants

## Key Features

### Persistence Example
```python
# Save model checkpoint (persists between sessions!)
model.save('saved_models/variant_caller_NA12878.h5')

# Load in next session
model = keras.models.load_model('saved_models/variant_caller_NA12878.h5')
```

### Longer Computations
```python
# Run intensive computations that would timeout on Colab
# Studio Lab: 12-hour GPU sessions, 4-hour CPU sessions
ensemble_predictions = train_ensemble_callers(
    samples=samples,
    epochs=50,
    checkpoint_dir='checkpoints/'
)
```

### Shared Utilities
```python
# Import from project modules (better code organization)
from src.data_utils import load_bam_files, generate_pileup_tensor
from src.analysis import call_variants, calculate_metrics
from src.visualization import plot_variant_density, plot_population_comparison
```

## Project Structure

```
tier-1/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment
│
├── notebooks/
│   ├── 01_data_preparation.ipynb      # Download and cache BAM files
│   ├── 02_feature_extraction.ipynb    # Generate pileup tensors
│   ├── 03_ensemble_training.ipynb     # Train variant caller models
│   ├── 04_consensus_calling.ipynb     # Combine predictions
│   └── 05_population_analysis.ipynb   # Population-specific analysis
│
├── src/
│   ├── __init__.py
│   ├── data_utils.py                  # BAM loading, pileup generation
│   ├── analysis.py                    # Variant calling, metrics
│   └── visualization.py               # Plotting functions
│
├── data/                              # Persistent data storage (gitignored)
│   ├── raw/                          # Downloaded BAM files
│   │   ├── NA12878.chr20.bam
│   │   ├── NA19238.chr20.bam
│   │   └── ... (8-10 BAM files)
│   ├── processed/                    # Cached pileup tensors
│   └── reference/                    # Reference genome, truth sets
│       ├── chr20.fa
│       └── GIAB_truth.vcf.gz
│
└── saved_models/                      # Model checkpoints (gitignored)
    ├── variant_caller_NA12878.h5
    ├── variant_caller_NA19238.h5
    └── ... (6-8 models)
```

## Why This Needs Studio Lab

| Requirement | Colab Free | Studio Lab |
|-------------|------------|------------|
| **10GB BAM files** | ❌ No storage | ✅ 15GB persistent |
| **5-6 hour training** | ❌ 90 min limit | ✅ 12 hour sessions |
| **Checkpointing** | ❌ Lost on disconnect | ✅ Persists forever |
| **Environment setup** | ❌ Reinstall samtools/bcftools | ✅ Conda persists |
| **Resume analysis** | ❌ Start from scratch | ✅ Pick up where you left off |
| **Team sharing** | ❌ Copy/paste notebooks | ✅ Git integration |

**Bottom line:** This research workflow is not viable on Colab Free.

## Time Estimate

**First Run:**
- Setup: 15 minutes (one-time)
- Data download: 45-60 minutes (one-time, ~10GB)
- Environment setup: 15 minutes (one-time)
- Feature extraction: 60-90 minutes
- Model training: 5-6 hours
- Analysis: 1-2 hours
- **Total: 8-10 hours**

**Subsequent Runs:**
- Data: Instant (cached)
- Environment: Instant (persisted)
- Features: Instant (cached) or regenerate if needed
- Training: 5-6 hours (or resume from checkpoint)
- **Total: 5-6 hours**

You can pause and resume at any time!

## Biological Insights

This project enables research questions like:

1. **Model Generalization**: Do variant callers trained on European samples work well on African samples?
2. **Population-Specific Variants**: Can we identify variants enriched in specific populations?
3. **Ensemble Benefits**: Does combining multiple models improve accuracy?
4. **Challenging Regions**: Which genomic regions have low inter-model agreement?
5. **Quality vs. Quantity**: How many samples needed for robust consensus calls?

## Technical Details

### Ensemble Architecture
- **Model 1-3:** CNN variant callers (different architectures)
- **Model 4-5:** Gradient boosting (XGBoost, LightGBM)
- **Model 6:** Random forest classifier
- **Consensus:** Weighted voting by validation performance

### Training Strategy
- **Per-sample models:** Train on individual's data
- **Transfer learning:** Fine-tune from base model
- **Multi-task:** Joint SNP/indel detection
- **Class balancing:** Handle variant rarity (~0.1%)

### Evaluation
- **GIAB truth set:** High-confidence variants for NA12878
- **Stratification:** By variant type (SNP, indel), frequency, region
- **Comparison:** GATK HaplotypeCaller, DeepVariant benchmarks

## Next Steps

After mastering Studio Lab:

- **Tier 2:** Introduction to AWS services (S3, Batch, Athena) - $5-15
  - Store 100GB+ of BAM files on S3
  - Distributed variant calling with AWS Batch
  - Query variants with Athena

- **Tier 3:** Production infrastructure with CloudFormation - $50-500/month
  - Process 1000+ samples
  - Automated QC and annotation
  - Clinical database integration

## Resources

- [SageMaker Studio Lab Documentation](https://studiolab.sagemaker.aws/docs)
- [1000 Genomes Project](https://www.internationalgenome.org/)
- [Genome in a Bottle (GIAB)](https://www.nist.gov/programs-projects/genome-bottle)
- [DeepVariant Paper](https://www.nature.com/articles/nbt.4235)
- [pysam Documentation](https://pysam.readthedocs.io/)

## Troubleshooting

### Environment Issues
```bash
# Reset environment
conda env remove -n genomics-studio-lab
conda env create -f environment.yml
```

### Storage Full
```bash
# Check usage
du -sh data/ saved_models/

# Clean old files
rm -rf data/processed/*.old
rm -rf saved_models/*.backup
```

### BAM File Corruption
```bash
# Re-index BAM files
samtools index data/raw/*.bam
```

### Session Timeout
Data persists! Just restart and continue where you left off.
Model checkpoints ensure you don't lose training progress.

---

**🧬 Built for SageMaker Studio Lab with [Claude Code](https://claude.com/claude-code)**
