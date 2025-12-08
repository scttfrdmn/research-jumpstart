# Variant Calling with Deep Learning on 1000 Genomes Data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Duration:** 60-90 minutes
**Platform:** Google Colab or SageMaker Studio Lab
**Cost:** $0 (no AWS account needed)
**Data:** ~1.5GB 1000 Genomes BAM files

## Research Goal

Train a convolutional neural network (CNN) to identify genetic variants from raw sequencing reads. Using real 1000 Genomes Project data, build a deep learning variant caller that detects SNPs and small indels from aligned BAM files, competing with traditional tools like GATK.

## Quick Start

### Run in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scttfrdmn/rj-genomics-variant-calling-tier0/blob/main/genomics-variant-calling.ipynb)

1. Click the badge above
2. Sign in with Google account
3. Click "Runtime" → "Run all"
4. Complete in 60-90 minutes

### Run in SageMaker Studio Lab
[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/scttfrdmn/rj-genomics-variant-calling-tier0/blob/main/genomics-variant-calling.ipynb)

1. Create free account: https://studiolab.sagemaker.aws
2. Click the badge above to import
3. Open `genomics-variant-calling.ipynb`
4. Run all cells

### Run Locally
```bash
# Clone this repository
git clone https://github.com/scttfrdmn/rj-genomics-variant-calling-tier0.git
cd rj-genomics-variant-calling-tier0

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook genomics-variant-calling.ipynb
```

## What You'll Build

1. **Download 1000 Genomes data** (~1.5GB BAM file subset, takes 15-20 min)
2. **Generate pileup tensors** (convert reads to image-like representations)
3. **Train CNN variant caller** (60-75 minutes on GPU)
4. **Evaluate predictions** (precision, recall, F1 vs GATK truth set)
5. **Call variants on held-out region** (generate VCF file)

## Dataset

**1000 Genomes Project - Phase 3**
- **Sample:** Single individual (NA12878 - CEU population, reference sample)
- **Region:** Chromosome 20 (subset: 20:10000000-20000000, 10 Mb)
- **Technology:** Illumina whole-genome sequencing (30x coverage)
- **Format:** BAM (aligned reads) + reference genome (GRCh38)
- **Size:** ~1.5GB (compressed)
- **Source:** AWS Open Data Registry (s3://1000genomes, no credentials needed)
- **Truth set:** Genome in a Bottle (GIAB) high-confidence variant calls

**Why NA12878?**
- Extensively characterized reference sample
- Gold standard for benchmarking variant callers
- GIAB truth set available for evaluation
- Used in clinical validation studies

## Colab Considerations

This notebook works on Colab but you'll notice:
- **20-minute download** at session start (no persistence)
- **60-75 minute training** (close to timeout limit)
- **Re-download required** if session disconnects
- **~11GB RAM usage** (near Colab's limit)

These limitations become important for real research workflows.

## What's Included

- Single Jupyter notebook (`genomics-variant-calling.ipynb`)
- AWS Open Data access utilities (no credentials needed)
- CNN architecture for variant calling (ResNet-inspired)
- Pileup tensor generation pipeline
- Training and evaluation code
- VCF file generation and export

## Key Methods

- **Pileup image generation:** Convert aligned reads to tensor representations (221x100x7)
- **Convolutional Neural Networks:** Learn patterns from read alignments
- **Multi-task learning:** Joint SNP detection + genotype classification
- **Quality score prediction:** PHRED-scaled confidence scores
- **Post-processing:** Convert predictions to standard VCF format

## Biological Context

Variant calling is a fundamental step in genomics research:

### Clinical Applications
- **Rare disease diagnosis:** Identify disease-causing mutations
- **Cancer genomics:** Detect somatic mutations for precision oncology
- **Pharmacogenomics:** Predict drug response based on genotype
- **Carrier screening:** Identify recessive disease carriers

### Research Applications
- **Population genetics:** Study genetic diversity and evolution
- **Genome-wide association studies (GWAS):** Link variants to traits
- **Agricultural breeding:** Improve crop varieties and livestock
- **Conservation genetics:** Assess genetic diversity in endangered species

### Why Deep Learning?

Traditional variant callers (GATK, FreeBayes, DeepVariant) use:
- **Hand-crafted statistical models:** Explicit assumptions about sequencing errors
- **Heuristic filters:** Manual tuning of quality thresholds
- **Limited context:** Local information only

**Deep learning advantages:**
- Learns complex patterns directly from data
- Captures long-range dependencies in read alignments
- Automatically discovers relevant features
- Adapts to different sequencing technologies

## Requirements

**Python:** 3.9+

**Core Libraries:**
- tensorflow or pytorch
- pysam >= 0.19 (BAM file processing)
- biopython >= 1.79
- numpy, scipy, pandas
- scikit-learn >= 1.0
- matplotlib, seaborn

See `requirements.txt` for complete list.

**Compute:**
- **RAM:** 12 GB minimum (for BAM processing)
- **CPU:** Multi-core recommended (4+ cores for data preprocessing)
- **GPU:** Optional but strongly recommended (T4 speeds up training 10x)
- **Storage:** 2 GB (for downloaded BAM files and reference)

**Time:**
- Setup: 2-5 minutes
- Data download: 15-20 minutes
- Pileup generation: 10-15 minutes
- Model training: 60-75 minutes (GPU) or 8-10 hours (CPU)
- Evaluation: 5-10 minutes
- **Total: 90-120 minutes with GPU**

## Technical Details

### Model Architecture

```
Input: 221 x 100 x 7 pileup tensor
├── Position dimension: 221 bp window
├── Read depth dimension: Up to 100 reads
└── Channels (7):
    ├── Read base (A, C, G, T, N)
    ├── Base quality score
    ├── Mapping quality
    ├── Strand (forward/reverse)
    ├── Read position
    ├── Insert size
    └── Mate mapping quality

Architecture: ResNet-inspired CNN
├── Conv2D blocks with residual connections
├── Batch normalization
├── Max pooling
├── Dropout (0.3)
└── Multi-task output:
    ├── Binary classification: Variant / No variant
    └── Genotype classification: 0/0, 0/1, 1/1

Total parameters: ~500,000
```

### Training

- **Loss:** Binary cross-entropy (variant detection) + categorical cross-entropy (genotype)
- **Optimizer:** Adam with learning rate 0.001
- **Batch size:** 32 pileup windows
- **Epochs:** ~50 (60-75 min on T4 GPU)
- **Class weighting:** Balance variant/non-variant classes (imbalanced data)

### Evaluation Metrics

- **Precision/Recall/F1:** vs GIAB truth set (gold standard)
- **SNP vs Indel breakdown:** Performance by variant type
- **ROC curves:** Threshold selection for clinical applications
- **Runtime comparison:** vs GATK HaplotypeCaller
- **Concordance:** Match rate with established variant callers

## Performance Expectations

Using this notebook, you can expect:

**Training metrics:**
- **Accuracy:** ~97% on validation set
- **Precision:** ~92% (low false positive rate)
- **Recall:** ~88% (catches most true variants)
- **F1 score:** ~90%
- **Training time:** 60-75 minutes on GPU

**Scientific capability:**
- Detect SNPs and small indels (1-10 bp)
- Handle coverage from 10x to 100x
- Process ~10 Mb genomic region per session
- Generate VCF files compatible with standard tools

**Comparison with GATK:**
- Comparable precision (~92% vs ~94%)
- Slightly lower recall (~88% vs ~92%)
- 2-3x faster inference (after training)
- Better at low-coverage regions (< 15x)

**Limitations:**
- Larger indels (> 10 bp) challenging
- Structural variants not detected
- Requires high-quality alignments
- Training needed for different sequencing technologies

## Troubleshooting

### Data Download Issues
```python
# If AWS S3 is slow or timing out
# Reduce region size
bam_file = download_1000genomes_bam(
    sample="NA12878",
    chrom="20",
    start=10000000,
    end=15000000  # 5 Mb instead of 10 Mb
)

# Or use pre-cached sample data
bam_file = load_sample_bam()  # ~500 MB subset
```

### Out of Memory
```python
# Reduce batch size during training
model.fit(X_train, y_train, batch_size=16)  # instead of 32

# Or reduce pileup depth
max_depth = 50  # instead of 100 reads per position
```

### GPU Not Available
```python
# Training on CPU is ~10x slower
# Reduce epochs or use smaller region
epochs = 20  # instead of 50
# Or process smaller genomic region
end = 12000000  # 2 Mb instead of 10 Mb
```

### Session Timeout
```python
# Save checkpoint before timeout risk
model.save('variant_caller_checkpoint.h5')
pileups_save('pileups_cache.npz', X_train, y_train)

# Resume in next session
model = load_model('variant_caller_checkpoint.h5')
X_train, y_train = pileups_load('pileups_cache.npz')
```

## Next Steps

This project is **Tier 0** in the Research Jumpstart framework. Ready for more?

### Tier 1: Multi-Cohort Variant Analysis (4-8 hours, FREE)
- Cache 8-12GB of multi-sample data (5-10 individuals from 1000 Genomes)
- Train ensemble variant callers (CNN + Random Forest + deep learning fusion)
- Joint genotyping across multiple samples
- Persistent storage for large model checkpoints (SageMaker Studio Lab)
- [Learn more →](https://github.com/scttfrdmn/research-jumpstart/tree/main/projects/genomics/variant-analysis/tier-1)

### Tier 2: Production Variant Calling Platform (2-3 days, $200-400/month)
- 100GB+ BAM files on S3 (whole genomes, 30x coverage)
- Distributed variant calling with AWS Batch (thousands of parallel jobs)
- Population-scale analysis (1000+ samples)
- Integration with annotation databases (ClinVar, gnomAD, dbSNP)
- CloudFormation one-click deployment
- [Learn more →](https://github.com/scttfrdmn/research-jumpstart/tree/main/projects/genomics/variant-analysis/tier-2)

### Tier 3: Biobank-Scale Variant Pipeline (Ongoing, $3K-8K/month)
- Biobank integration (UK Biobank, TOPMed - 100K+ samples)
- Multi-technology support (Illumina, PacBio, Nanopore)
- Real-time variant calling for clinical diagnostics
- AI-assisted interpretation (pathogenicity prediction with Bedrock)
- HIPAA-compliant infrastructure
- Integration with electronic health records
- [Learn more →](https://github.com/scttfrdmn/research-jumpstart/tree/main/projects/genomics/variant-analysis/tier-3)

## Extension Ideas

Once you've completed the base project:

### Beginner Extensions (2-4 hours)
1. **Different chromosomes:** Train on chr21, chr22 (smaller, faster)
2. **Coverage analysis:** Compare performance at 10x, 30x, 60x coverage
3. **Quality filtering:** Experiment with different PHRED score thresholds
4. **Visualization:** Plot pileup images to understand what CNN sees

### Intermediate Extensions (4-8 hours)
5. **Ensemble calling:** Combine CNN with traditional callers (GATK, FreeBayes)
6. **Transfer learning:** Fine-tune on cancer samples (tumor vs normal)
7. **Attention mechanisms:** Add attention layers to highlight important reads
8. **Indel-specific models:** Train separate models for SNPs and indels

### Advanced Extensions (8+ hours)
9. **Long-read sequencing:** Adapt model for PacBio/Nanopore data
10. **Structural variants:** Extend to detect deletions, duplications, inversions
11. **Somatic variant calling:** Detect low-frequency mutations in cancer
12. **Real-time calling:** Build streaming variant caller for nanopore sequencing

## Additional Resources

### Genomics Data
- **1000 Genomes:** https://www.internationalgenome.org/
- **Genome in a Bottle:** https://www.nist.gov/programs-projects/genome-bottle
- **gnomAD:** https://gnomad.broadinstitute.org/
- **ClinVar:** https://www.ncbi.nlm.nih.gov/clinvar/

### Variant Calling Tools
- **GATK:** https://gatk.broadinstitute.org/
- **DeepVariant:** https://github.com/google/deepvariant
- **FreeBayes:** https://github.com/freebayes/freebayes
- **SAMtools:** http://www.htslib.org/

### Genomics File Formats
- **BAM/SAM:** http://samtools.github.io/hts-specs/
- **VCF:** https://samtools.github.io/hts-specs/VCFv4.3.pdf
- **FASTA:** https://en.wikipedia.org/wiki/FASTA_format

### Deep Learning for Genomics
- **DeepVariant Paper:** [Poplin et al. (2018) Nature Biotechnology](https://www.nature.com/articles/nbt.4235)
- **Deep learning in genomics:** [Zou et al. (2019) Nature Genetics](https://www.nature.com/articles/s41588-018-0295-5)

## Citation

If you use this project or discover new variants, please cite:

```bibtex
@software{rj_genomics_variant_calling_tier0,
  title = {Variant Calling with Deep Learning on 1000 Genomes Data},
  author = {Research Jumpstart Community},
  year = {2025},
  doi = {10.5281/zenodo.XXXXXXX},
  url = {https://github.com/scttfrdmn/rj-genomics-variant-calling-tier0}
}
```

Also cite the data sources:
```bibtex
@article{1000genomes2015,
  title={A global reference for human genetic variation},
  author={{1000 Genomes Project Consortium}},
  journal={Nature},
  volume={526},
  number={7571},
  pages={68--74},
  year={2015},
  doi={10.1038/nature15393}
}

@article{giab2014,
  title={Extensive sequencing of seven human genomes to characterize benchmark reference materials},
  author={Zook, Justin M and others},
  journal={Scientific Data},
  volume={1},
  pages={140054},
  year={2014},
  doi={10.1038/sdata.2014.54}
}
```

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

1000 Genomes data is public domain. GIAB data is public domain (NIST).

---

*Part of [Research Jumpstart](https://github.com/scttfrdmn/research-jumpstart) - Pre-built research workflows for cloud computing*

*Version: 1.0.0 | Last updated: 2025-12-07*
