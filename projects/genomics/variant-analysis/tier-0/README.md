# Variant Calling with Deep Learning on 1000 Genomes Data

**Duration:** 60-90 minutes
**Platform:** Google Colab or SageMaker Studio Lab
**Cost:** $0 (no AWS account needed)
**Data:** ~1.5GB 1000 Genomes BAM files

## Research Goal

Train a convolutional neural network (CNN) to identify genetic variants from raw sequencing reads. Using real 1000 Genomes Project data, build a deep learning variant caller that detects SNPs and small indels from aligned BAM files, competing with traditional tools like GATK.

## Quick Start

### Run in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/genomics/variant-analysis/tier-0/genomics-variant-calling.ipynb)

### Run in SageMaker Studio Lab
[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/genomics/variant-analysis/tier-0/genomics-variant-calling.ipynb)

## What You'll Build

1. **Download 1000 Genomes data** (~1.5GB BAM file subset, takes 15-20 min)
2. **Generate pileup tensors** (convert reads to image-like representations)
3. **Train CNN variant caller** (60-75 minutes on GPU)
4. **Evaluate predictions** (precision, recall, F1 vs GATK truth set)
5. **Call variants on held-out region** (generate VCF file)

## Dataset

**1000 Genomes Project - Phase 3**
- Sample: Single individual (NA12878 - CEU population)
- Region: Chromosome 20 (subset: 20:10000000-20000000)
- Data: Illumina whole-genome sequencing (30x coverage)
- Format: BAM (aligned reads) + reference genome
- Size: ~1.5GB (compressed)
- Source: AWS Open Data Registry (s3://1000genomes)
- Truth set: GIAB high-confidence variant calls

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
- CNN architecture for variant calling
- Training and evaluation pipeline
- VCF file generation

## Key Methods

- **Pileup image generation:** Convert aligned reads to tensor representations
- **Convolutional Neural Networks:** Learn patterns from read alignments
- **Multi-task learning:** Joint SNP/indel detection
- **Quality score prediction:** PHRED-scaled confidence scores
- **Post-processing:** Convert predictions to standard VCF format

## Biological Context

Variant calling is a fundamental step in genomics research:
- **Clinical diagnostics:** Identify disease-causing mutations
- **Population genetics:** Study genetic diversity
- **Cancer genomics:** Detect somatic mutations
- **Agricultural breeding:** Improve crop varieties

Traditional variant callers (GATK, FreeBayes) use hand-crafted statistical models. Deep learning approaches can learn complex patterns directly from data, potentially improving accuracy for challenging variant types.

## Next Steps

**Experiencing limitations?** This project pushes Colab to its limits:

- **Tier 1:** [Multi-cohort variant analysis](../tier-1/) on Studio Lab
  - Cache 8-12GB of multi-sample data (download once, use forever)
  - Train ensemble variant callers (5-6 hours continuous)
  - Persistent environments and checkpoints
  - No session timeouts

- **Tier 2:** [AWS-integrated workflows](../tier-2/) with S3 and Batch
  - Store 100GB+ of BAM files on S3
  - Distributed variant calling with AWS Batch
  - Population-scale analysis

- **Tier 3:** [Production pipeline](../tier-3/) with full CloudFormation
  - Process 1000+ samples (multi-TB)
  - Automated QC and annotation
  - Integration with clinical databases

## Requirements

Pre-installed in Colab and Studio Lab:
- Python 3.9+, TensorFlow/PyTorch
- pysam, biopython
- scikit-learn, scipy
- matplotlib, seaborn

**Note:** First run downloads 1.5GB of data (15-20 minutes)

## Technical Details

### Model Architecture
- **Input:** 221x100x7 pileup tensor (position x read depth x channels)
- **Channels:** Read base, base quality, mapping quality, strand, etc.
- **Architecture:** ResNet-inspired CNN with residual connections
- **Output:** Per-position variant probabilities + genotype

### Training
- **Loss:** Binary cross-entropy + genotype classification
- **Optimizer:** Adam with learning rate scheduling
- **Batch size:** 32 pileup windows
- **Epochs:** ~50 (60-75 min on T4 GPU)

### Evaluation Metrics
- **Precision/Recall/F1:** vs GIAB truth set
- **SNP/Indel breakdown:** Performance by variant type
- **ROC curves:** Threshold selection
- **Runtime comparison:** vs GATK HaplotypeCaller

## Common Issues

**Out of memory:** Reduce batch size or use smaller genomic region
**Download timeout:** Restart and notebook will resume
**GPU not available:** Falls back to CPU (much slower, ~4-6 hours)

## References

- Poplin et al. (2018) "A universal SNP and small-indel variant caller using deep neural networks" *Nature Biotechnology*
- 1000 Genomes Project Consortium (2015) *Nature*
- Genome in a Bottle Consortium (GIAB) truth sets

---

**Built for Google Colab and SageMaker Studio Lab**
