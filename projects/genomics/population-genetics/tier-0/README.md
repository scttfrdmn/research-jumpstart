# Population Genetics Analysis with 1000 Genomes

**Duration:** 60-90 minutes
**Platform:** Google Colab or SageMaker Studio Lab
**Cost:** $0 (no AWS account needed)
**Data:** 1000 Genomes Project (chromosome 22, ~2GB)

## Research Goal

Analyze population genetic structure, detect signatures of natural selection, and identify genetic variants using publicly available human genomic data from the 1000 Genomes Project. Learn fundamental population genetics methods including PCA, FST, and selection scans.

## Quick Start

### Run in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/genomics/population-genetics/tier-0/population-genetics.ipynb)

### Run in SageMaker Studio Lab
[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/genomics/population-genetics/tier-0/population-genetics.ipynb)

## What You'll Build

1. **Download 1000 Genomes data** (chromosome 22 VCF, ~2GB, 20-25 minutes)
2. **Quality control and filtering** (MAF, HWE, missingness filters)
3. **Population structure analysis** (PCA visualization, 15-20 minutes)
4. **FST calculation** (genetic differentiation between populations)
5. **Selection scan** (Tajima's D for recent selection)
6. **Visualization** (PCA plots, Manhattan plots, allele frequency spectra)

## Dataset

**1000 Genomes Project - Chromosome 22**
- Individuals: 2,504 from 26 populations
- Populations: African, European, East Asian, South Asian, American
- Variants: ~1.1 million SNPs on chr22
- Coverage: 30× whole genome sequencing
- Format: VCF (Variant Call Format)
- Size: ~2GB compressed
- Source: https://www.internationalgenome.org/
- Reference: GRCh38/hg38

**Population Groups:**
- AFR: Yoruba, Esan, Gambian, Luhya, Mende
- EUR: Utah, British, Finnish, Iberian, Toscani
- EAS: Han Chinese, Japanese, Kinh Vietnamese
- SAS: Bengali, Gujarati, Telugu, Punjabi, Tamil
- AMR: Colombian, Mexican, Peruvian, Puerto Rican

## Colab Considerations

This notebook works on Colab but you'll notice:
- **20-25 minute download** (no persistence, re-download on disconnect)
- **Single chromosome** (chr22 only vs. genome-wide analysis)
- **Limited sample size** (2,504 individuals vs. biobank-scale)
- **Basic statistics** (no complex haplotype analysis)
- **Memory constraints** (~11GB for chr22 only)

These limitations prevent comprehensive population genetics research.

## What's Included

- Single Jupyter notebook (`population-genetics.ipynb`)
- 1000 Genomes data access utilities
- VCF parsing with cyvcf2/scikit-allel
- PCA implementation
- FST calculation
- Tajima's D selection scan
- Population structure visualization

## Key Methods

- **Principal Component Analysis (PCA):** Population structure inference
- **FST (Fixation Index):** Genetic differentiation between populations
- **Tajima's D:** Test for selective neutrality
- **Minor Allele Frequency (MAF):** Variant filtering
- **Hardy-Weinberg Equilibrium:** Quality control
- **Linkage Disequilibrium:** LD pruning for PCA

## Analysis Outputs

1. **PCA Plot:** Population clustering in genetic space
2. **FST Matrix:** Pairwise genetic distances between populations
3. **Selection Signals:** Genomic regions under recent selection
4. **Allele Frequency Spectra:** Distribution across populations
5. **Population Trees:** Phylogenetic relationships

## Next Steps

**Need genome-wide analysis?** This project analyzes one chromosome:

- **Tier 1:** [Whole-genome population genetics](../tier-1/) on Studio Lab
  - All 22 autosomes + X chromosome (50GB+ data)
  - ADMIXTURE for ancestry estimation
  - iHS and XP-EHH selection scans
  - Persistent storage for large VCF files

- **Tier 2:** [AWS-integrated analysis](../tier-2/) with cloud storage
  - 1000 Genomes full dataset on S3 (84.7M variants)
  - Distributed computing for genome-wide scans
  - Hail for large-scale genomic analysis
  - SageMaker for GWAS integration

- **Tier 3:** [Biobank-scale genetics](../tier-3/) with CloudFormation
  - UK Biobank (500K individuals)
  - Distributed haplotype-based analysis
  - AWS Batch for parallelized selection scans
  - Integration with phenotype databases
  - GWAS and polygenic score calculation

## Requirements

Pre-installed in Colab and Studio Lab:
- Python 3.9+
- scikit-allel (genomic analysis)
- cyvcf2 (VCF parsing)
- pandas, numpy
- scikit-learn (PCA)
- matplotlib, seaborn

**Data Download:** Requires 2GB storage and 20-25 minute download time
