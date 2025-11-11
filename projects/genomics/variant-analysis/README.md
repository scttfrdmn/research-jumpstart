# Genomic Variant Analysis

**Difficulty**: 🟡 Intermediate | **Time**: ⏱️ 2-3 hours (Studio Lab)

Analyze genetic variants from VCF files to identify mutations, calculate allele frequencies, and visualize genomic features.

## Status

**Studio Lab**: 🚧 Lightweight quickstart (in development)
**Unified Studio**: ⏳ Planned

## Quick Start (Studio Lab)

```bash
git clone https://github.com/research-jumpstart/research-jumpstart.git
cd research-jumpstart/projects/genomics/variant-analysis/studio-lab
conda env create -f environment.yml
conda activate genomics-analysis
jupyter notebook quickstart.ipynb
```

## What You'll Learn

- Load and parse VCF (Variant Call Format) files
- Calculate variant statistics (SNPs, indels, quality scores)
- Compute allele frequencies
- Filter variants by quality
- Visualize variant distributions
- Annotate functional effects

## Key Analyses

1. **Variant Quality Control**
   - Quality score distributions
   - Depth of coverage analysis
   - Missing data assessment

2. **Allele Frequency Calculation**
   - Population-level frequencies
   - Hardy-Weinberg equilibrium
   - Minor allele frequency (MAF)

3. **Variant Annotation**
   - Gene mapping
   - Functional consequence prediction
   - dbSNP cross-reference

4. **Visualization**
   - Manhattan plots
   - Variant density along chromosomes
   - Quality score distributions

## Sample Data

### AWS Open Data Registry (For Real Analysis)

Access large-scale genomics datasets from AWS for free:

**1000 Genomes Project** (s3://1000genomes)
- 2,504 individuals from 26 populations
- Complete VCF, BAM, CRAM files
- ~200 TB of data
- Public access, no credentials required

**gnomAD** (s3://gnomad-public-us-east-1)
- 125,748 exomes + 71,702 genomes
- Population allele frequencies
- ~20 TB of variant data

**TCGA** (s3://tcga-2-open)
- Cancer genomics data
- 33 cancer types, 11,000+ patients
- WGS, WXS, RNA-Seq

```python
# Access AWS Open Data (see studio-lab/aws_data_access.py)
from aws_data_access import list_1000genomes, download_sample_vcf

# List chromosome 22 variants
files = list_1000genomes(chromosome='chr22', phase='phase3')

# Download for analysis
download_sample_vcf(chromosome='chr22', output_dir='data/')
```

See `studio-lab/aws_data_access.py` for complete examples.

### Synthetic Sample

Included synthetic VCF file with:
- 100 variants across chr22
- SNPs and small indels
- Quality scores and genotypes
- Educational/demonstration purposes

## Cost

**Studio Lab**: Free forever
**Unified Studio**: ~$15-25 per analysis (real 1000 Genomes data)

## Resources

- [VCF Format Specification](https://samtools.github.io/hts-specs/VCFv4.2.pdf)
- [1000 Genomes Project](https://www.internationalgenome.org/)
- [PyVCF Documentation](https://pyvcf.readthedocs.io/)

*Last updated: 2025-11-09*
