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

Synthetic VCF file with:
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
