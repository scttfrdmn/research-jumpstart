# Population Genetics at Scale

Large-scale population genetics analysis using 1000 Genomes and biobank data. Analyze population structure, detect signatures of natural selection, calculate genetic differentiation, and perform GWAS on cloud infrastructure.

## Quick Start by Tier

**New here?** Start with tier-0 (60-90 min, free) to learn population genetics fundamentals with 1000 Genomes data.

### 🟢 Tier 0: Population Structure & Selection (60-90 min, FREE)
**[Launch tier-0 project →](tier-0/)**

Analyze population genetics with 1000 Genomes chromosome 22:
- ✅ Real 1000 Genomes data (~2GB, chr22, 2,504 individuals from 26 populations)
- ✅ Principal Component Analysis (PCA) for population structure
- ✅ FST calculation (genetic differentiation between populations)
- ✅ Tajima's D selection scan for recent natural selection
- ✅ Quality control (MAF, HWE, missingness filters)
- ✅ Complete in 60-90 minutes
- ✅ No AWS account needed (Colab or Studio Lab)

**Platform**: Google Colab or SageMaker Studio Lab
**Cost**: $0

[View tier-0 README →](tier-0/README.md) | [Open in Colab →](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/genomics/population-genetics/tier-0/population-genetics.ipynb)

---

### 🟡 Tier 1: Whole-Genome Population Analysis (4-8 hours, FREE)
**[Launch tier-1 project →](tier-1/)**

Genome-wide population genetics with all chromosomes:
- ✅ 50GB+ whole-genome data (all 22 autosomes + X, 84.7M variants)
- ✅ ADMIXTURE for ancestry estimation (model-based clustering)
- ✅ Advanced selection scans (iHS, XP-EHH for extended haplotypes)
- ✅ Genome-wide FST and Tajima's D windowed analysis
- ✅ LD-based haplotype block detection
- ✅ Persistent storage (Studio Lab)
- ✅ Still free, no AWS account

**Platform**: SageMaker Studio Lab
**Cost**: $0

[View tier-1 README →](tier-1/README.md)

---

### 🟠 Tier 2: Production Genomic Analysis (2-3 days, $20-50 per analysis)
**[Launch tier-2 project →](tier-2/)**

Research-grade population genetics infrastructure:
- ✅ CloudFormation one-click deployment
- ✅ 1000 Genomes full dataset on S3 (84.7M variants, no download needed)
- ✅ Distributed computing with Hail for large-scale genomic analysis
- ✅ SageMaker for GWAS and machine learning integration
- ✅ Genome-wide selection scans with AWS Batch parallelization
- ✅ VCF annotation with VEP (Variant Effect Predictor)
- ✅ Publication-ready outputs and visualizations

**Platform**: AWS with CloudFormation
**Cost**: $20-50 per complete genome-wide analysis

[View tier-2 README →](tier-2/README.md)

---

### 🔴 Tier 3: Biobank-Scale Genetics Platform (Ongoing, $2K-5K/month)
**[Launch tier-3 project →](tier-3/)**

Production platform for population genetics research:
- ✅ UK Biobank integration (500K individuals, whole-genome sequencing)
- ✅ Distributed haplotype-based analysis at massive scale
- ✅ Multi-population GWAS with meta-analysis
- ✅ Polygenic score calculation and validation
- ✅ Integration with phenotype databases (clinical, behavioral)
- ✅ AI-assisted interpretation (Amazon Bedrock for variant annotation)
- ✅ Team collaboration with versioned analyses

**Platform**: AWS multi-account with enterprise support
**Cost**: $2K-5K/month

[View tier-3 README →](tier-3/README.md)

---

## What You'll Learn

Across all tiers, this project teaches:
- Population structure inference with PCA and ADMIXTURE
- Genetic differentiation metrics (FST, pairwise distances)
- Selection scan methods (Tajima's D, iHS, XP-EHH for natural selection)
- VCF file manipulation and quality control (MAF, HWE, LD pruning)
- Genome-wide association studies (GWAS) at scale
- Distributed genomic analysis with Hail on cloud infrastructure

## Technologies & Tools

- **Data sources**: 1000 Genomes Project (84.7M variants), gnomAD, UK Biobank, TOPMed
- **Languages**: Python 3.9+
- **Core libraries**: pandas, numpy, scipy, scikit-learn
- **Genomics tools**: scikit-allel, cyvcf2, pysam, PLINK, Hail (distributed genomics)
- **Selection scans**: selscan (iHS, XP-EHH), rehh
- **Visualization**: matplotlib, seaborn, plotly, IGV
- **Cloud services** (tier 2+): S3 (1000 Genomes public dataset), Batch (distributed scans), SageMaker (GWAS ML), Glue, Athena, Bedrock

## Project Structure

```
population-genetics/
├── tier-0/              # Chr22 analysis (60-90 min, FREE)
│   ├── population-genetics.ipynb
│   ├── README.md
│   └── requirements.txt
├── tier-1/              # Whole-genome (4-8 hours, FREE)
│   ├── notebooks/
│   ├── src/
│   ├── data/
│   ├── environment.yml
│   └── README.md
├── tier-2/              # Production (2-3 days, $20-50/analysis)
│   ├── cloudformation/
│   ├── notebooks/
│   ├── src/
│   ├── tests/
│   └── README.md
└── tier-3/              # Biobank-scale (ongoing, $2K-5K/mo)
    ├── cloudformation/
    ├── notebooks/
    ├── src/
    ├── infrastructure/
    └── README.md
```

## Progression Path

```
Tier 0           → Tier 1          → Tier 2            → Tier 3
Chr22 Analysis     Whole-Genome       Production          Biobank-Scale
2,504 individuals  84.7M variants     Hail + AWS Batch    500K individuals
60-90 min          4-8 hours          2-3 days            Ongoing
FREE               FREE               $20-50/analysis     $2K-5K/mo
```

You can:
- ✅ Skip tiers if you have AWS experience and large-scale genomics needs
- ✅ Stop at any tier - tier-1 is great for papers, tier-2 for grants
- ✅ Mix and match - use tier-0 for methods, tier-2 for publications

[Understanding tiers →](../../../docs/projects/tiers.md)

## Population Genetics Applications

- **Population structure**: PCA, ADMIXTURE for ancestry and migration patterns
- **Natural selection**: Detect genomic regions under selection (Tajima's D, iHS, XP-EHH)
- **Genetic differentiation**: FST between populations, isolation-by-distance
- **GWAS integration**: Associate variants with phenotypes across populations
- **Demographic inference**: Estimate population size history and admixture events
- **Polygenic scores**: Calculate genetic risk across ancestries

## Related Projects

- **[Variant Analysis](../variant-analysis/)** - Variant calling and annotation pipelines
- **[Medical - Disease Prediction](../../medical/disease-prediction/)** - Genetic risk prediction
- **[Archaeology - Site Analysis](../../archaeology/site-analysis/)** - Ancient DNA analysis methods

## Common Use Cases

- **Academic research**: Publish population genetics papers, study human evolution
- **Evolutionary biology**: Detect adaptive alleles, study selection pressures
- **Medical genetics**: Understand disease risk across populations, pharmacogenomics
- **Forensic genetics**: Ancestry inference, population assignment
- **Conservation genetics**: Study endangered species population structure
- **Agricultural genomics**: Analyze crop and livestock genetic diversity

## Cost Estimates

**Tier 2 Production (Genome-Wide Analysis)**:
- **S3 data access** (1000 Genomes public dataset, no egress): $0
- **Compute** (c5.4xlarge, 8 hours for full analysis): $5.44
- **AWS Batch** (parallelized selection scans, 20 jobs): $10-15
- **SageMaker** (optional GWAS ML): ml.m5.2xlarge, 2 hours = $1.54
- **Storage** (results, 10GB): $0.23/month
- **Total**: $20-50 per complete genome-wide analysis

**Optimization tips**:
- Use spot instances for Batch jobs (60-70% savings)
- Cache LD-pruned VCFs to skip preprocessing
- Use Hail's optimized Parquet format for repeated analyses
- Process chromosomes in parallel for faster completion

## Support

- **Questions**: [GitHub Discussions](https://github.com/scttfrdmn/research-jumpstart/discussions)
- **Issues**: [GitHub Issues](https://github.com/scttfrdmn/research-jumpstart/issues)
- **Office Hours**: [Every Tuesday](../../../docs/community/office-hours.md)

## Citation

If you use this project in your research, please cite:

```bibtex
@software{research_jumpstart_population_genetics,
  title = {Population Genetics at Scale: Research Jumpstart},
  author = {Research Jumpstart Community},
  year = {2025},
  url = {https://github.com/scttfrdmn/research-jumpstart},
  note = {Accessed: [date]}
}
```

Also cite the 1000 Genomes data:

```bibtex
@article{1000genomes2015,
  title={A global reference for human genetic variation},
  author={1000 Genomes Project Consortium},
  journal={Nature},
  volume={526},
  pages={68--74},
  year={2015},
  doi={10.1038/nature15393}
}
```

**Data sources**:
- **1000 Genomes**: https://www.internationalgenome.org/
- **gnomAD**: https://gnomad.broadinstitute.org/
- **UK Biobank**: https://www.ukbiobank.ac.uk/

## License

Apache 2.0 - See [LICENSE](../../../LICENSE) for details.

---

*Part of [Research Jumpstart](https://github.com/scttfrdmn/research-jumpstart) - Pre-built research workflows for cloud computing*
