# Variant Analysis at Scale

Large-scale genomic variant calling and analysis using deep learning for SNP/indel detection, variant annotation, and population-scale analysis on cloud infrastructure.

## Quick Start by Tier

**New here?** Start with tier-0 (60-90 min, free) to learn variant calling with deep learning.

### 🟢 Tier 0: Variant Calling with Deep Learning (60-90 min, FREE)
**[Launch tier-0 project →](tier-0/)**

Train a CNN to identify genetic variants from sequencing reads:
- ✅ Real 1000 Genomes data (~1.5GB BAM files, chromosome 20 subset, 30x coverage)
- ✅ CNN variant caller (pileup tensor processing, ResNet-inspired architecture)
- ✅ Multi-task learning (SNP + indel detection, genotype classification)
- ✅ Evaluation vs GATK/GIAB truth sets (precision, recall, F1 scores)
- ✅ VCF file generation with quality scores
- ✅ Complete in 60-90 minutes
- ✅ No AWS account needed (Colab or Studio Lab)

**Platform**: Google Colab or SageMaker Studio Lab
**Cost**: $0

[View tier-0 README →](tier-0/README.md) | [Open in Colab →](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/genomics/variant-analysis/tier-0/genomics-variant-calling.ipynb)

---

### 🟡 Tier 1: Multi-Cohort Variant Analysis (4-8 hours, FREE)
**[Launch tier-1 project →](tier-1/)**

Comprehensive variant analysis across multiple samples:
- ✅ 8-12GB multi-sample data (5-10 individuals from 1000 Genomes)
- ✅ Ensemble variant callers (CNN, Random Forest, deep learning fusion)
- ✅ Joint variant calling across cohorts
- ✅ Quality control and filtering pipelines
- ✅ Persistent storage and checkpoints (Studio Lab)
- ✅ Still free, no AWS account

**Platform**: SageMaker Studio Lab
**Cost**: $0

[View tier-1 README →](tier-1/README.md)

---

### 🟠 Tier 2: Production Variant Calling (2-3 days, $200-400/month)
**[Launch tier-2 project →](tier-2/)**

Research-grade variant calling infrastructure:
- ✅ CloudFormation one-click deployment
- ✅ 100GB+ BAM files on S3 (1000 Genomes full access, no download)
- ✅ Distributed variant calling with AWS Batch (process 100+ samples in parallel)
- ✅ SageMaker for deep learning variant calling at scale
- ✅ Automated VCF annotation with VEP (Variant Effect Predictor)
- ✅ Quality control dashboards and filtering pipelines
- ✅ Publication-ready variant catalogs

**Platform**: AWS with CloudFormation
**Cost**: $200-400/month for continuous analysis

[View tier-2 README →](tier-2/README.md)

---

### 🔴 Tier 3: Biobank-Scale Variant Platform (Ongoing, $3K-8K/month)
**[Launch tier-3 project →](tier-3/)**

Production platform for genomics research centers:
- ✅ Biobank integration (UK Biobank, TOPMed - 100K+ samples)
- ✅ Distributed variant calling across thousands of genomes
- ✅ Real-time variant annotation and clinical interpretation
- ✅ Integration with phenotype databases
- ✅ Machine learning for variant pathogenicity prediction
- ✅ AI-assisted interpretation (Amazon Bedrock for variant reports)
- ✅ Team collaboration with versioned analyses

**Platform**: AWS multi-account with enterprise support
**Cost**: $3K-8K/month

[View tier-3 README →](tier-3/README.md)

---

## What You'll Learn

Across all tiers, this project teaches:
- Deep learning for variant calling (CNNs on pileup images)
- Variant calling quality control (VQSR, hard filtering)
- VCF file manipulation and annotation
- Multi-sample joint calling and genotyping
- Variant effect prediction and clinical interpretation
- Distributed genomic analysis on cloud infrastructure

## Technologies & Tools

- **Data sources**: 1000 Genomes Project, gnomAD, UK Biobank, TOPMed, GIAB truth sets
- **Languages**: Python 3.9+
- **Core libraries**: pandas, numpy, scipy, scikit-learn
- **Genomics tools**: pysam, cyvcf2, biopython, GATK, BCFtools, VEP
- **ML frameworks**: TensorFlow/PyTorch (CNN variant calling), scikit-learn
- **Cloud services** (tier 2+): S3 (1000 Genomes public dataset), Batch (distributed calling), SageMaker (ML training), Glue, Athena, Bedrock

## Project Structure

```
variant-analysis/
├── tier-0/              # Variant calling (60-90 min, FREE)
│   ├── genomics-variant-calling.ipynb
│   ├── README.md
│   └── requirements.txt
├── tier-1/              # Multi-cohort (4-8 hours, FREE)
│   ├── notebooks/
│   ├── src/
│   ├── data/
│   ├── environment.yml
│   └── README.md
├── tier-2/              # Production (2-3 days, $200-400/mo)
│   ├── cloudformation/
│   ├── notebooks/
│   ├── src/
│   ├── tests/
│   └── README.md
└── tier-3/              # Biobank-scale (ongoing, $3K-8K/mo)
    ├── cloudformation/
    ├── notebooks/
    ├── src/
    ├── infrastructure/
    └── README.md
```

## Progression Path

```
Tier 0           → Tier 1          → Tier 2            → Tier 3
Variant Calling    Multi-Cohort       Production          Biobank-Scale
Single sample      5-10 samples       100+ samples        100K+ samples
60-90 min          4-8 hours          2-3 days            Ongoing
FREE               FREE               $200-400/mo         $3K-8K/mo
```

You can:
- ✅ Skip tiers if you have AWS experience and large-scale variant calling needs
- ✅ Stop at any tier - tier-1 is great for methods papers, tier-2 for cohort studies
- ✅ Mix and match - use tier-0 for prototyping, tier-2 for publications

[Understanding tiers →](../../../docs/projects/tiers.md)

## Variant Analysis Applications

- **Clinical diagnostics**: Identify disease-causing mutations in patient genomes
- **Population genetics**: Study allele frequency distributions and selection signatures
- **Cancer genomics**: Detect somatic mutations in tumor samples
- **Rare disease research**: Find pathogenic variants in Mendelian disorders
- **Pharmacogenomics**: Identify genetic markers for drug response
- **Agricultural genomics**: Improve crop varieties through genomic selection

## Related Projects

- **[Population Genetics](../population-genetics/)** - Population structure and selection analysis
- **[Medical - Disease Prediction](../../medical/disease-prediction/)** - Genetic risk prediction models
- **[Neuroscience - Brain Imaging](../../neuroscience/brain-imaging/)** - Neurogenetics applications

## Common Use Cases

- **Clinical researchers**: Identify pathogenic variants in patient cohorts
- **Population geneticists**: Study variant distributions across populations
- **Cancer researchers**: Detect somatic mutations and driver genes
- **Precision medicine**: Integrate variants with clinical phenotypes
- **Method developers**: Benchmark new variant calling algorithms
- **Bioinformaticians**: Build scalable variant calling pipelines

## Cost Estimates

**Tier 2 Production (100 Whole Genomes)**:
- **S3 data access** (1000 Genomes public dataset, no egress): $0
- **AWS Batch** (distributed variant calling, 100 samples × 4 hours): $150
- **SageMaker** (deep learning variant calling): ml.p3.2xlarge, 20 hours = $60
- **VEP annotation** (Lambda + DynamoDB): $30
- **Storage** (VCF files, 50GB): $1.15/month
- **Total**: $200-400/month for continuous cohort analysis

**Optimization tips**:
- Use spot instances for Batch jobs (60-70% savings)
- Cache intermediate files (pileups, gVCFs) to avoid reprocessing
- Use CRAM format instead of BAM for 50% storage savings
- Process samples in batches to optimize parallelization

## Support

- **Questions**: [GitHub Discussions](https://github.com/scttfrdmn/research-jumpstart/discussions)
- **Issues**: [GitHub Issues](https://github.com/scttfrdmn/research-jumpstart/issues)
- **Office Hours**: [Every Tuesday](../../../docs/community/office-hours.md)

## Citation

If you use this project in your research, please cite:

```bibtex
@software{research_jumpstart_variant_analysis,
  title = {Variant Analysis at Scale: Research Jumpstart},
  author = {Research Jumpstart Community},
  year = {2025},
  url = {https://github.com/scttfrdmn/research-jumpstart},
  note = {Accessed: [date]}
}
```

Also cite the appropriate data sources:
- **1000 Genomes**: https://www.internationalgenome.org/
- **gnomAD**: https://gnomad.broadinstitute.org/
- **GIAB**: https://www.nist.gov/programs-projects/genome-bottle

## License

Apache 2.0 - See [LICENSE](../../../LICENSE) for details.

---

*Part of [Research Jumpstart](https://github.com/scttfrdmn/research-jumpstart) - Pre-built research workflows for cloud computing*
