# Population Genetics at Scale

**Tier 1 Flagship Project**

Large-scale population genetics analysis using the 1000 Genomes Project and other public genomic datasets on AWS.

## Quick Start

The main implementation is in the [`unified-studio/`](unified-studio/) directory.

**[→ View Full Documentation](unified-studio/README.md)**

## Features

- 1000 Genomes Project data access (84.7M variants, 2,504 individuals)
- Population structure analysis (PCA, ADMIXTURE)
- Selection scan detection (iHS, XP-EHH, Tajima's D)
- GWAS integration and analysis
- AI-powered interpretation with Amazon Bedrock

## Cost Estimate

**$20-50** for complete analysis of chromosome 22 across all populations

## Technologies

- **Data:** 1000 Genomes (VCF, CRAM), gnomAD, UK Biobank
- **Analysis:** scikit-allel, pysam, cyvcf2, PLINK
- **AWS:** S3, Glue, Athena, SageMaker, Bedrock
- **Compute:** c5.4xlarge recommended for analysis

## Quick Links

- [Getting Started Guide](unified-studio/README.md#getting-started)
- [Core Analyses](unified-studio/README.md#core-analyses)
- [CloudFormation Template](unified-studio/cloudformation/genomics-stack.yml)
- [Source Code](unified-studio/src/)
