# Population Genetics at Scale - Tier 1 Flagship

**Duration:** 4-5 days | **Platform:** AWS Unified Studio | **Cost:** $20-50

Production-ready population genetics analysis using the complete 1000 Genomes dataset with distributed computing, variant calling pipelines, and AWS Bedrock AI for interpretation.

## Overview

This flagship project demonstrates enterprise-scale genomics research on AWS, analyzing genetic variation across 2,504 individuals from 26 populations. Using AWS services (S3, Athena, SageMaker, Bedrock), it showcases how to process terabytes of genomic data efficiently.

## What You'll Build

### Infrastructure
- **CloudFormation Stack:** Complete AWS infrastructure as code
- **S3 Data Lake:** Organized genomic data with partitioning
- **Athena Queries:** SQL interface for variant data
- **SageMaker Notebooks:** Interactive analysis environment
- **Bedrock Integration:** AI-powered variant interpretation

### Analysis Pipeline
1. **Variant Processing:** Filter and annotate millions of variants
2. **Population Structure:** PCA, ADMIXTURE, FST calculations
3. **Selection Signatures:** Identify regions under positive selection
4. **Functional Annotation:** Map variants to genes and pathways
5. **AI Interpretation:** Use Claude via Bedrock for insights

## Dataset

**1000 Genomes Project Phase 3**
- **Source:** `s3://1000genomes/phase3/`
- **Size:** 200 TB (full dataset)
- **Samples:** 2,504 individuals
- **Populations:** 26 populations, 5 super-populations (AFR, AMR, EAS, EUR, SAS)
- **Variants:** 84.7 million SNPs, 3.6 million indels
- **Format:** VCF, BAM, CRAM files

**Analysis Subset (for cost control):**
- Chromosomes 21-22: ~2 million variants
- All 2,504 individuals
- Estimated cost: $20-30 for full analysis

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    AWS Unified Studio                     │
│  ┌──────────┐    ┌──────────┐    ┌────────────────┐    │
│  │ SageMaker│───▶│  Athena  │───▶│ Bedrock Claude │    │
│  │ Notebooks│    │   SQL    │    │  Interpretation│    │
│  └──────────┘    └──────────┘    └────────────────┘    │
│        │                │                    │           │
│        └────────────────┴────────────────────┘           │
│                         │                                 │
│                         ▼                                 │
│  ┌─────────────────────────────────────────────────┐    │
│  │              S3 Data Lake                        │    │
│  │  ┌──────────────┐  ┌──────────────┐            │    │
│  │  │ 1000 Genomes │  │  Processed   │            │    │
│  │  │   Raw VCF    │  │   Variants   │            │    │
│  │  └──────────────┘  └──────────────┘            │    │
│  └─────────────────────────────────────────────────┘    │
│                                                           │
│  ┌─────────────────────────────────────────────────┐    │
│  │          Glue Data Catalog                       │    │
│  │  - Variant tables                                │    │
│  │  - Population metadata                           │    │
│  │  - Annotation databases                          │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

## Features

### 1. Scalable Variant Processing
- **Distributed VCF parsing** with Dask/Spark
- **Athena SQL queries** for variant filtering
- **Parquet optimization** for fast column access
- **Glue Catalog integration** for metadata

### 2. Population Genetics Analyses

**Principal Component Analysis (PCA):**
```python
from src.population_structure import calculate_pca

# Perform PCA on 2,504 individuals
pca_results = calculate_pca(
    vcf_path='s3://1000genomes/phase3/integrated_sv_map/',
    chromosomes=['chr21', 'chr22'],
    n_components=10
)

# Visualize population structure
plot_pca(pca_results, population_labels)
```

**FST Calculations:**
```python
from src.selection import calculate_fst

# Calculate FST between populations
fst_results = calculate_fst(
    vcf_path='s3://data/variants.parquet',
    pop1='YRI',  # Yoruba in Nigeria
    pop2='CEU',  # Utah residents (CEPH) with Northern and Western European ancestry
    window_size=50000
)
```

**ADMIXTURE Analysis:**
```python
from src.population_structure import run_admixture

# Infer ancestry proportions
admixture_results = run_admixture(
    genotype_matrix=genotypes,
    K=5,  # 5 ancestral populations
    cv_folds=5
)
```

### 3. Selection Signatures

**iHS (Integrated Haplotype Score):**
- Detects recent positive selection
- Identifies extended haplotypes
- Generates Manhattan plots

**XP-EHH (Cross-Population Extended Haplotype Homozygosity):**
- Compares selection between populations
- Identifies population-specific adaptations

**Tajima's D:**
- Tests for neutral evolution
- Detects balancing vs directional selection

### 4. Functional Annotation

**Variant Effect Prediction:**
- SnpEff/VEP annotation
- CADD scores for deleteriousness
- ClinVar pathogenicity
- dbSNP rsID mapping

**Gene-Based Analysis:**
- Map variants to genes
- Pathway enrichment (KEGG, Reactome)
- Disease association (GWAS Catalog)
- Expression QTL overlap (GTEx)

### 5. AI-Powered Interpretation

**Bedrock Claude Integration:**
```python
from src.bedrock_client import interpret_variants

# Get AI interpretation of findings
interpretation = interpret_variants(
    variant_list=significant_variants,
    population='YRI',
    analysis_type='selection_scan',
    context="Identified SNPs show strong selection signature in West African population"
)

print(interpretation)
# "These variants are located in the DARC gene region, which encodes the
#  Duffy antigen receptor. The strong selection signal in West African
#  populations is consistent with known malarial resistance adaptations..."
```

## CloudFormation Stack

### Resources Created

```yaml
Resources:
  # S3 Buckets
  - GenomicsDataLake: Stores processed variants
  - ResultsBucket: Analysis outputs

  # Glue Database
  - VariantsDatabase: Data catalog for Athena
  - PopulationMetadata: Sample annotations

  # Athena Workgroup
  - GenomicsWorkgroup: Query execution

  # SageMaker
  - NotebookInstance: ml.t3.xlarge
  - ExecutionRole: IAM permissions

  # Bedrock
  - ClaudeModelAccess: AI interpretation
```

### Deployment

```bash
# Deploy stack
aws cloudformation create-stack \
  --stack-name genomics-population-analysis \
  --template-body file://cloudformation/genomics-stack.yml \
  --parameters file://cloudformation/parameters.json \
  --capabilities CAPABILITY_IAM

# Wait for completion
aws cloudformation wait stack-create-complete \
  --stack-name genomics-population-analysis

# Get outputs
aws cloudformation describe-stacks \
  --stack-name genomics-population-analysis \
  --query 'Stacks[0].Outputs'
```

## Project Structure

```
population-genetics/unified-studio/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── setup.py                       # Package installation
│
├── cloudformation/
│   ├── genomics-stack.yml        # Main CFN template
│   └── parameters.json           # Stack parameters
│
├── src/
│   ├── __init__.py
│   ├── data_access.py            # S3/Athena data loading
│   ├── variant_processing.py     # VCF parsing, filtering
│   ├── population_structure.py   # PCA, ADMIXTURE, FST
│   ├── selection.py              # iHS, XP-EHH, Tajima's D
│   ├── annotation.py             # Functional annotation
│   ├── visualization.py          # Plots (Manhattan, PCA, etc.)
│   └── bedrock_client.py         # AI interpretation
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_population_structure.ipynb
│   ├── 03_selection_analysis.ipynb
│   ├── 04_functional_annotation.ipynb
│   └── 05_ai_interpretation.ipynb
│
└── tests/
    ├── test_variant_processing.py
    ├── test_population_structure.py
    └── test_selection.py
```

## Key Analyses

### 1. Global Population Structure

**Analysis:** PCA reveals 5 continental super-populations

```python
# Notebook: 02_population_structure.ipynb

# Load genotypes for chr21-22
genotypes = load_genotypes(chromosomes=['21', '22'])

# Calculate PCA
pca = calculate_pca(genotypes, n_components=10)

# Plot
plot_pca_by_population(pca, populations=['AFR', 'EUR', 'EAS', 'SAS', 'AMR'])
```

**Expected Results:**
- PC1: African vs Non-African (explains ~12% variance)
- PC2: East Asian vs European (explains ~8% variance)
- Clear clustering by continental ancestry

### 2. Selection in Lactase Gene (LCT)

**Analysis:** Detect positive selection for lactose tolerance in Europeans

```python
# Notebook: 03_selection_analysis.ipynb

# Calculate iHS scores for chr2 (LCT region)
ihs_scores = calculate_ihs(
    vcf_path='s3://1000genomes/phase3/...',
    chromosome='2',
    region='136545410-136594750',  # LCT gene
    population='CEU'
)

# Plot Manhattan
plot_manhattan(ihs_scores, highlight_gene='LCT')
```

**Expected Results:**
- Strong iHS signal in Europeans (CEU, GBR, TSI)
- Weak signal in Africans (YRI, LWK)
- Corresponds to known rs4988235 SNP

### 3. Malaria Resistance (DARC gene)

**Analysis:** XP-EHH between African and European populations

```python
# Calculate cross-population EHH
xp_ehh = calculate_xp_ehh(
    vcf_path='s3://1000genomes/phase3/...',
    chromosome='1',
    pop1='YRI',  # West African
    pop2='CEU',  # European
    region='159174683-159184679'  # DARC gene
)
```

**Expected Results:**
- Strong XP-EHH signal at DARC locus
- Near-fixation of Duffy-negative allele in YRI
- Protective against Plasmodium vivax malaria

### 4. GWAS Catalog Integration

**Analysis:** Overlap selection signals with known disease associations

```python
# Load GWAS Catalog
gwas_catalog = load_gwas_catalog()

# Find overlaps
overlaps = find_selection_gwas_overlaps(
    selection_regions=significant_ihs_regions,
    gwas_catalog=gwas_catalog
)

# Interpret with Bedrock
interpretation = interpret_selection_gwas(overlaps)
```

## Performance Optimization

### 1. Data Partitioning

```python
# Partition VCF by chromosome and population
partitioned_path = partition_variants(
    input_vcf='s3://1000genomes/phase3/integrated_sv_map/',
    output_path='s3://genomics-data-lake/variants/',
    partition_by=['chromosome', 'super_population']
)
```

### 2. Athena Queries

```sql
-- Fast query on partitioned data
SELECT chromosome, position, ref, alt, allele_frequency
FROM variants
WHERE chromosome = '22'
  AND super_population = 'AFR'
  AND allele_frequency > 0.05
LIMIT 1000;
```

### 3. Distributed Computing

```python
import dask.dataframe as dd

# Load 84M variants with Dask
variants_df = dd.read_parquet(
    's3://genomics-data-lake/variants/',
    columns=['CHROM', 'POS', 'REF', 'ALT', 'AF'],
    engine='pyarrow'
)

# Distributed filtering
filtered = variants_df[variants_df['AF'] > 0.05].compute()
```

## Cost Breakdown

### Data Storage (Monthly)
- **S3 Standard:** 100 GB processed data @ $0.023/GB = $2.30
- **S3 Infrequent Access:** 500 GB raw data @ $0.0125/GB = $6.25

### Compute (One-time analysis)
- **SageMaker Notebook:** ml.t3.xlarge, 20 hours @ $0.192/hr = $3.84
- **Athena Queries:** 100 GB scanned @ $5/TB = $0.50
- **Bedrock Claude:** 100K tokens @ $0.03/1K = $3.00

### Total Estimated Cost
- **Initial Setup:** $7.34
- **Full Analysis:** $7.34 + monthly storage
- **Monthly Maintenance:** $8.55 (storage only)

**Total for Complete Project:** $15-20

## Biological Insights

### 1. Lactose Tolerance Evolution

**Finding:** Strong selection for lactase persistence in Europeans

**Explanation:** Agricultural transition 10,000 years ago led to dairy consumption. Populations with livestock domestication experienced strong selection for lactose tolerance into adulthood. The rs4988235-T allele shows one of the strongest selection signatures in the human genome.

### 2. Malaria Resistance Adaptations

**Finding:** Multiple independent adaptations in malaria-endemic regions

**Examples:**
- **DARC null allele (FY*O):** Near-fixation in West Africa, protects against P. vivax
- **HbS (sickle cell):** Balancing selection maintains allele despite homozygous lethality
- **G6PD deficiency:** X-linked variant common in Mediterranean and African populations

### 3. High-Altitude Adaptation

**Finding:** Population-specific adaptations to hypoxia

**Tibetan adaptation (EPAS1, EGLN1):** Different from Andean populations, represents convergent evolution to similar environmental pressure

## Extensions

### 1. Add More Populations
- Include additional reference panels (gnomAD, HGDP)
- Incorporate ancient DNA (Reich Lab datasets)

### 2. Phenotype Integration
- UK Biobank GWAS results
- PharmGKB pharmacogenomics
- Expression QTLs from GTEx

### 3. Advanced Methods
- ADMIXTOOLS2 for admixture graphs
- ChromoPainter for fine-scale population structure
- IBDseq for identity-by-descent analysis

### 4. Machine Learning
- SageMaker AutoML for phenotype prediction
- Deep learning for variant effect prediction
- Clustering algorithms for population discovery

## Scientific References

1. **1000 Genomes Project Consortium** (2015). "A global reference for human genetic variation." *Nature* 526: 68-74.

2. **Voight et al.** (2006). "A map of recent positive selection in the human genome." *PLoS Biology* 4(3): e72.

3. **Sabeti et al.** (2007). "Genome-wide detection and characterization of positive selection in human populations." *Nature* 449: 913-918.

4. **Patterson et al.** (2012). "Ancient admixture in human history." *Genetics* 192(3): 1065-1093.

## Troubleshooting

### Issue: Athena query timeout

```python
# Solution: Partition data by chromosome
CREATE TABLE variants_partitioned
WITH (
  partitioned_by = ARRAY['chromosome'],
  format = 'PARQUET'
)
AS SELECT * FROM variants_full;
```

### Issue: Out of memory during PCA

```python
# Solution: Use randomized SVD for large matrices
from sklearn.decomposition import PCA

pca = PCA(n_components=10, svd_solver='randomized', random_state=42)
pca.fit(genotypes_matrix)
```

### Issue: Slow VCF parsing

```python
# Solution: Use cyvcf2 instead of PyVCF
from cyvcf2 import VCF

for variant in VCF('variants.vcf.gz'):
    # 10-100x faster than PyVCF
    process(variant)
```

## Support

- **Documentation:** See individual notebook READMEs
- **Issues:** GitHub Issues
- **AWS Support:** CloudFormation stack issues
- **Scientific Questions:** Consult population genetics literature

## License

This project is provided as educational material for AWS Research Jumpstart.

---

**Ready to analyze human genetic variation at scale?**

Deploy the CloudFormation stack and start exploring 2,504 genomes across 26 populations!
