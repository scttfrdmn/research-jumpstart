# Genomic Variant Analysis - Unified Studio

Production-ready genomic variant analysis toolkit for AWS. Scale VCF analysis to thousands of samples with enterprise infrastructure.

## Overview

This project provides:
- **CloudFormation Infrastructure**: S3 buckets, IAM roles, monitoring, cost controls
- **Python Analysis Package**: VCF loading, variant filtering, annotation, visualization
- **Production Notebooks**: Reproducible analysis workflows in SageMaker
- **Public Dataset Access**: Pre-configured for 1000 Genomes, gnomAD

## Architecture

```
unified-studio/
├── cloudformation/
│   └── genomics-infrastructure.yaml    # AWS infrastructure template
├── src/
│   ├── __init__.py
│   ├── data_access.py                  # S3 and local VCF loading
│   ├── variant_analysis.py             # Analysis functions
│   └── visualization.py                # Plotting functions
├── notebooks/                          # Analysis notebooks (optional)
├── requirements.txt
├── setup.py
└── README.md
```

## Quick Start

### 1. Deploy AWS Infrastructure

```bash
# Configure AWS CLI
aws configure

# Deploy CloudFormation stack
cd cloudformation
aws cloudformation create-stack \
  --stack-name genomics-variant-analysis-dev \
  --template-body file://genomics-infrastructure.yaml \
  --parameters \
    ParameterKey=Environment,ParameterValue=dev \
    ParameterKey=MonthlyBudgetLimit,ParameterValue=50 \
  --capabilities CAPABILITY_NAMED_IAM

# Wait for stack creation
aws cloudformation wait stack-create-complete \
  --stack-name genomics-variant-analysis-dev

# Get outputs
aws cloudformation describe-stacks \
  --stack-name genomics-variant-analysis-dev \
  --query 'Stacks[0].Outputs'
```

### 2. Export Environment Variables

```bash
# From CloudFormation outputs
export DATA_BUCKET=$(aws cloudformation describe-stacks \
  --stack-name genomics-variant-analysis-dev \
  --query 'Stacks[0].Outputs[?OutputKey==`DataBucketName`].OutputValue' \
  --output text)

export RESULTS_BUCKET=$(aws cloudformation describe-stacks \
  --stack-name genomics-variant-analysis-dev \
  --query 'Stacks[0].Outputs[?OutputKey==`ResultsBucketName`].OutputValue' \
  --output text)

export ROLE_ARN=$(aws cloudformation describe-stacks \
  --stack-name genomics-variant-analysis-dev \
  --query 'Stacks[0].Outputs[?OutputKey==`AnalysisRoleArn`].OutputValue' \
  --output text)
```

### 3. Install Python Package

```bash
# Clone repository
git clone https://github.com/yourusername/research-jumpstart.git
cd research-jumpstart/projects/genomics/variant-analysis/unified-studio

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .

# Or install from requirements.txt
pip install -r requirements.txt
```

### 4. Upload VCF Files

```bash
# Upload your VCF files to the data bucket
aws s3 cp your_variants.vcf s3://${DATA_BUCKET}/vcf/your_variants.vcf

# Or sync entire directory
aws s3 sync ./vcf_files/ s3://${DATA_BUCKET}/vcf/
```

### 5. Run Analysis

```python
from src.data_access import GenomicsDataAccess
from src.variant_analysis import filter_variants_by_quality, annotate_variants
from src.visualization import plot_manhattan, plot_quality_distribution

# Initialize data access
data_access = GenomicsDataAccess(use_anon=False, region='us-east-1')

# Load VCF from S3
df = data_access.load_vcf_from_s3(
    bucket='your-data-bucket',
    key='vcf/sample.vcf'
)

# Filter variants
filtered = filter_variants_by_quality(df, min_qual=30, min_dp=10)

# Annotate
annotated = annotate_variants(filtered)

# Visualize
plot_manhattan(annotated, save_path='manhattan.png')
plot_quality_distribution(annotated, save_path='quality.png')

# Save results
data_access.save_results(
    annotated,
    bucket='your-results-bucket',
    key='analysis/annotated_variants.csv'
)
```

## Infrastructure Components

### S3 Buckets

**Data Bucket** (`genomics-variant-analysis-data-{env}-{account}`)
- Stores VCF files and genomic data
- Encryption: AES256
- Versioning: Enabled
- Lifecycle: Transition to IA after 30 days, delete after 90 days

**Results Bucket** (`genomics-variant-analysis-results-{env}-{account}`)
- Stores analysis outputs (CSV, plots, reports)
- Encryption: AES256
- Lifecycle: Delete after 90 days

### IAM Role

**AnalysisRole** (`genomics-variant-analysis-role-{env}`)
- Used by: SageMaker, Lambda, Batch
- Permissions:
  - Read/write to data and results buckets
  - Read-only access to public genomics datasets (1000 Genomes, gnomAD)
  - CloudWatch Logs write access

### Monitoring

**CloudWatch Logs** (`/aws/genomics-variant-analysis/{env}`)
- Log retention: 30 days
- Captures all analysis execution logs

**SNS Topic** (`genomics-variant-analysis-alerts-{env}`)
- Receives cost and error alerts
- Subscribe via email or Lambda

**Cost Alarm**
- Monitors S3 costs
- Threshold: $50/month (configurable)
- Action: SNS notification

## API Documentation

### GenomicsDataAccess

```python
class GenomicsDataAccess:
    """Handle loading and saving genomic variant data."""

    def __init__(self, use_anon: bool = False, region: str = 'us-east-1'):
        """Initialize data access client."""

    def load_vcf_from_s3(self, bucket: str, key: str) -> pd.DataFrame:
        """Load VCF file from S3 and parse into DataFrame."""

    def load_vcf_from_local(self, file_path: str) -> pd.DataFrame:
        """Load VCF file from local filesystem."""

    def save_results(self, df: pd.DataFrame, bucket: str, key: str):
        """Save analysis results to S3."""

    def list_vcf_files(self, bucket: str, prefix: str = '') -> List[str]:
        """List VCF files in S3 bucket."""
```

### Variant Analysis Functions

```python
def calculate_allele_frequencies(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate minor allele frequencies (MAF) for variants."""

def filter_variants_by_quality(
    df: pd.DataFrame,
    min_qual: float = 30.0,
    min_dp: int = 10,
    max_missing: float = 0.1
) -> pd.DataFrame:
    """Filter variants based on quality metrics."""

def annotate_variants(df: pd.DataFrame, add_consequences: bool = True) -> pd.DataFrame:
    """Add functional annotations to variants."""

def summarize_variants(df: pd.DataFrame) -> Dict:
    """Generate summary statistics for variant dataset."""

def identify_high_impact_variants(
    df: pd.DataFrame,
    max_af: float = 0.01,
    min_qual: float = 50.0
) -> pd.DataFrame:
    """Identify potentially high-impact rare variants."""

def compare_variant_sets(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    name1: str = "Set1",
    name2: str = "Set2"
) -> Dict:
    """Compare two variant datasets."""
```

### Visualization Functions

```python
def plot_manhattan(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    p_col: str = 'QUAL',
    sig_threshold: float = 30.0
) -> None:
    """Create Manhattan plot for variant quality across chromosomes."""

def plot_variant_density(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    bin_size: int = 1000000
) -> None:
    """Plot variant density along chromosomes."""

def plot_quality_distribution(
    df: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """Plot quality metric distributions."""

def plot_allele_frequency_spectrum(
    df: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """Plot site frequency spectrum (SFS)."""

def plot_chromosome_summary(
    df: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """Create summary bar plot of variants per chromosome."""
```

## Cost Estimates

### Development Environment (Small)
- **S3 Storage**: 100 GB VCF files → ~$2.30/month
- **S3 Requests**: 10,000 GET/PUT → ~$0.05/month
- **SageMaker Studio**: 10 hours ml.t3.medium → ~$4.60/month
- **Data Transfer**: 50 GB out → ~$4.50/month
- **CloudWatch**: Logs and monitoring → ~$3.00/month
- **Total**: ~$15-20/month

### Production Environment (Medium)
- **S3 Storage**: 1 TB VCF files → ~$23/month
- **S3 Requests**: 100,000 operations → ~$0.50/month
- **SageMaker Training**: 50 hours ml.m5.xlarge → ~$230/month
- **Data Transfer**: 200 GB out → ~$18/month
- **CloudWatch**: Enhanced monitoring → ~$10/month
- **Total**: ~$280-300/month

### Enterprise Environment (Large)
- **S3 Storage**: 10 TB genomic data → ~$230/month
- **S3 Glacier**: 50 TB archived → ~$200/month
- **SageMaker Processing**: 200 hours ml.m5.4xlarge → ~$3,680/month
- **Batch Compute**: 500 vCPU-hours → ~$150/month
- **Data Transfer**: 1 TB out → ~$90/month
- **CloudWatch**: Full monitoring → ~$30/month
- **Total**: ~$4,500-5,000/month

### Cost Optimization Tips

1. **Use Lifecycle Policies**: Transition old data to Glacier
2. **Spot Instances**: Use Spot for batch processing (70% savings)
3. **Data Transfer**: Process data in same region as storage
4. **Compression**: Use VCF.gz instead of raw VCF (90% reduction)
5. **S3 Intelligent-Tiering**: Automatic cost optimization
6. **Request Optimization**: Batch operations, use S3 Select

## Public Datasets

Pre-configured access to public genomics datasets:

### 1000 Genomes Project
```python
data_access = GenomicsDataAccess(use_anon=True)
df = data_access.load_vcf_from_s3(
    bucket='1000genomes',
    key='release/20130502/ALL.chr22.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz'
)
```

### gnomAD
```python
data_access = GenomicsDataAccess(use_anon=True)
df = data_access.load_vcf_from_s3(
    bucket='gnomad-public-us-east-1',
    key='release/2.1.1/vcf/exomes/gnomad.exomes.r2.1.1.sites.1.vcf.bgz'
)
```

## Usage Examples

### Example 1: Quality Control Pipeline

```python
from src.data_access import GenomicsDataAccess
from src.variant_analysis import filter_variants_by_quality, summarize_variants
from src.visualization import plot_quality_distribution

# Load data
data_access = GenomicsDataAccess()
df = data_access.load_vcf_from_s3('my-bucket', 'data/sample.vcf')

# QC metrics before filtering
print("Before filtering:")
print(summarize_variants(df))

# Filter
filtered = filter_variants_by_quality(df, min_qual=30, min_dp=10)

# QC metrics after filtering
print("\nAfter filtering:")
print(summarize_variants(filtered))

# Visualize
plot_quality_distribution(filtered, save_path='qc_report.png')
```

### Example 2: Rare Variant Analysis

```python
from src.variant_analysis import (
    calculate_allele_frequencies,
    identify_high_impact_variants,
    annotate_variants
)

# Calculate MAF
df = calculate_allele_frequencies(df)

# Find rare, high-impact variants
high_impact = identify_high_impact_variants(
    df,
    max_af=0.01,    # Rare: AF < 1%
    min_qual=50.0   # High confidence
)

# Annotate
annotated = annotate_variants(high_impact)

# Filter for predicted deleterious
deleterious = annotated[
    annotated['predicted_consequence'].isin([
        'frameshift_variant',
        'stop_gained',
        'splice_donor_variant'
    ])
]

print(f"Found {len(deleterious)} potentially deleterious rare variants")
```

### Example 3: Compare Two Cohorts

```python
from src.variant_analysis import compare_variant_sets

# Load case and control datasets
cases = data_access.load_vcf_from_s3('bucket', 'cases.vcf')
controls = data_access.load_vcf_from_s3('bucket', 'controls.vcf')

# Compare
comparison = compare_variant_sets(cases, controls, 'Cases', 'Controls')

print(f"Cases: {comparison['Cases_count']} variants")
print(f"Controls: {comparison['Controls_count']} variants")
print(f"Shared: {comparison['shared']} variants")
print(f"Case-specific: {comparison['Cases_unique']} variants")
print(f"Jaccard index: {comparison['jaccard_index']:.3f}")
```

## Deployment Checklist

- [ ] AWS account with appropriate permissions
- [ ] AWS CLI configured with credentials
- [ ] Deploy CloudFormation stack
- [ ] Export environment variables
- [ ] Subscribe to SNS alert topic
- [ ] Upload VCF data to S3
- [ ] Install Python package
- [ ] Run test analysis
- [ ] Set up budget alerts
- [ ] Document data retention policy
- [ ] Configure IAM user access
- [ ] Enable CloudTrail logging (optional)
- [ ] Set up automated backups

## Troubleshooting

### Issue: "Access Denied" when accessing S3
**Solution**: Verify IAM role has correct permissions. Check the S3AccessPolicy in CloudFormation template.

### Issue: VCF parsing errors
**Solution**: Ensure VCF files are properly formatted. Check for corrupted files. Use VCF validation tools.

### Issue: High S3 costs
**Solution**: Enable lifecycle policies, compress VCF files, use S3 Intelligent-Tiering, delete intermediate files.

### Issue: Slow analysis
**Solution**: Use larger instance types, batch processing, parallel execution, pre-filter data.

### Issue: Memory errors with large VCF
**Solution**: Process chromosomes separately, use chunked reading, increase instance memory, use Dask for distributed processing.

## Development

### Running Tests
```bash
pip install -e ".[dev]"
pytest tests/
```

### Code Quality
```bash
black src/
flake8 src/
mypy src/
```

## Resources

- **VCF Format**: [Specification](https://samtools.github.io/hts-specs/VCFv4.3.pdf)
- **1000 Genomes**: [Project Website](https://www.internationalgenome.org/)
- **gnomAD**: [Database](https://gnomad.broadinstitute.org/)
- **AWS Genomics**: [Best Practices](https://docs.aws.amazon.com/genomics/)
- **pysam Documentation**: [Read the Docs](https://pysam.readthedocs.io/)

## License

MIT License - See LICENSE file for details

## Support

For issues and questions:
- GitHub Issues: [Project Issues](https://github.com/yourusername/research-jumpstart/issues)
- Email: support@example.com

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{genomics_unified_studio,
  title = {Genomic Variant Analysis - Unified Studio},
  author = {Research Jumpstart},
  year = {2025},
  url = {https://github.com/yourusername/research-jumpstart}
}
```
