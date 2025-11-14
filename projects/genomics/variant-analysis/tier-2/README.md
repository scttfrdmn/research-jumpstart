# Genomic Variant Analysis with S3 and Lambda

**Duration:** 2-4 hours | **Cost:** $10-15 | **Platform:** AWS (boto3, no CloudFormation)

Analyze genomic variants at scale using AWS serverless services. Upload BAM files to S3, run variant calling with Lambda, store results in DynamoDB, and query them locally.

## Research Goal

Perform scalable variant calling on genomic regions using AWS Lambda without managing infrastructure. Learn how to:
- Store large genomic datasets in S3
- Deploy serverless variant calling functions
- Query results from DynamoDB
- Compare local vs. cloud processing

## What You'll Build

A complete serverless variant analysis pipeline:

```
1. Upload BAM files to S3
   ↓
2. Trigger Lambda for variant calling on specific regions
   ↓
3. Store VCF results and metadata in DynamoDB
   ↓
4. Query and analyze results locally
   ↓
5. Generate summary statistics and visualizations
```

## Prerequisites

### AWS Setup
- AWS account with free tier eligible or pay-as-you-go
- AWS CLI configured with credentials
- IAM permissions for S3, Lambda, DynamoDB

### Local Environment
- Python 3.8+
- Git
- ~2GB disk space for sample data
- Internet connection for AWS API calls

### Required Tools
```bash
# Install AWS CLI (if not already installed)
pip install awscli-local boto3

# Or use system package manager
# macOS: brew install awscli
# Ubuntu: apt-get install awscli
```

## Architecture Diagram

```
Your Machine (Jupyter Notebook)
    |
    +---> S3 Bucket (input BAM files + reference genome)
    |
    +---> Lambda Function (runs variant calling)
    |          ↓ triggers on S3 upload
    |          ↓ processes genomic regions
    |          ↓ generates VCF
    |
    +---> DynamoDB Table (stores variant metadata)
    |          - CHROM, POS, REF, ALT, QUAL, etc.
    |          - Query by region, quality, etc.
    |
    +---> Local Analysis
             (fetch results from DynamoDB)
             (download VCF from S3)
             (visualize and summarize)
```

## Quick Start (15 minutes)

### 1. Clone and Setup
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/research-jumpstart.git
cd research-jumpstart/projects/genomics/variant-analysis/tier-2

# Create Python environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure AWS Credentials
```bash
aws configure
# Enter your AWS Access Key ID
# Enter your AWS Secret Access Key
# Enter default region (e.g., us-east-1)
# Enter default output format (json)
```

### 3. Run Setup Guide
Follow `/setup_guide.md` step-by-step:
- Create S3 bucket
- Create DynamoDB table
- Create IAM role for Lambda
- Deploy Lambda function

### 4. Upload Sample Data
```bash
# Scripts handles this automatically
python scripts/upload_to_s3.py
```

### 5. Run Analysis Notebook
```bash
jupyter notebook notebooks/variant_analysis.ipynb
```

## File Structure

```
tier-2/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── setup_guide.md                         # Step-by-step AWS setup
├── cleanup_guide.md                       # Resource cleanup instructions
│
├── notebooks/
│   └── variant_analysis.ipynb             # Main Jupyter notebook
│                                          # - Upload BAM files
│                                          # - Invoke Lambda
│                                          # - Query DynamoDB
│                                          # - Visualize results
│
└── scripts/
    ├── upload_to_s3.py                    # Upload BAM/reference to S3
    ├── lambda_function.py                 # Lambda code for variant calling
    ├── query_results.py                   # Query and analyze DynamoDB
    └── sample_config.json                 # Configuration template
```

## Key Features

### Serverless Variant Calling
- Lambda executes variant calling on demand
- No EC2 instances to manage
- Pay only for execution time (~$0.20 per 100 invocations)

### Scalable Storage
- S3 stores BAM files and results
- Automatic cleanup with lifecycle policies
- Version control for reproducibility

### NoSQL Metadata
- DynamoDB stores variant information
- Query by chromosome, position, quality
- Real-time analysis without VCF re-parsing

### Local Analysis
- Download and analyze results on your machine
- Generate publication-ready visualizations
- Compare multiple runs

## AWS Services Used

| Service | Purpose | Cost |
|---------|---------|------|
| **S3** | Store BAM files, reference genome, VCF results | ~$0.023/GB/month for storage |
| **Lambda** | Run variant calling serverless functions | ~$0.0000002 per invocation |
| **DynamoDB** | NoSQL database for variant metadata | ~$1.25/GB/month for on-demand |
| **IAM** | Access control and permissions | Free |
| **CloudWatch** | Logging and monitoring | Free tier included |

## Cost Breakdown

For a typical 2-4 hour project run:

| Component | Quantity | Cost |
|-----------|----------|------|
| S3 Storage (10GB BAM files, 1 week) | 10 GB | $0.23 |
| S3 Data Transfer (download results) | 1-2 GB | $0.09-$0.18 |
| Lambda Invocations (100 calls @ 30 sec each) | 100 calls | $0.20 |
| Lambda Duration (100 calls × 30 sec) | 3000 seconds | $0.04 |
| DynamoDB Storage (1000 variants, 1 week) | ~100 KB | $0.01 |
| CloudWatch Logs | ~10 MB | Free |
| **Total** | | **$10-15** |

## Time Estimate

| Phase | Duration | Notes |
|-------|----------|-------|
| AWS Setup (one-time) | 20 minutes | S3, DynamoDB, Lambda, IAM |
| Data Upload | 10-20 minutes | ~10GB BAM files to S3 |
| Lambda Deployment | 5 minutes | Upload code and dependencies |
| Notebook Execution | 60-90 minutes | Query, analyze, visualize |
| Cleanup | 5 minutes | Delete AWS resources |
| **Total** | 2-4 hours | First-time run |

## What You'll Learn

### Technical Skills
- How to use boto3 for AWS service interactions
- Deploy and test Lambda functions
- Design NoSQL databases with DynamoDB
- Manage AWS IAM policies and roles
- Monitor costs and resource usage

### Genomics Skills
- Variant calling fundamentals (SNPs, indels)
- VCF file format parsing
- Quality filtering and annotation
- Population genetics analysis

### AWS Architecture
- Event-driven computing (Lambda)
- Serverless cost models
- NoSQL database design
- S3 object lifecycle management

## Next Steps

### After Completing Tier 2
1. **Scale to production** with Tier 3 CloudFormation
2. **Process 1000+ samples** with AWS Batch
3. **Add machine learning** with SageMaker
4. **Implement CI/CD** with AWS CodePipeline

### Explore Other Tier 2 Projects
- Climate Data Analysis (S3 + Lambda + Athena)
- Medical Image Analysis (S3 + Lambda + DynamoDB)
- Astronomy Source Detection (S3 + Lambda)

## Troubleshooting

### Common Issues

**Q: "Access Denied" when uploading to S3?**
A: Check AWS credentials and IAM permissions. Ensure your user has S3 PutObject access.

**Q: Lambda function times out?**
A: Increase Lambda timeout from 30 seconds to 5 minutes in AWS console. Large BAM regions may need more time.

**Q: DynamoDB queries return no results?**
A: Ensure Lambda function is properly writing to DynamoDB. Check CloudWatch logs for errors.

**Q: High costs at the end?**
A: Follow cleanup_guide.md to delete S3 objects, DynamoDB tables, and Lambda functions immediately.

**Q: Can't find sample data?**
A: Sample BAM files are downloaded from public repositories. Check internet connection and S3 access permissions.

### Debugging Tips

1. **Check Lambda CloudWatch Logs**
   ```bash
   aws logs tail /aws/lambda/variant-calling --follow
   ```

2. **Monitor S3 bucket**
   ```bash
   aws s3 ls s3://your-bucket-name/ --recursive
   ```

3. **Query DynamoDB table**
   ```bash
   aws dynamodb scan --table-name variant-metadata
   ```

4. **Check AWS costs in real-time**
   - AWS Console > Cost Explorer > Last 7 days

## Resources

- **VCF Format**: [Specification (v4.3)](https://samtools.github.io/hts-specs/VCFv4.3.pdf)
- **boto3 Documentation**: [AWS SDK for Python](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- **Lambda Limits**: [AWS Lambda Quotas](https://docs.aws.amazon.com/lambda/latest/dg/limits.html)
- **DynamoDB Best Practices**: [AWS DynamoDB](https://docs.aws.amazon.com/dynamodb/)
- **1000 Genomes BAM Files**: [Registry of Open Data](https://www.internationalgenome.org/)

## Support

For questions or issues:
1. Check the troubleshooting section above
2. Review setup_guide.md for configuration details
3. Open a GitHub issue in the repository
4. Consult AWS documentation

## Next: AWS Setup

**Ready to get started?** Follow `/setup_guide.md` for detailed step-by-step instructions to create and configure all AWS resources.

---

Built with [Claude Code](https://claude.com/claude-code) | MIT License | 2025
