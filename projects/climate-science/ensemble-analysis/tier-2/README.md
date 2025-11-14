# Climate Data Analysis with S3 and Lambda - Tier 2 Project

**Duration:** 2-4 hours | **Cost:** $8-12 | **Platform:** AWS + Local machine

Analyze CMIP6 climate model data using serverless AWS services. Upload climate datasets to S3, process netCDF files with Lambda, and query results using Athena—all without managing servers.

---

## What You'll Build

A cloud-native climate data analysis pipeline that demonstrates:

1. **Data Storage** - Upload ~5GB CMIP6 subset to S3
2. **Serverless Processing** - Lambda functions to process netCDF files in parallel
3. **Results Storage** - Store processed results back to S3
4. **Data Querying** - Query results with Athena or Jupyter notebook

This bridges the gap between local analysis (Tier 1) and production infrastructure (Tier 3).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Your Local Machine                                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Python Scripts                                             │ │
│  │ • upload_to_s3.py - Upload CMIP6 data                    │ │
│  │ • lambda_function.py - Process netCDF files              │ │
│  │ • query_results.py - Analyze results                     │ │
│  │ • climate_analysis.ipynb - Jupyter notebook              │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  AWS Services                                                    │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  │  S3 Bucket       │  │ Lambda Function  │  │  Athena (SQL)    │
│  │                  │  │                  │  │                  │
│  │ • Raw data       │→ │ netCDF           │→ │ Query results    │
│  │ • Processed      │  │ processing       │  │ (optional)       │
│  │   results        │  │ (serverless)     │  │                  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘
│  ┌──────────────────────────────────────────────────────────────┐
│  │  IAM Role (Permissions)                                      │
│  │  • S3 read/write                                             │
│  │  • Lambda execution                                          │
│  │  • CloudWatch logging                                        │
│  └──────────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Required Knowledge
- Basic Python (pandas, boto3)
- Understanding of netCDF files
- AWS fundamentals (S3, Lambda, IAM)

### Required Tools
- **Python 3.8+** with:
  - boto3 (AWS SDK)
  - xarray (netCDF handling)
  - pandas (data manipulation)
  - matplotlib (visualization)
  - netCDF4 (netCDF support)

- **AWS Account** with:
  - S3 bucket access
  - Lambda permissions
  - IAM role creation capability
  - (Optional) Athena query capability

### Installation

```bash
# Clone the project
git clone https://github.com/research-jumpstart/research-jumpstart.git
cd research-jumpstart/projects/climate-science/ensemble-analysis/tier-2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start (15 minutes)

### Step 1: Set Up AWS (10 minutes)
```bash
# Follow setup_guide.md for detailed instructions
# Creates:
# - S3 bucket: climate-data-{your-id}
# - IAM role: lambda-climate-processor
# - Lambda function: process-climate-data
```

### Step 2: Upload Sample Data (3 minutes)
```bash
python scripts/upload_to_s3.py
```

### Step 3: Process Data (2 minutes)
```bash
python scripts/lambda_function.py
```

### Step 4: Query Results (2 minutes)
```bash
python scripts/query_results.py
```

### Step 5: Visualize (5 minutes)
Open `notebooks/climate_analysis.ipynb` in Jupyter and run all cells.

---

## Detailed Workflow

### 1. Data Preparation (Setup)

**What's happening:**
- Download a small CMIP6 dataset (~5GB)
- Convert to S3-friendly format if needed
- Create S3 bucket with proper permissions

**Files involved:**
- `setup_guide.md` - Step-by-step AWS setup
- `scripts/upload_to_s3.py` - Upload automation

**Time:** 20-30 minutes (includes setup_guide steps)

### 2. Lambda Processing

**What's happening:**
- Lambda reads netCDF files from S3
- Extracts temperature and precipitation data
- Calculates regional statistics
- Writes results back to S3 (CSV/JSON format)

**Lambda function**:
```python
# Event: S3 upload trigger
{
    'Records': [{
        'bucket': {'name': 'climate-data-xxxx'},
        'object': {'key': 'raw/CESM2_temperature_2100.nc'}
    }]
}

# Processing: Extract regional mean temperature
# Output: results/CESM2_temperature_2100_processed.json
```

**Files involved:**
- `scripts/lambda_function.py` - Process function code
- `setup_guide.md` - Lambda deployment steps

**Time:** 5-10 minutes execution

### 3. Results Storage

**What's happening:**
- Processed results stored in S3 as JSON/CSV
- Original netCDF files kept for reference
- Organized folder structure for easy access

**S3 Structure:**
```
s3://climate-data-{your-id}/
├── raw/                          # Original netCDF files
│   ├── CESM2_temperature_2100.nc
│   ├── GFDL_temperature_2100.nc
│   └── ...
├── results/                       # Processed CSV/JSON
│   ├── CESM2_statistics.json
│   ├── GFDL_statistics.json
│   └── ...
└── logs/                          # Lambda execution logs
    └── processing_log.txt
```

### 4. Results Analysis

**What's happening:**
- Download results to local machine
- Analyze with Jupyter notebook
- Create publication-quality figures
- (Optional) Query with Athena using SQL

**Files involved:**
- `notebooks/climate_analysis.ipynb` - Analysis notebook
- `scripts/query_results.py` - Download and analyze
- (Optional) Athena queries for direct SQL access

**Time:** 30-45 minutes analysis

---

## Project Files

```
tier-2/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup_guide.md                     # AWS setup instructions
├── cleanup_guide.md                   # Resource deletion guide
│
├── notebooks/
│   └── climate_analysis.ipynb        # Main analysis notebook
│
├── scripts/
│   ├── upload_to_s3.py               # Upload data to S3
│   ├── lambda_function.py            # Lambda processing function
│   ├── query_results.py              # Download and analyze results
│   └── __init__.py
│
├── sample_data/
│   └── README.md                     # Sample data documentation
│
└── docs/
    ├── architecture.md               # Detailed architecture
    ├── cost_breakdown.md             # Detailed cost analysis
    └── troubleshooting.md            # Common issues
```

---

## Cost Breakdown

**Total estimated cost: $8-12 per run**

| Service | Usage | Cost |
|---------|-------|------|
| **S3 Storage** | 5GB × 7 days | $0.35 |
| **S3 Requests** | ~1000 PUT/GET requests | $0.05 |
| **Lambda Executions** | 100 invocations × 1 min | $1.50 |
| **Lambda Compute** | 100 GB-seconds | $2.00 |
| **Data Transfer** | Upload + download (5GB) | $0.50 |
| **Athena Queries** | 10 queries × 10GB scanned | $5.00 |
| **Total** | | **$9.40** |

**Cost optimization tips:**
1. Delete S3 objects after analysis ($0.35 savings)
2. Use Lambda for < 1 minute processing
3. Run Athena queries in batch (not interactive)
4. Use S3 Intelligent-Tiering for long-term storage

**Free Tier Usage:**
- **S3**: First 5GB free (12 months)
- **Lambda**: 1M invocations free (12 months)
- **Athena**: First 1TB scanned free (per month)

---

## Key Learning Objectives

### AWS Services
- ✅ S3 bucket creation and management
- ✅ Lambda function deployment and triggers
- ✅ IAM role creation with least privilege
- ✅ CloudWatch monitoring and logs
- ✅ (Optional) Athena for serverless SQL queries

### Cloud Concepts
- ✅ Object storage vs block storage
- ✅ Serverless computing (no servers to manage)
- ✅ Event-driven architecture
- ✅ Parallelizable workloads
- ✅ Cost-conscious design

### Climate Data Skills
- ✅ netCDF file handling
- ✅ Regional statistics calculation
- ✅ Ensemble analysis
- ✅ Climate data visualization

---

## Time Estimates

**First Time Setup:**
- Read setup_guide.md: 10 minutes
- Create S3 bucket: 2 minutes
- Create IAM role: 5 minutes
- Deploy Lambda: 5 minutes
- Configure S3 upload: 3 minutes
- **Subtotal setup: 25 minutes**

**Data Processing:**
- Upload data: 15-30 minutes (5GB, depends on connection)
- Lambda processing: 5-10 minutes
- **Subtotal processing: 20-40 minutes**

**Analysis:**
- Query results: 5 minutes
- Jupyter analysis: 30-45 minutes
- Generate figures: 10-15 minutes
- **Subtotal analysis: 45-65 minutes**

**Total time: 1.5-2.5 hours** (including setup)

---

## Prerequisites

### AWS Account Setup
1. Create AWS account: https://aws.amazon.com/
2. (Optional) Activate free tier: https://console.aws.amazon.com/billing/
3. Create IAM user for programmatic access

### Local Setup
```bash
# Install Python 3.8+ (if needed)
python --version

# Install AWS CLI
pip install awscli

# Configure AWS credentials
aws configure
# Enter: Access Key ID
# Enter: Secret Access Key
# Enter: Region (us-east-1 recommended)
# Enter: Output format (json)
```

### Data Preparation
- Sample CMIP6 data provided in `sample_data/`
- Or download your own: https://pcmdi.llnl.gov/CMIP6/

---

## Running the Project

### Option 1: Automated (Recommended for First Time)
```bash
# Step 1: Setup AWS services (follow prompts)
python setup.py

# Step 2: Upload data
python scripts/upload_to_s3.py

# Step 3: Deploy Lambda (follow setup_guide.md)
# Manual: Deploy scripts/lambda_function.py to Lambda console

# Step 4: Analyze results
jupyter notebook notebooks/climate_analysis.ipynb
```

### Option 2: Manual (Detailed Control)
```bash
# 1. Create S3 bucket manually
aws s3 mb s3://climate-data-$(date +%s) --region us-east-1

# 2. Upload data
aws s3 cp sample_data/ s3://climate-data-xxxx/raw/ --recursive

# 3. Deploy Lambda (see setup_guide.md)
# 4. Run analysis notebook
jupyter notebook notebooks/climate_analysis.ipynb
```

---

## Jupyter Notebook Workflow

The main analysis notebook includes:

1. **Data Loading**
   - Download results from S3
   - Load into pandas/xarray

2. **Analysis**
   - Temperature anomaly calculation
   - Regional statistics
   - Ensemble metrics

3. **Visualization**
   - Time series plots
   - Regional maps
   - Uncertainty quantification

4. **Export**
   - Save figures
   - Generate reports

---

## What You'll Discover

### Climate Insights
- How different climate models project future temperatures
- Regional variability in climate projections
- Ensemble uncertainty quantification
- Model agreement assessment

### AWS Insights
- Serverless computing advantages
- Parallelizable workflow design
- Cost-effective cloud analysis
- Scale from local to petabyte-scale data

### Research Insights
- Reproducibility: Same code, same results
- Collaboration: Share workflows and results
- Scale: Process 100GB datasets as easily as 1GB
- Persistence: Results saved permanently in cloud

---

## Next Steps

### Extend This Project
1. **More Models**: Add more CMIP6 models to analysis
2. **More Variables**: Include precipitation, sea ice, etc.
3. **Different Scenarios**: Compare SSP1-2.6 vs SSP5-8.5
4. **Automation**: Set up Lambda trigger on S3 upload
5. **Notifications**: Email results using SNS

### Move to Tier 3 (Production)
- CloudFormation for infrastructure-as-code
- Production-grade Lambda functions
- Multi-region deployment
- Advanced monitoring and alerting
- Cost optimization techniques

---

## Troubleshooting

### Common Issues

**"botocore.exceptions.NoCredentialsError"**
```bash
# Solution: Configure AWS credentials
aws configure
# Enter your Access Key ID, Secret Key, region, output format
```

**"S3 bucket already exists"**
```bash
# Solution: Use a unique bucket name
s3://climate-data-$(date +%s)-yourname
```

**"Lambda timeout"**
```python
# Solution: Increase timeout in Lambda console
# Default: 3 seconds
# Recommended: 300 seconds (5 minutes)
# For larger files: 600 seconds (10 minutes)
```

**"Out of memory in Lambda"**
```python
# Solution: Increase Lambda memory allocation
# Default: 128 MB
# Recommended: 512 MB or 1024 MB
# Cost increases slightly with memory
```

**"Data too large for Lambda"**
```python
# Solution: Use Step Functions for complex workflows
# Or process in batches with multiple Lambda invocations
```

See `docs/troubleshooting.md` for more solutions.

---

## Cleanup Guide

When finished, delete AWS resources to stop incurring charges:

```bash
# Delete S3 bucket and contents
aws s3 rm s3://climate-data-xxxx --recursive
aws s3 rb s3://climate-data-xxxx

# Delete Lambda function
aws lambda delete-function --function-name process-climate-data

# Delete IAM role
aws iam delete-role-policy --role-name lambda-climate-processor \
  --policy-name lambda-policy
aws iam delete-role --role-name lambda-climate-processor

# Or use: python cleanup.py (automated)
```

See `cleanup_guide.md` for detailed instructions.

---

## Resources

### AWS Documentation
- [S3 Getting Started](https://docs.aws.amazon.com/s3/latest/userguide/getting-started.html)
- [Lambda Developer Guide](https://docs.aws.amazon.com/lambda/latest/dg/)
- [IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
- [Athena User Guide](https://docs.aws.amazon.com/athena/latest/ug/)

### Climate Data
- [CMIP6 on AWS](https://registry.opendata.aws/cmip6/)
- [CMIP6 Documentation](https://pcmdi.llnl.gov/CMIP6/)
- [xarray Tutorial](https://tutorial.xarray.dev/)

### Python Libraries
- [boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- [xarray Documentation](https://xarray.pydata.org/)
- [pandas Documentation](https://pandas.pydata.org/docs/)

---

## Getting Help

### Project Issues
- GitHub Issues: https://github.com/research-jumpstart/research-jumpstart/issues
- Tag: `climate-science`, `tier-2`, `aws`

### AWS Support
- AWS Support Center: https://console.aws.amazon.com/support/
- AWS Forums: https://forums.aws.amazon.com/
- Stack Overflow: Tag `amazon-aws`, `boto3`

### Climate Data Help
- Pangeo Discourse: https://discourse.pangeo.io/
- CMIP6 Forum: https://www.wcrp-climate.org/

---

## Cost Tracking

### Monitor Your Spending

```bash
# Check current AWS charges
aws ce get-cost-and-usage \
  --time-period Start=2025-01-01,End=2025-01-31 \
  --granularity MONTHLY \
  --metrics "BlendedCost"

# Set up billing alerts in AWS console:
# https://docs.aws.amazon.com/billing/latest/userguide/budgets-create.html
```

Recommended alerts:
- $10 threshold (warning)
- $25 threshold (warning)
- $50 threshold (critical)

---

## Key Differences from Tier 1

| Aspect | Tier 1 (Studio Lab) | Tier 2 (AWS) |
|--------|-------------------|------------|
| **Platform** | Free SageMaker Studio Lab | AWS account required |
| **Cost** | Free | $8-12 per run |
| **Storage** | 15GB persistent | Unlimited (pay per GB) |
| **Compute** | 4GB RAM, 12-hour sessions | Scalable (pay per second) |
| **Data Scale** | Limited to 15GB | Petabytes possible |
| **Parallelization** | Single notebook | Multiple Lambda functions |
| **Persistence** | Session-based | Permanent S3 storage |
| **Collaboration** | Limited | Full team access |

---

## What's Next?

After completing this project:

**Skill Building**
- AWS Lambda advanced features
- Serverless architecture patterns
- Cost optimization techniques
- Security best practices

**Project Extensions**
- Real-time climate data processing
- Automated analysis pipelines
- Integration with other services (SNS, SQS)
- Dashboard creation (QuickSight, Grafana)

**Tier 3 Transition**
- Production infrastructure with CloudFormation
- High-availability architecture
- Advanced monitoring and alerting
- Multi-region deployment

---

## Citation

If you use this project in your research:

```bibtex
@software{research_jumpstart_climate_tier2,
  title = {Climate Data Analysis with S3 and Lambda: Tier 2},
  author = {Research Jumpstart Community},
  year = {2025},
  url = {https://github.com/research-jumpstart/research-jumpstart},
  note = {Accessed: [date]}
}
```

---

## License

Apache License 2.0 - See [LICENSE](../../../../LICENSE) for details.

---

**🚀 Ready to start?** Follow the [setup_guide.md](setup_guide.md) to get started!

**Last updated:** 2025-11-14 | Research Jumpstart v1.0.0
