# Archaeological Site Analysis with S3 and Lambda - Tier 2

**Duration:** 2-4 hours | **Cost:** $8-14 | **Platform:** AWS + Local Machine

Process and analyze archaeological artifact data using AWS serverless services. Upload artifact datasets to S3, classify and analyze with Lambda functions, store results in DynamoDB, and query with Athena—all without managing servers.

---

## What You'll Build

A cloud-native archaeological data pipeline demonstrating:

1. **Artifact Data Management** - Upload artifact records (~2GB) to S3
2. **Serverless Classification** - Lambda functions for artifact typology and analysis
3. **NoSQL Database** - Store artifact catalog in DynamoDB
4. **Spatial Queries** - Query artifacts by site, period, and location
5. **Archaeological Analysis** - Morphometric analysis, chronology, spatial patterns

This bridges the gap between local Studio Lab analysis (Tier 1) and production infrastructure (Tier 3).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Your Local Machine                                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Python Scripts                                             │ │
│  │ • upload_to_s3.py - Upload artifact data                 │ │
│  │ • lambda_function.py - Classify and analyze              │ │
│  │ • query_results.py - Query DynamoDB catalog              │ │
│  │ • archaeology_analysis.ipynb - Jupyter notebook          │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  AWS Services                                                    │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  │  S3 Bucket       │  │ Lambda Function  │  │  DynamoDB        │
│  │                  │  │                  │  │                  │
│  │ • Artifact data  │→ │ • Classification │→ │ Artifact catalog │
│  │ • Photos/images  │  │ • Morphometric   │  │ • Type, period   │
│  │ • Site maps      │  │ • Spatial        │  │ • Measurements   │
│  │ • Results        │  │ • Dating         │  │ • GPS location   │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘
│                                                      │
│  ┌──────────────────────────────────────────────────┘
│  │  Athena (SQL Queries)
│  │  • Query by type, period, site
│  │  • Spatial distribution analysis
│  │  • Chronological patterns
│  └──────────────────────────────────────────────────────────────┐
│  ┌──────────────────────────────────────────────────────────────┐
│  │  IAM Role (Permissions)                                      │
│  │  • S3 read/write                                             │
│  │  • DynamoDB read/write                                       │
│  │  • Lambda execution                                          │
│  │  • CloudWatch logging                                        │
│  └──────────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Required Knowledge
- Basic Python (pandas, boto3)
- Archaeological terminology (stratigraphy, typology, chronology)
- AWS fundamentals (S3, Lambda, IAM, DynamoDB)

### Required Tools
- **Python 3.8+** with:
  - boto3 (AWS SDK)
  - pandas (data manipulation)
  - numpy (numerical analysis)
  - scipy (statistical analysis)
  - matplotlib, seaborn (visualization)
  - scikit-learn (clustering, classification)

- **AWS Account** with:
  - S3 bucket access
  - Lambda permissions
  - DynamoDB permissions
  - IAM role creation capability
  - (Optional) Athena query capability

### Installation

```bash
# Clone the project
git clone https://github.com/research-jumpstart/research-jumpstart.git
cd research-jumpstart/projects/archaeology/site-analysis/tier-2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start (15 minutes)

### Step 1: Set Up AWS (15 minutes)
```bash
# Follow setup_guide.md for detailed instructions
# Creates:
# - S3 bucket: archaeology-data-{your-id}
# - IAM role: lambda-archaeology-processor
# - Lambda function: classify-artifacts
# - DynamoDB table: ArtifactCatalog
```

### Step 2: Upload Sample Data (3 minutes)
```bash
python scripts/upload_to_s3.py
```

### Step 3: Process Data (5 minutes)
```bash
# Lambda will automatically process uploaded artifacts
# Or manually trigger:
python scripts/invoke_lambda.py
```

### Step 4: Query Results (2 minutes)
```bash
python scripts/query_results.py
```

### Step 5: Visualize (10 minutes)
Open `notebooks/archaeology_analysis.ipynb` in Jupyter and run all cells.

---

## Detailed Workflow

### 1. Data Preparation (Setup)

**What's happening:**
- Generate sample artifact dataset or use your own
- Upload CSV files with artifact measurements
- Upload artifact images (optional)
- Create S3 bucket with proper organization

**Files involved:**
- `setup_guide.md` - Step-by-step AWS setup
- `scripts/upload_to_s3.py` - Upload automation

**Artifact data includes:**
- Artifact type (pottery, lithics, bones, coins, architecture)
- Measurements (length, width, thickness, weight)
- Material (ceramic, stone, bone, metal, etc.)
- GPS coordinates
- Stratigraphic context
- Dating information (absolute, relative)
- Site and excavation unit

**Time:** 20-30 minutes (includes setup_guide steps)

### 2. Lambda Processing

**What's happening:**
- Lambda reads artifact CSV from S3
- Performs artifact classification
  - Typological classification (pottery types, tool types, etc.)
  - Morphometric analysis (dimensions, ratios, indices)
  - Material identification
  - Period/chronology assignment
- Spatial distribution analysis
  - GPS clustering
  - Site structure patterns
- Dating and chronology
  - Relative dating from stratigraphy
  - Absolute dating integration
- Writes results to DynamoDB

**Lambda function:**
```python
# Event: S3 upload trigger
{
    'Records': [{
        'bucket': {'name': 'archaeology-data-xxxx'},
        'object': {'key': 'raw/site_A_artifacts.csv'}
    }]
}

# Processing: Classify artifacts, calculate morphometrics
# Output: DynamoDB records + summary in S3
```

**Files involved:**
- `scripts/lambda_function.py` - Classification and analysis code
- `setup_guide.md` - Lambda deployment steps

**Time:** 5-10 minutes execution

### 3. Results Storage

**What's happening:**
- Each artifact stored as DynamoDB record
- Classification results with confidence scores
- Morphometric measurements and derived indices
- Spatial coordinates for GIS analysis
- Summary statistics in S3 as JSON

**DynamoDB Schema:**
```
ArtifactCatalog Table:
- artifact_id (Primary Key)
- site_id
- artifact_type (pottery, lithic, bone, coin, architecture)
- material
- period (Neolithic, Bronze Age, Iron Age, etc.)
- measurements {length, width, thickness, weight}
- morphometric_indices {L/W ratio, thickness index, etc.}
- gps_lat, gps_lon
- stratigraphic_unit
- dating_method
- dating_value
- excavation_date
- classification_confidence
```

**S3 Structure:**
```
s3://archaeology-data-{your-id}/
├── raw/                           # Original artifact data
│   ├── site_A_artifacts.csv
│   ├── site_B_artifacts.csv
│   └── images/
│       ├── artifact_001.jpg
│       └── artifact_002.jpg
├── processed/                      # Classified data
│   ├── site_A_classified.json
│   └── site_B_classified.json
├── analysis/                       # Analysis results
│   ├── spatial_distribution.json
│   ├── chronology_summary.json
│   └── typology_summary.json
└── logs/                          # Lambda execution logs
    └── processing_log.txt
```

### 4. Results Analysis

**What's happening:**
- Query DynamoDB for artifacts by type, period, site
- Spatial analysis of artifact distribution
- Chronological analysis and seriation
- Typological clustering
- Generate publication-quality visualizations

**Files involved:**
- `notebooks/archaeology_analysis.ipynb` - Main analysis workflow
- `scripts/query_results.py` - DynamoDB queries
- (Optional) Athena queries for complex SQL analysis

**Analysis includes:**
- Artifact distribution maps
- Typological classification dendrograms
- Chronological seriation diagrams
- Morphometric scatter plots and PCA
- Site structure and activity areas
- Material sourcing patterns

**Time:** 45-60 minutes analysis

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
│   └── archaeology_analysis.ipynb    # Main analysis notebook
│
└── scripts/
    ├── upload_to_s3.py               # Upload artifact data to S3
    ├── lambda_function.py            # Lambda artifact processing
    ├── query_results.py              # Query DynamoDB catalog
    └── __init__.py
```

---

## Cost Breakdown

**Total estimated cost: $8-14 per run**

| Service | Usage | Cost |
|---------|-------|------|
| **S3 Storage** | 2GB × 7 days | $0.15 |
| **S3 Requests** | ~500 PUT/GET requests | $0.03 |
| **Lambda Executions** | 50 invocations × 2 min | $1.00 |
| **Lambda Compute** | 50 × 128MB × 120s | $1.25 |
| **DynamoDB Write** | 1000 items | $1.25 |
| **DynamoDB Storage** | 10MB × 7 days | $0.03 |
| **DynamoDB Read** | 100 queries | $0.25 |
| **Athena Queries** | 5 queries × 2GB scanned | $5.00 |
| **Data Transfer** | Download results (500MB) | $0.05 |
| **Total** | | **$9.01** |

**Cost optimization tips:**
1. Delete S3 objects after analysis ($0.15 savings)
2. Use Lambda memory efficiently (128-256MB sufficient)
3. Run Athena queries in batch (not interactive)
4. Use DynamoDB on-demand pricing
5. Delete DynamoDB table when finished ($0.03 savings)

**Free Tier Usage:**
- **S3**: First 5GB free (12 months)
- **Lambda**: 1M invocations free (12 months)
- **DynamoDB**: 25GB storage, 200M requests free (always free)
- **Athena**: First 1TB scanned free (per month)

---

## Key Learning Objectives

### AWS Services
- ✅ S3 bucket creation and management
- ✅ Lambda function deployment and triggers
- ✅ DynamoDB table design and queries
- ✅ IAM role creation with least privilege
- ✅ CloudWatch monitoring and logs
- ✅ (Optional) Athena for serverless SQL queries

### Cloud Concepts
- ✅ Object storage vs NoSQL database
- ✅ Serverless computing (no servers to manage)
- ✅ Event-driven architecture
- ✅ Scalable data processing
- ✅ Cost-conscious design

### Archaeological Skills
- ✅ Artifact typological classification
- ✅ Morphometric analysis and indices
- ✅ Spatial distribution analysis
- ✅ Chronological analysis and seriation
- ✅ Stratigraphic interpretation
- ✅ Material culture analysis

---

## Time Estimates

**First Time Setup:**
- Read setup_guide.md: 10 minutes
- Create S3 bucket: 2 minutes
- Create DynamoDB table: 3 minutes
- Create IAM role: 5 minutes
- Deploy Lambda: 5 minutes
- **Subtotal setup: 25 minutes**

**Data Processing:**
- Generate sample data: 5 minutes
- Upload data: 10 minutes
- Lambda processing: 10-15 minutes
- **Subtotal processing: 25-30 minutes**

**Analysis:**
- Query DynamoDB: 5 minutes
- Jupyter analysis: 45-60 minutes
- Generate figures: 10-15 minutes
- **Subtotal analysis: 60-80 minutes**

**Total time: 2-2.5 hours** (including setup)

---

## Prerequisites Setup

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

### Sample Data
- Sample artifact data generated automatically by scripts
- Or use your own archaeological dataset (CSV format)
- Image files optional but recommended

---

## Running the Project

### Option 1: Automated (Recommended)
```bash
# Step 1: Setup AWS services (follow prompts)
# See setup_guide.md for manual steps

# Step 2: Upload data
python scripts/upload_to_s3.py

# Step 3: Lambda processes automatically via S3 trigger
# Or manually invoke:
aws lambda invoke \
  --function-name classify-artifacts \
  --payload '{"bucket":"archaeology-data-xxxx","key":"raw/artifacts.csv"}' \
  response.json

# Step 4: Analyze results
jupyter notebook notebooks/archaeology_analysis.ipynb
```

### Option 2: Manual (Detailed Control)
```bash
# 1. Create S3 bucket manually
aws s3 mb s3://archaeology-data-$(date +%s) --region us-east-1

# 2. Create DynamoDB table
aws dynamodb create-table \
  --table-name ArtifactCatalog \
  --attribute-definitions \
    AttributeName=artifact_id,AttributeType=S \
  --key-schema AttributeName=artifact_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST

# 3. Deploy Lambda (see setup_guide.md)
# 4. Run analysis notebook
jupyter notebook notebooks/archaeology_analysis.ipynb
```

---

## Jupyter Notebook Workflow

The main analysis notebook includes:

1. **Data Generation**
   - Create sample artifact dataset
   - Multiple artifact types and periods
   - Realistic measurements and locations

2. **Upload to S3**
   - Boto3 upload with progress tracking
   - Automatic folder organization

3. **Lambda Processing**
   - Trigger classification
   - Monitor CloudWatch logs
   - Check processing status

4. **Query Results**
   - Query by artifact type
   - Filter by period and site
   - Spatial queries by location

5. **Visualization**
   - Artifact distribution maps
   - Typological dendrograms
   - Chronological seriation
   - Morphometric analysis (PCA, scatter plots)
   - Settlement pattern analysis

6. **Export**
   - Save figures for publication
   - Export data for GIS (Shapefiles, GeoJSON)
   - Generate summary reports

---

## What You'll Discover

### Archaeological Insights
- Artifact distribution patterns across sites
- Typological classification and variation
- Chronological sequences and phases
- Spatial organization of activity areas
- Material culture patterns
- Settlement structure and organization

### AWS Insights
- Serverless computing for archaeological data
- NoSQL database design for artifacts
- Cost-effective cloud analysis
- Scalable data processing
- Event-driven workflows

### Research Insights
- Reproducibility: Same code, same results
- Collaboration: Share workflows and data
- Scale: Process 10,000+ artifacts as easily as 100
- Integration: Combine with GIS and dating tools

---

## Next Steps

### Extend This Project
1. **More Sites**: Add multiple excavation sites
2. **Image Analysis**: Add computer vision for artifact photos
3. **GIS Integration**: Export to ArcGIS or QGIS
4. **Dating Integration**: Incorporate radiocarbon dates
5. **Collaboration**: Share data with team members
6. **Automation**: Auto-process daily excavation data

### Move to Tier 3 (Production)
- CloudFormation for infrastructure-as-code
- Production-grade Lambda with layers
- Multi-region deployment
- Advanced monitoring and alerting
- Cost optimization techniques
- API for external access

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
s3://archaeology-data-$(date +%s)-yourname
```

**"Lambda timeout"**
```python
# Solution: Increase timeout in Lambda console
# Default: 3 seconds
# Recommended: 300 seconds (5 minutes)
# For large datasets: 600 seconds (10 minutes)
```

**"DynamoDB ProvisionedThroughputExceededException"**
```bash
# Solution: Use on-demand billing mode
aws dynamodb update-table \
  --table-name ArtifactCatalog \
  --billing-mode PAY_PER_REQUEST
```

**"Lambda out of memory"**
```python
# Solution: Increase Lambda memory allocation
# Default: 128 MB
# Recommended: 256 MB or 512 MB for image processing
```

**"Cannot query DynamoDB by period or site"**
```bash
# Solution: Create Global Secondary Index (GSI)
aws dynamodb update-table \
  --table-name ArtifactCatalog \
  --attribute-definitions \
    AttributeName=period,AttributeType=S \
    AttributeName=site_id,AttributeType=S \
  --global-secondary-index-updates '[{
    "Create": {
      "IndexName": "period-index",
      "KeySchema": [{"AttributeName":"period","KeyType":"HASH"}],
      "Projection": {"ProjectionType":"ALL"}
    }
  }]'
```

See `docs/troubleshooting.md` for more solutions.

---

## Cleanup Guide

When finished, delete AWS resources to stop incurring charges:

```bash
# Delete S3 bucket and contents
aws s3 rm s3://archaeology-data-xxxx --recursive
aws s3 rb s3://archaeology-data-xxxx

# Delete DynamoDB table
aws dynamodb delete-table --table-name ArtifactCatalog

# Delete Lambda function
aws lambda delete-function --function-name classify-artifacts

# Delete IAM role
aws iam detach-role-policy \
  --role-name lambda-archaeology-processor \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
aws iam detach-role-policy \
  --role-name lambda-archaeology-processor \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
aws iam detach-role-policy \
  --role-name lambda-archaeology-processor \
  --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess
aws iam delete-role --role-name lambda-archaeology-processor

# Or use: python scripts/cleanup.py (automated)
```

See `cleanup_guide.md` for detailed instructions.

---

## Resources

### AWS Documentation
- [S3 Getting Started](https://docs.aws.amazon.com/s3/latest/userguide/getting-started.html)
- [Lambda Developer Guide](https://docs.aws.amazon.com/lambda/latest/dg/)
- [DynamoDB Developer Guide](https://docs.aws.amazon.com/dynamodb/)
- [IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
- [Athena User Guide](https://docs.aws.amazon.com/athena/latest/ug/)

### Archaeological Resources
- [Open Context](https://opencontext.org/) - Open archaeological data
- [tDAR](https://www.tdar.org/) - The Digital Archaeological Record
- [Archaeological Data Service](https://archaeologydataservice.ac.uk/)
- Harris Matrix (stratigraphy): [Matrix Tutorial](https://harrismatrix.wordpress.com/)

### Python Libraries
- [boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- [pandas Documentation](https://pandas.pydata.org/docs/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [GeoPandas for spatial data](https://geopandas.org/)

---

## Getting Help

### Project Issues
- GitHub Issues: https://github.com/research-jumpstart/research-jumpstart/issues
- Tag: `archaeology`, `tier-2`, `aws`

### AWS Support
- AWS Support Center: https://console.aws.amazon.com/support/
- AWS Forums: https://forums.aws.amazon.com/
- Stack Overflow: Tag `amazon-aws`, `boto3`

### Archaeological Data Help
- Open Context Forums: https://opencontext.org/about/community
- tDAR Support: https://www.tdar.org/support/

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
| **Cost** | Free | $8-14 per run |
| **Storage** | 15GB persistent | Unlimited (pay per GB) |
| **Compute** | 4GB RAM, 12-hour sessions | Scalable (pay per second) |
| **Data Scale** | Limited to 15GB | Millions of artifacts possible |
| **Database** | CSV/SQLite | DynamoDB (NoSQL) |
| **Persistence** | Session-based | Permanent S3/DynamoDB storage |
| **Collaboration** | Limited | Full team access |
| **Queries** | pandas/SQL | DynamoDB + Athena |

---

## What's Next?

After completing this project:

**Skill Building**
- AWS Lambda advanced features
- DynamoDB query optimization
- Spatial analysis with GeoPandas
- Computer vision for artifact images
- Cost optimization techniques

**Project Extensions**
- Real-time excavation data processing
- Automated artifact photo analysis
- 3D model integration
- Dating calibration workflows
- GIS integration (QGIS, ArcGIS)

**Tier 3 Transition**
- Production infrastructure with CloudFormation
- High-availability architecture
- Advanced monitoring and alerting
- Multi-region deployment
- API for data access

---

## Citation

If you use this project in your research:

```bibtex
@software{research_jumpstart_archaeology_tier2,
  title = {Archaeological Site Analysis with S3 and Lambda: Tier 2},
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

**Ready to start?** Follow the [setup_guide.md](setup_guide.md) to get started!

**Last updated:** 2025-11-14 | Research Jumpstart v1.0.0
