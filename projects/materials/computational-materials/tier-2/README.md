# Materials Property Prediction with S3 and Lambda - Tier 2 Project

**Duration:** 2-4 hours | **Cost:** $8-12 | **Platform:** AWS + Local machine

Predict materials properties using serverless AWS services. Upload crystal structures to S3, compute properties with Lambda, store results in DynamoDB, and query them for analysis—all without managing servers.

---

## What You'll Build

A cloud-native materials property prediction pipeline that demonstrates:

1. **Data Storage** - Upload crystal structure files (CIF/POSCAR) to S3
2. **Serverless Processing** - Lambda functions to calculate material properties
3. **Results Storage** - Store computed properties in DynamoDB
4. **Data Querying** - Query materials by properties (density, space group, etc.)

This bridges the gap between local analysis (Tier 1) and production infrastructure (Tier 3).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Your Local Machine                                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Python Scripts                                             │ │
│  │ • upload_to_s3.py - Upload crystal structures            │ │
│  │ • lambda_function.py - Compute properties                │ │
│  │ • query_results.py - Query DynamoDB                      │ │
│  │ • materials_analysis.ipynb - Jupyter notebook            │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  AWS Services                                                    │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  │  S3 Bucket       │  │ Lambda Function  │  │  DynamoDB        │
│  │                  │  │                  │  │                  │
│  │ • CIF/POSCAR     │→ │ Crystal structure│→ │ Materials        │
│  │   structures     │  │ property         │  │ properties       │
│  │ • Reference data │  │ calculation      │  │ (queryable)      │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘
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
- Understanding of crystal structures (CIF format)
- AWS fundamentals (S3, Lambda, DynamoDB, IAM)

### Required Tools
- **Python 3.8+** with:
  - boto3 (AWS SDK)
  - pymatgen (crystal structure analysis)
  - pandas (data manipulation)
  - matplotlib (visualization)
  - numpy (numerical computations)

- **AWS Account** with:
  - S3 bucket access
  - Lambda permissions
  - DynamoDB access
  - IAM role creation capability

### Installation

```bash
# Clone the project
git clone https://github.com/research-jumpstart/research-jumpstart.git
cd research-jumpstart/projects/materials/computational-materials/tier-2

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
# - S3 bucket: materials-data-{your-id}
# - DynamoDB table: MaterialsProperties
# - IAM role: lambda-materials-processor
# - Lambda function: process-crystal-structure
```

### Step 2: Upload Sample Data (2 minutes)
```bash
python scripts/upload_to_s3.py --bucket materials-data-{your-id}
```

### Step 3: Process Structures (2 minutes)
```bash
# Lambda processes automatically via S3 trigger
# Or invoke manually:
python scripts/trigger_lambda.py
```

### Step 4: Query Results (1 minute)
```bash
python scripts/query_results.py --property density --min 5.0 --max 10.0
```

### Step 5: Visualize (5 minutes)
Open `notebooks/materials_analysis.ipynb` in Jupyter and run all cells.

---

## Detailed Workflow

### 1. Data Preparation (Setup)

**What's happening:**
- Download or prepare crystal structure files (CIF/POSCAR format)
- Upload to S3 bucket with proper organization
- Create DynamoDB table for property storage

**Files involved:**
- `setup_guide.md` - Step-by-step AWS setup
- `scripts/upload_to_s3.py` - Upload automation

**Time:** 15-20 minutes (includes setup_guide steps)

### 2. Lambda Processing

**What's happening:**
- Lambda reads crystal structure files from S3
- Parses CIF/POSCAR format to extract atomic coordinates
- Calculates material properties:
  - Density (g/cm³)
  - Volume (Å³)
  - Space group
  - Lattice parameters
  - Chemical formula
  - Number of atoms
- Stores results in DynamoDB for fast queries

**Lambda function**:
```python
# Event: S3 upload trigger
{
    'Records': [{
        'bucket': {'name': 'materials-data-xxxx'},
        'object': {'key': 'structures/mp-1234.cif'}
    }]
}

# Processing: Extract crystal properties
# Output: DynamoDB entry with computed properties
```

**Files involved:**
- `scripts/lambda_function.py` - Property calculation code
- `setup_guide.md` - Lambda deployment steps

**Time:** 2-5 minutes per 100 structures

### 3. Results Storage

**What's happening:**
- Computed properties stored in DynamoDB
- Indexed by material ID for fast lookup
- Original structures kept in S3 for reference
- Results queryable by any property

**DynamoDB Schema:**
```json
{
  "material_id": "mp-1234",
  "formula": "Si",
  "space_group": "Fd-3m (227)",
  "density": 2.33,
  "volume": 40.88,
  "lattice_a": 3.867,
  "lattice_b": 3.867,
  "lattice_c": 3.867,
  "num_atoms": 2,
  "crystal_system": "cubic",
  "s3_key": "structures/mp-1234.cif",
  "processed_at": "2025-01-14T10:30:00Z"
}
```

**S3 Structure:**
```
s3://materials-data-{your-id}/
├── structures/                    # Crystal structure files
│   ├── mp-1234.cif
│   ├── mp-5678.cif
│   └── ...
├── reference/                     # Reference data
│   ├── elements.json
│   └── space_groups.json
└── logs/                          # Processing logs
    └── processing_log.txt
```

### 4. Results Analysis

**What's happening:**
- Query DynamoDB for materials matching criteria
- Download structures from S3 for detailed analysis
- Visualize property distributions
- Compare materials by composition or properties

**Files involved:**
- `notebooks/materials_analysis.ipynb` - Analysis notebook
- `scripts/query_results.py` - Query and analyze

**Time:** 30-60 minutes analysis

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
│   └── materials_analysis.ipynb      # Main analysis notebook
│
├── scripts/
│   ├── upload_to_s3.py               # Upload structures to S3
│   ├── lambda_function.py            # Lambda processing function
│   ├── query_results.py              # Query DynamoDB
│   └── __init__.py
│
└── sample_data/
    ├── README.md                     # Sample data documentation
    └── structures/                   # Sample CIF files
```

---

## Cost Breakdown

**Total estimated cost: $8-12 per run**

| Service | Usage | Cost |
|---------|-------|------|
| **S3 Storage** | 1GB × 7 days | $0.07 |
| **S3 Requests** | ~1000 PUT/GET requests | $0.05 |
| **Lambda Executions** | 500 invocations × 30 sec | $0.10 |
| **Lambda Compute** | 500 × 512MB × 30s = 7500 GB-sec | $1.25 |
| **DynamoDB Storage** | 500 items × 1KB = 500KB | $0.01 |
| **DynamoDB Writes** | 500 write requests | $0.63 |
| **DynamoDB Reads** | 1000 read requests | $0.25 |
| **Data Transfer** | Upload + download (1GB) | $0.10 |
| **CloudWatch Logs** | ~50 MB | Free |
| **Total** | | **$2.46** |

**Cost optimization tips:**
1. Delete S3 objects after analysis ($0.07 savings)
2. Use DynamoDB on-demand pricing (pay per request)
3. Keep Lambda timeout to 1 minute per structure
4. Process structures in batches of 100

**Free Tier Usage:**
- **S3**: First 5GB free (12 months)
- **Lambda**: 1M invocations free (12 months)
- **DynamoDB**: 25GB storage + 25 WCU + 25 RCU free (always)

---

## Key Learning Objectives

### AWS Services
- ✅ S3 bucket creation and object management
- ✅ Lambda function deployment and event triggers
- ✅ DynamoDB table design and queries
- ✅ IAM role creation with least privilege
- ✅ CloudWatch monitoring and logs

### Cloud Concepts
- ✅ Object storage for scientific data
- ✅ Serverless computing (no servers to manage)
- ✅ NoSQL databases for flexible schemas
- ✅ Event-driven architecture
- ✅ Cost-conscious design patterns

### Materials Science Skills
- ✅ Crystal structure file formats (CIF, POSCAR)
- ✅ Space group determination
- ✅ Density and volume calculations
- ✅ Lattice parameter analysis
- ✅ Materials property prediction

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
- Upload structures: 5-10 minutes (100 files)
- Lambda processing: 5-10 minutes (automatic)
- **Subtotal processing: 10-20 minutes**

**Analysis:**
- Query results: 5 minutes
- Jupyter analysis: 30-45 minutes
- Generate figures: 10-15 minutes
- **Subtotal analysis: 45-65 minutes**

**Total time: 1.5-2 hours** (including setup)

---

## Running the Project

### Option 1: Automated (Recommended for First Time)
```bash
# Step 1: Setup AWS services (follow setup_guide.md)
# Manual: Create S3, DynamoDB, Lambda, IAM via console

# Step 2: Upload data
python scripts/upload_to_s3.py \
  --bucket materials-data-{your-id} \
  --directory sample_data/structures

# Step 3: Lambda processes automatically via S3 trigger
# Or trigger manually:
aws lambda invoke \
  --function-name process-crystal-structure \
  --payload '{"test": true}' \
  response.json

# Step 4: Analyze results
jupyter notebook notebooks/materials_analysis.ipynb
```

### Option 2: Manual (Detailed Control)
```bash
# 1. Create S3 bucket manually
aws s3 mb s3://materials-data-$(date +%s) --region us-east-1

# 2. Upload structures
aws s3 cp sample_data/structures/ \
  s3://materials-data-xxxx/structures/ --recursive

# 3. Create DynamoDB table
aws dynamodb create-table \
  --table-name MaterialsProperties \
  --attribute-definitions \
    AttributeName=material_id,AttributeType=S \
  --key-schema AttributeName=material_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST

# 4. Deploy Lambda (see setup_guide.md)
# 5. Run analysis notebook
jupyter notebook notebooks/materials_analysis.ipynb
```

---

## Jupyter Notebook Workflow

The main analysis notebook includes:

1. **Data Loading**
   - Query DynamoDB for materials
   - Download structures from S3
   - Load into pymatgen for analysis

2. **Property Analysis**
   - Density distribution
   - Space group statistics
   - Lattice parameter correlations
   - Chemical composition analysis

3. **Visualization**
   - Property histograms
   - Scatter plots (density vs volume)
   - Space group bar charts
   - Crystal system pie charts

4. **Export**
   - Save filtered datasets
   - Generate summary tables
   - Export figures

---

## What You'll Discover

### Materials Insights
- Distribution of material densities across different crystal systems
- Common space groups in your dataset
- Relationship between volume and number of atoms
- Lattice parameter patterns

### AWS Insights
- Serverless computing for scientific workflows
- NoSQL database design for materials data
- Cost-effective cloud processing
- Scale from 100 to 100,000 materials

### Research Insights
- Reproducibility: Same structures, same properties
- Collaboration: Share datasets and workflows
- Scale: Process large materials databases efficiently
- Persistence: Results stored permanently in cloud

---

## Next Steps

### Extend This Project
1. **More Properties**: Add band gap, magnetization, elastic constants
2. **Machine Learning**: Train models to predict properties
3. **Database Integration**: Connect to Materials Project API
4. **Batch Processing**: Process thousands of structures
5. **Notifications**: Email results using SNS

### Move to Tier 3 (Production)
- CloudFormation for infrastructure-as-code
- Production-grade Lambda with layers
- Advanced DynamoDB indexing
- Multi-region deployment
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
s3://materials-data-$(whoami)-$(date +%s)
```

**"Lambda timeout"**
```python
# Solution: Increase timeout in Lambda console
# Default: 3 seconds
# Recommended: 60 seconds per structure
# For complex structures: 120 seconds
```

**"Out of memory in Lambda"**
```python
# Solution: Increase Lambda memory allocation
# Default: 128 MB
# Recommended: 512 MB (includes more CPU)
# For large structures: 1024 MB
```

**"DynamoDB ProvisionedThroughputExceededException"**
```bash
# Solution: Use on-demand billing mode
aws dynamodb update-table \
  --table-name MaterialsProperties \
  --billing-mode PAY_PER_REQUEST
```

**"CIF parsing errors"**
```python
# Solution: Validate CIF files before upload
# Use pymatgen to check:
from pymatgen.core import Structure
try:
    struct = Structure.from_file("structure.cif")
except Exception as e:
    print(f"Invalid CIF: {e}")
```

See `setup_guide.md` for more solutions.

---

## Cleanup Guide

When finished, delete AWS resources to stop incurring charges:

```bash
# Delete S3 bucket and contents
aws s3 rm s3://materials-data-xxxx --recursive
aws s3 rb s3://materials-data-xxxx

# Delete DynamoDB table
aws dynamodb delete-table --table-name MaterialsProperties

# Delete Lambda function
aws lambda delete-function --function-name process-crystal-structure

# Delete IAM role
aws iam detach-role-policy \
  --role-name lambda-materials-processor \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
aws iam detach-role-policy \
  --role-name lambda-materials-processor \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
aws iam detach-role-policy \
  --role-name lambda-materials-processor \
  --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess
aws iam delete-role --role-name lambda-materials-processor

# Or use: python cleanup.py (automated)
```

See `cleanup_guide.md` for detailed instructions.

---

## Resources

### AWS Documentation
- [S3 Getting Started](https://docs.aws.amazon.com/s3/latest/userguide/getting-started.html)
- [Lambda Developer Guide](https://docs.aws.amazon.com/lambda/latest/dg/)
- [DynamoDB Developer Guide](https://docs.aws.amazon.com/dynamodb/latest/developerguide/)
- [IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)

### Materials Data
- [Materials Project](https://materialsproject.org/)
- [Crystallography Open Database](http://www.crystallography.net/)
- [ICSD (Inorganic Crystal Structure Database)](https://icsd.products.fiz-karlsruhe.de/)

### Python Libraries
- [boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- [pymatgen Documentation](https://pymatgen.org/)
- [pandas Documentation](https://pandas.pydata.org/docs/)

---

## Getting Help

### Project Issues
- GitHub Issues: https://github.com/research-jumpstart/research-jumpstart/issues
- Tag: `materials-science`, `tier-2`, `aws`

### AWS Support
- AWS Support Center: https://console.aws.amazon.com/support/
- AWS Forums: https://forums.aws.amazon.com/
- Stack Overflow: Tag `amazon-aws`, `boto3`

### Materials Science Help
- Pymatgen Discourse: https://matsci.org/c/pymatgen
- Materials Project Forum: https://discuss.materialsproject.org/

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
| **Data Scale** | Limited to ~1000 structures | Millions possible |
| **Database** | Local files | DynamoDB (queryable) |
| **Persistence** | Session-based | Permanent S3/DynamoDB storage |
| **Collaboration** | Limited | Full team access |

---

## What's Next?

After completing this project:

**Skill Building**
- AWS Lambda advanced features
- DynamoDB query optimization
- Materials property prediction with ML
- Cost optimization techniques

**Project Extensions**
- Real-time property calculations
- Automated materials screening pipelines
- Integration with Materials Project API
- Dashboard creation (QuickSight)

**Tier 3 Transition**
- Production infrastructure with CloudFormation
- High-availability architecture
- Advanced monitoring and alerting
- Multi-region deployment

---

## Citation

If you use this project in your research:

```bibtex
@software{research_jumpstart_materials_tier2,
  title = {Materials Property Prediction with S3 and Lambda: Tier 2},
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

**Last updated:** 2025-01-14 | Research Jumpstart v1.0.0
