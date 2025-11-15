# Molecular Property Analysis with AWS - Tier 2 Project

**Duration:** 2-4 hours | **Cost:** $9-14 | **Platform:** AWS + Local machine

Analyze molecular properties at scale using serverless AWS services. Upload molecular structures to S3, calculate drug-like properties with Lambda, store results in DynamoDB, and query with Athena—all without managing servers.

---

## What You'll Build

A cloud-native molecular analysis pipeline that demonstrates:

1. **Data Storage** - Upload molecular structures (SMILES, SDF, MOL2) to S3
2. **Serverless Processing** - Lambda functions calculate molecular properties
3. **NoSQL Database** - Store property data in DynamoDB for fast queries
4. **SQL Queries** - Query results with Athena (optional)
5. **Local Analysis** - Download and analyze results in Jupyter

This bridges the gap between Studio Lab analysis (Tier 1) and production infrastructure (Tier 3).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Your Local Machine                                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Python Scripts & Jupyter                                    │ │
│  │ • upload_to_s3.py - Upload molecular structures            │ │
│  │ • lambda_function.py - Property calculation code           │ │
│  │ • query_results.py - Analyze DynamoDB results              │ │
│  │ • molecular_analysis.ipynb - Full workflow notebook        │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  AWS Services                                                    │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  │  S3 Bucket       │  │ Lambda Function  │  │  DynamoDB        │
│  │                  │  │                  │  │                  │
│  │ • Molecular      │→ │ Calculate        │→ │ Store molecular  │
│  │   structures     │  │ properties:      │  │ properties       │
│  │   (SMILES/SDF)   │  │ • MW, LogP       │  │                  │
│  │                  │  │ • TPSA, HBD/HBA  │  │ Query by:        │
│  │                  │  │ • Lipinski       │  │ • Property range │
│  │                  │  │ • Drug-likeness  │  │ • Compound class │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘
│  ┌──────────────────────────────────────────────────────────────┐
│  │  Athena (Optional SQL Queries)                               │
│  │  • Query DynamoDB export or S3 results                       │
│  │  • Filter drug-like compounds                                │
│  │  • Complex property-based searches                           │
│  └──────────────────────────────────────────────────────────────┘
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
- Understanding of molecular structures (SMILES notation)
- Chemistry basics (molecular weight, solubility, drug-likeness)
- AWS fundamentals (S3, Lambda, IAM, DynamoDB)

### Required Tools
- **Python 3.8+** with:
  - boto3 (AWS SDK)
  - pandas (data manipulation)
  - rdkit-pypi (molecular calculations) - optional
  - matplotlib, seaborn (visualization)
  - jupyter (interactive analysis)

- **AWS Account** with:
  - S3 bucket access
  - Lambda permissions
  - DynamoDB table creation
  - IAM role creation capability
  - (Optional) Athena query capability

### Installation

```bash
# Clone the project
git clone https://github.com/research-jumpstart/research-jumpstart.git
cd research-jumpstart/projects/chemistry/molecular-analysis/tier-2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start (20 minutes)

### Step 1: Set Up AWS (15 minutes)
```bash
# Follow setup_guide.md for detailed instructions
# Creates:
# - S3 bucket: molecular-data-{your-id}
# - IAM role: lambda-molecular-analyzer
# - Lambda function: analyze-molecule
# - DynamoDB table: MolecularProperties
```

### Step 2: Upload Sample Molecules (2 minutes)
```bash
python scripts/upload_to_s3.py --bucket molecular-data-{your-id}
```

### Step 3: Process Molecules (2 minutes)
```bash
# Lambda auto-triggers on upload, or invoke manually
python scripts/test_lambda.py
```

### Step 4: Query Results (1 minute)
```bash
python scripts/query_results.py --table MolecularProperties
```

### Step 5: Visualize in Jupyter (10 minutes)
```bash
jupyter notebook notebooks/molecular_analysis.ipynb
```

---

## Detailed Workflow

### 1. Data Preparation (Setup)

**What's happening:**
- Create S3 bucket for molecular structures
- Organize molecules by compound class (drugs, natural products, etc.)
- Set up DynamoDB table schema for property storage

**Files involved:**
- `setup_guide.md` - Step-by-step AWS setup
- `scripts/upload_to_s3.py` - Upload automation

**Time:** 15-20 minutes (includes setup_guide steps)

### 2. Lambda Processing

**What's happening:**
- Lambda reads SMILES strings from S3
- Calculates molecular properties:
  - **Molecular Weight (MW)**: Size constraint
  - **LogP**: Lipophilicity (octanol-water partition)
  - **TPSA**: Topological Polar Surface Area (absorption)
  - **HBD/HBA**: Hydrogen bond donors/acceptors
  - **Rotatable Bonds**: Molecular flexibility
  - **Aromatic Rings**: Structure features
  - **Lipinski's Rule of Five**: Drug-likeness compliance
- Validates SMILES syntax
- Writes results to DynamoDB

**Lambda function flow**:
```python
# Event: S3 upload trigger
{
    'Records': [{
        'bucket': {'name': 'molecular-data-xxxx'},
        'object': {'key': 'molecules/drugs.smi'}
    }]
}

# Processing: Calculate properties for each molecule
# Output: DynamoDB items with molecular properties
{
    'molecule_id': 'CHEM000001',
    'smiles': 'CC(C)Cc1ccc(cc1)C(C)C(O)=O',
    'molecular_weight': 206.28,
    'logp': 3.97,
    'tpsa': 37.3,
    'hbd': 1,
    'hba': 2,
    'lipinski_compliant': True
}
```

**Files involved:**
- `scripts/lambda_function.py` - Property calculation code
- `setup_guide.md` - Lambda deployment steps

**Time:** 5-10 minutes execution (depends on molecule count)

### 3. Results Storage

**What's happening:**
- Molecular properties stored in DynamoDB
- NoSQL schema allows flexible queries
- Indexed by molecule_id and property values
- S3 retains original molecular structures

**DynamoDB Schema:**
```
Table: MolecularProperties
Partition Key: molecule_id (String)
Sort Key: compound_class (String)

Attributes:
- smiles (String)
- name (String)
- molecular_weight (Number)
- logp (Number)
- tpsa (Number)
- hbd (Number)
- hba (Number)
- rotatable_bonds (Number)
- aromatic_rings (Number)
- lipinski_compliant (Boolean)
- timestamp (String)
```

**S3 Structure:**
```
s3://molecular-data-{your-id}/
├── molecules/                    # Original structures
│   ├── drugs/
│   │   ├── aspirin.smi
│   │   ├── ibuprofen.smi
│   │   └── ...
│   ├── natural_products/
│   │   ├── caffeine.smi
│   │   ├── morphine.smi
│   │   └── ...
│   └── screening_library/
│       └── compounds_001.smi
└── logs/                         # Lambda execution logs
    └── processing_log.txt
```

### 4. Results Analysis

**What's happening:**
- Query DynamoDB for molecules by property ranges
- Filter drug-like compounds (Lipinski compliant)
- Analyze property distributions
- Identify promising candidates
- Create publication-quality visualizations

**Query examples:**
```python
# Find drug-like molecules with MW < 500
response = dynamodb.query(
    TableName='MolecularProperties',
    FilterExpression='molecular_weight < :mw AND lipinski_compliant = :true',
    ExpressionAttributeValues={':mw': 500, ':true': True}
)

# Find soluble molecules (LogP < 3)
response = dynamodb.scan(
    TableName='MolecularProperties',
    FilterExpression='logp < :logp',
    ExpressionAttributeValues={':logp': 3.0}
)
```

**Files involved:**
- `notebooks/molecular_analysis.ipynb` - Analysis notebook
- `scripts/query_results.py` - Query and download results
- (Optional) Athena queries for SQL-based analysis

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
│   └── molecular_analysis.ipynb      # Main analysis notebook
│
├── scripts/
│   ├── upload_to_s3.py               # Upload molecular structures
│   ├── lambda_function.py            # Lambda property calculation
│   ├── query_results.py              # Query DynamoDB results
│   ├── test_lambda.py                # Test Lambda locally
│   └── __init__.py
│
├── sample_data/
│   ├── drugs.smi                     # Sample drug molecules
│   ├── natural_products.smi          # Sample natural products
│   └── README.md                     # Sample data documentation
│
└── docs/
    ├── chemistry_concepts.md         # Molecular property explanations
    ├── lipinski_rules.md             # Drug-likeness criteria
    └── troubleshooting.md            # Common issues
```

---

## Cost Breakdown

**Total estimated cost: $9-14 per run**

| Service | Usage | Cost |
|---------|-------|------|
| **S3 Storage** | 1GB × 7 days | $0.16 |
| **S3 Requests** | ~5,000 PUT/GET requests | $0.03 |
| **Lambda Executions** | 10,000 invocations × 5 sec | $2.00 |
| **Lambda Compute** | 10,000 × 256MB × 5 sec | $4.17 |
| **DynamoDB Storage** | 10,000 items × 1KB each | $2.50 |
| **DynamoDB Requests** | 10,000 writes + 5,000 reads | $1.50 |
| **Data Transfer** | Upload + download (2GB) | $0.18 |
| **Athena Queries** | 10 queries × 100MB scanned | $0.01 |
| **CloudWatch Logs** | 50MB logs | Free |
| **Total** | | **$10.55** |

**Cost optimization tips:**
1. Use on-demand DynamoDB pricing (no provisioned capacity)
2. Delete S3 objects after analysis ($0.16 savings)
3. Set Lambda timeout to 30 seconds max
4. Use DynamoDB batch writes (reduce write costs)
5. Export DynamoDB to S3 for Athena (cheaper than direct queries)

**Free Tier Usage:**
- **S3**: First 5GB free (12 months)
- **Lambda**: 1M invocations + 400,000 GB-seconds free (monthly)
- **DynamoDB**: 25GB storage + 25 read/write capacity units free (monthly)
- **Athena**: First 1TB scanned free (per month)

**Note:** With Free Tier, your first run may cost $0-3!

---

## Key Learning Objectives

### AWS Services
- S3 bucket creation and object lifecycle management
- Lambda function deployment and event triggers
- DynamoDB table design and NoSQL queries
- IAM role creation with least privilege
- CloudWatch monitoring and logs
- (Optional) Athena for serverless SQL queries

### Cloud Concepts
- Object storage vs database storage
- Serverless computing (no servers to manage)
- Event-driven architecture (S3 → Lambda → DynamoDB)
- NoSQL database design patterns
- Cost-conscious design decisions

### Chemistry Skills
- SMILES notation and molecular representation
- Molecular descriptors and properties
- Lipinski's Rule of Five for drug-likeness
- ADME properties (Absorption, Distribution, Metabolism, Excretion)
- Structure-property relationships
- Virtual screening fundamentals

---

## Time Estimates

**First Time Setup:**
- Read setup_guide.md: 10 minutes
- Create S3 bucket: 2 minutes
- Create IAM role: 5 minutes
- Deploy Lambda: 10 minutes
- Create DynamoDB table: 3 minutes
- **Subtotal setup: 30 minutes**

**Data Processing:**
- Upload molecular structures: 5 minutes
- Lambda processing (10K molecules): 10-15 minutes
- **Subtotal processing: 15-20 minutes**

**Analysis:**
- Query DynamoDB: 5 minutes
- Jupyter analysis: 30-45 minutes
- Generate visualizations: 10-15 minutes
- **Subtotal analysis: 45-65 minutes**

**Total time: 1.5-2 hours** (including setup)

---

## Prerequisites

### AWS Account Setup
1. Create AWS account: https://aws.amazon.com/
2. (Optional) Activate free tier: https://console.aws.amazon.com/billing/
3. Create IAM user for programmatic access
4. Set billing alerts ($10 and $20 thresholds)

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
- Sample molecules provided in `sample_data/`
- Or bring your own SMILES files
- Or download from PubChem/ChEMBL (see Tier 1)

---

## Running the Project

### Option 1: Automated (Recommended)
```bash
# Step 1: Setup AWS services (follow prompts)
# See setup_guide.md for manual steps

# Step 2: Upload molecular data
python scripts/upload_to_s3.py \
  --bucket molecular-data-{your-id} \
  --data-dir sample_data/

# Step 3: Lambda auto-processes on S3 upload
# Or invoke manually:
aws lambda invoke \
  --function-name analyze-molecule \
  --payload '{"bucket":"molecular-data-xxxx","key":"molecules/drugs.smi"}' \
  response.json

# Step 4: Query results
python scripts/query_results.py \
  --table MolecularProperties \
  --output results.csv

# Step 5: Analyze in Jupyter
jupyter notebook notebooks/molecular_analysis.ipynb
```

### Option 2: Manual (Detailed Control)
```bash
# 1. Create S3 bucket manually
aws s3 mb s3://molecular-data-$(date +%s) --region us-east-1

# 2. Upload molecules
aws s3 cp sample_data/ s3://molecular-data-xxxx/molecules/ --recursive

# 3. Deploy Lambda (see setup_guide.md for full steps)
cd scripts
zip lambda_deployment.zip lambda_function.py
aws lambda create-function \
  --function-name analyze-molecule \
  --runtime python3.11 \
  --role arn:aws:iam::ACCOUNT:role/lambda-molecular-analyzer \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://lambda_deployment.zip

# 4. Query DynamoDB
aws dynamodb scan --table-name MolecularProperties

# 5. Run analysis notebook
jupyter notebook notebooks/molecular_analysis.ipynb
```

---

## Jupyter Notebook Workflow

The main analysis notebook includes:

1. **Data Loading**
   - Query DynamoDB for molecular properties
   - Download results to pandas DataFrame
   - Load original structures from S3

2. **Property Analysis**
   - Distribution of molecular weights
   - LogP vs TPSA scatter plots
   - Lipinski compliance statistics
   - Identify outliers and interesting molecules

3. **Drug-Likeness Filtering**
   - Apply Lipinski's Rule of Five
   - Find drug-like candidates
   - Compare different compound classes

4. **Visualization**
   - Property distributions (histograms)
   - Chemical space plots (PCA/UMAP)
   - Structure-property relationships
   - Export results for publication

5. **Virtual Screening**
   - Filter by property ranges
   - Rank molecules by drug-likeness score
   - Generate hit list for further analysis

---

## What You'll Discover

### Chemistry Insights
- Distribution of molecular properties in different compound classes
- How drug molecules differ from natural products
- Structure-property relationships
- Identification of promising drug candidates
- ADME property correlations

### AWS Insights
- Serverless computing advantages for scientific workflows
- NoSQL database design for molecular data
- Cost-effective cloud analysis strategies
- Event-driven architecture patterns
- Scalability from 10K to 10M molecules

### Research Insights
- Reproducibility: Same code, same results
- Collaboration: Share workflows and datasets
- Scale: Process 100K molecules as easily as 100
- Persistence: Results stored permanently in cloud
- Integration: Easy to connect with other AWS services

---

## Next Steps

### Extend This Project
1. **More Properties**: Add MACCS keys, Morgan fingerprints, 3D descriptors
2. **Larger Datasets**: Process 100K-1M molecules from PubChem
3. **Machine Learning**: Predict bioactivity using property data
4. **Similarity Search**: Find structurally similar molecules
5. **Batch Processing**: Use AWS Batch for large-scale screening
6. **Notifications**: Email results using SNS
7. **API Gateway**: Create REST API for molecular property lookups

### Move to Tier 3 (Production)
- CloudFormation for infrastructure-as-code
- Production-grade Lambda layers with RDKit
- Multi-region deployment for global access
- Advanced monitoring and alerting
- Auto-scaling DynamoDB tables
- Cost optimization with Reserved Capacity

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
BUCKET_NAME="molecular-data-$(whoami)-$(date +%s)"
aws s3 mb "s3://$BUCKET_NAME" --region us-east-1
```

**"Lambda timeout"**
```python
# Solution: Increase timeout in Lambda console
# Default: 3 seconds
# Recommended: 30 seconds (sufficient for most molecules)
# For large SDF files: 60 seconds

# Or update via CLI:
aws lambda update-function-configuration \
  --function-name analyze-molecule \
  --timeout 30
```

**"RDKit not available in Lambda"**
```python
# Solution: Use Lambda Layers or pure Python calculations
# The lambda_function.py includes fallback calculations
# Or deploy RDKit as a Lambda Layer (see setup_guide.md)

# Simple approach: Use pure Python SMILES parsing
# Included in lambda_function.py for basic properties
```

**"DynamoDB ConditionalCheckFailedException"**
```python
# Solution: Molecule already exists in database
# Either skip duplicates or update with overwrite flag
table.put_item(
    Item=item,
    ConditionExpression='attribute_not_exists(molecule_id)'
)
```

**"High AWS costs"**
```bash
# Solution: Check for runaway Lambda invocations
aws logs tail /aws/lambda/analyze-molecule --since 1h

# Check DynamoDB capacity mode
aws dynamodb describe-table --table-name MolecularProperties

# Ensure it's on-demand (not provisioned)
# Delete old data
aws dynamodb scan --table-name MolecularProperties \
  --filter-expression "age > :days" \
  --expression-attribute-values '{":days":{"N":"7"}}'
```

See `docs/troubleshooting.md` for more solutions.

---

## Cleanup Guide

When finished, delete AWS resources to stop incurring charges:

```bash
# Quick cleanup
BUCKET_NAME="molecular-data-xxxx"  # Replace with your bucket

# Delete S3 bucket and contents
aws s3 rm s3://$BUCKET_NAME --recursive
aws s3 rb s3://$BUCKET_NAME

# Delete DynamoDB table
aws dynamodb delete-table --table-name MolecularProperties

# Delete Lambda function
aws lambda delete-function --function-name analyze-molecule

# Delete IAM role
aws iam delete-role-policy --role-name lambda-molecular-analyzer \
  --policy-name lambda-policy
aws iam delete-role --role-name lambda-molecular-analyzer

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
- [Athena User Guide](https://docs.aws.amazon.com/athena/latest/ug/)

### Chemistry Resources
- [RDKit Documentation](https://www.rdkit.org/docs/)
- [Lipinski's Rule of Five](https://en.wikipedia.org/wiki/Lipinski%27s_rule_of_five)
- [SMILES Tutorial](https://www.daylight.com/dayhtml/doc/theory/theory.smiles.html)
- [PubChem on AWS](https://registry.opendata.aws/pubchem/)
- [ChEMBL Database](https://www.ebi.ac.uk/chembl/)

### Python Libraries
- [boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- [pandas Documentation](https://pandas.pydata.org/docs/)
- [RDKit Python API](https://www.rdkit.org/docs/GettingStartedInPython.html)

---

## Getting Help

### Project Issues
- GitHub Issues: https://github.com/research-jumpstart/research-jumpstart/issues
- Tag: `chemistry`, `tier-2`, `aws`

### AWS Support
- AWS Support Center: https://console.aws.amazon.com/support/
- AWS Forums: https://forums.aws.amazon.com/
- Stack Overflow: Tag `amazon-aws`, `boto3`, `aws-lambda`

### Chemistry Help
- RDKit Discussions: https://github.com/rdkit/rdkit/discussions
- Chemistry Stack Exchange: https://chemistry.stackexchange.com/

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
- $5 threshold (warning)
- $10 threshold (warning)
- $20 threshold (critical)

---

## Key Differences from Tier 1

| Aspect | Tier 1 (Studio Lab) | Tier 2 (AWS) |
|--------|-------------------|------------|
| **Platform** | Free SageMaker Studio Lab | AWS account required |
| **Cost** | Free | $9-14 per run |
| **Storage** | 15GB persistent | Unlimited S3 (pay per GB) |
| **Compute** | 4GB RAM, 12-hour sessions | Scalable Lambda (pay per invocation) |
| **Data Scale** | 10K-100K molecules | 1M+ molecules possible |
| **Processing** | Sequential in notebook | Parallel Lambda invocations |
| **Persistence** | Session-based | Permanent S3 + DynamoDB storage |
| **Collaboration** | Limited | Full team access via IAM |
| **Database** | Local files/pickle | DynamoDB NoSQL database |
| **Queries** | pandas filtering | SQL with Athena + DynamoDB queries |

---

## What's Next?

After completing this project:

**Skill Building**
- AWS Lambda advanced features (layers, destinations)
- DynamoDB advanced queries (GSI, LSI)
- Serverless architecture patterns
- Cost optimization techniques
- Security best practices

**Project Extensions**
- Real-time molecular property API
- Automated screening pipelines
- Integration with machine learning models
- Similarity search with vector databases
- Dashboard creation (QuickSight)

**Tier 3 Transition**
- Production infrastructure with CloudFormation
- High-availability architecture
- Advanced monitoring and alerting
- Multi-region deployment
- CI/CD pipelines

---

## Citation

If you use this project in your research:

```bibtex
@software{research_jumpstart_chemistry_tier2,
  title = {Molecular Property Analysis with AWS: Tier 2},
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
