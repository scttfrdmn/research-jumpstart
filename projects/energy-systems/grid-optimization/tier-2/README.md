# Smart Grid Optimization with S3 and Lambda - Tier 2 Project

**Duration:** 2-4 hours | **Cost:** $8-14 | **Platform:** AWS + Local machine

Analyze smart grid data using serverless AWS services. Upload grid load profiles, generation data, and voltage measurements to S3, process with Lambda for optimization analysis, store results in DynamoDB, and query with Athena—all without managing servers.

---

## What You'll Build

A cloud-native smart grid analytics pipeline that demonstrates:

1. **Data Storage** - Upload grid sensor data (~5-10GB) to S3
2. **Serverless Processing** - Lambda functions to analyze load patterns and grid stability
3. **Results Storage** - Store optimization metrics in DynamoDB
4. **Data Querying** - Query results with Athena and visualize in notebooks
5. **Alerting** - SNS notifications for grid anomalies

This bridges the gap between local analysis (Tier 1) and production infrastructure (Tier 3).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Your Local Machine                                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Python Scripts & Jupyter Notebook                          │ │
│  │ • upload_to_s3.py - Upload grid data (CSV/JSON)          │ │
│  │ • lambda_function.py - Grid optimization logic           │ │
│  │ • query_results.py - Analyze results                     │ │
│  │ • grid_analysis.ipynb - Jupyter notebook                 │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  AWS Services                                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  S3 Bucket   │  │ Lambda       │  │  DynamoDB    │           │
│  │              │  │ Function     │  │  Table       │           │
│  │ • Raw data   │→ │              │→ │              │           │
│  │   (load,     │  │ • Load       │  │ GridAnalysis │           │
│  │    generation│  │   forecasting│  │ • Timestamps │           │
│  │    voltage)  │  │ • Renewable  │  │ • Metrics    │           │
│  │ • Results    │  │   integration│  │ • Locations  │           │
│  └──────────────┘  │ • Grid       │  └──────────────┘           │
│                    │   stability  │                              │
│  ┌──────────────┐  │ • Peak       │  ┌──────────────┐           │
│  │  Athena      │  │   demand     │  │  SNS Topic   │           │
│  │  (SQL)       │  └──────────────┘  │              │           │
│  │              │                     │ Grid Anomaly │           │
│  │ Query grid   │                     │ Alerts       │           │
│  │ analytics    │                     │              │           │
│  └──────────────┘                     └──────────────┘           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  IAM Role (Permissions)                                  │   │
│  │  • S3 read/write                                         │   │
│  │  • DynamoDB read/write                                   │   │
│  │  • Lambda execution                                      │   │
│  │  • SNS publish                                           │   │
│  │  • CloudWatch logging                                    │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Required Knowledge
- Basic Python (pandas, boto3)
- Understanding of power systems (load, generation, voltage)
- AWS fundamentals (S3, Lambda, IAM, DynamoDB)

### Required Tools
- **Python 3.8+** with:
  - boto3 (AWS SDK)
  - pandas (data manipulation)
  - numpy (numerical computing)
  - matplotlib, seaborn (visualization)
  - scipy (optimization)

- **AWS Account** with:
  - S3 bucket access
  - Lambda permissions
  - DynamoDB permissions
  - SNS permissions
  - IAM role creation capability
  - (Optional) Athena query capability

### Installation

```bash
# Clone the project
git clone https://github.com/research-jumpstart/research-jumpstart.git
cd research-jumpstart/projects/energy-systems/grid-optimization/tier-2

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
# - S3 bucket: energy-grid-{your-id}
# - IAM role: lambda-grid-optimizer
# - Lambda function: optimize-energy-grid
# - DynamoDB table: GridAnalysis
# - SNS topic: grid-anomaly-alerts
```

### Step 2: Upload Sample Data (2 minutes)
```bash
python scripts/upload_to_s3.py --bucket energy-grid-{your-id}
```

### Step 3: Process Data (2 minutes)
```bash
# Lambda will process automatically or invoke manually
python scripts/lambda_function.py --test
```

### Step 4: Query Results (1 minute)
```bash
python scripts/query_results.py --bucket energy-grid-{your-id}
```

### Step 5: Visualize (5 minutes)
Open `notebooks/grid_analysis.ipynb` in Jupyter and run all cells.

---

## Detailed Workflow

### 1. Data Preparation (Setup)

**What's happening:**
- Generate or upload smart grid time-series data
- Load profiles (MW), generation (MW), voltage (kV), frequency (Hz)
- Renewable generation (solar, wind) with timestamps
- Store data in S3 with proper structure

**Files involved:**
- `setup_guide.md` - Step-by-step AWS setup
- `scripts/upload_to_s3.py` - Upload automation

**Data format:**
```csv
timestamp,location,load_mw,generation_mw,voltage_kv,frequency_hz,solar_mw,wind_mw
2025-01-14T00:00:00,substation_001,125.5,130.2,13.8,60.01,15.3,8.7
2025-01-14T00:15:00,substation_001,128.3,132.1,13.79,60.00,12.1,9.2
```

**Time:** 15-20 minutes (includes setup_guide steps)

### 2. Lambda Processing

**What's happening:**
- Lambda reads grid data from S3 (CSV/JSON)
- Performs load forecasting and demand analysis
- Calculates renewable energy integration metrics
- Analyzes grid stability (voltage, frequency)
- Identifies peak demand periods
- Calculates energy efficiency metrics
- Sends SNS alerts for grid instability
- Stores results in DynamoDB

**Lambda function**:
```python
# Event: S3 upload trigger
{
    'Records': [{
        'bucket': {'name': 'energy-grid-xxxx'},
        'object': {'key': 'raw/grid_data_2025-01-14.csv'}
    }]
}

# Processing:
# - Load forecasting (next 24 hours)
# - Renewable penetration analysis
# - Grid stability assessment
# - Peak demand identification

# Output:
# - DynamoDB: GridAnalysis table
# - S3: results/grid_data_2025-01-14_analysis.json
# - SNS: Alert if voltage < 13.5 kV or frequency deviation > 0.05 Hz
```

**Files involved:**
- `scripts/lambda_function.py` - Grid optimization logic
- `setup_guide.md` - Lambda deployment steps

**Time:** 5-10 minutes execution

### 3. Results Storage

**What's happening:**
- DynamoDB stores time-series metrics for fast queries
- S3 stores detailed analysis results (JSON)
- Organized folder structure for easy access
- CloudWatch logs for debugging

**S3 Structure:**
```
s3://energy-grid-{your-id}/
├── raw/                           # Original grid data (CSV/JSON)
│   ├── grid_data_2025-01-14.csv
│   ├── grid_data_2025-01-15.csv
│   └── ...
├── results/                        # Processed analysis (JSON)
│   ├── grid_data_2025-01-14_analysis.json
│   ├── grid_data_2025-01-15_analysis.json
│   └── ...
└── logs/                           # Processing logs
    └── processing_log.txt
```

**DynamoDB Schema:**
```json
{
  "timestamp": "2025-01-14T12:00:00",
  "location": "substation_001",
  "load_avg_mw": 145.2,
  "load_peak_mw": 178.5,
  "renewable_penetration": 0.23,
  "voltage_avg_kv": 13.78,
  "frequency_avg_hz": 59.99,
  "stability_score": 0.92,
  "efficiency_score": 0.88,
  "alert_status": "normal"
}
```

### 4. Results Analysis

**What's happening:**
- Query DynamoDB for specific time periods or locations
- Download detailed results from S3
- Analyze with Jupyter notebook
- Create publication-quality visualizations
- (Optional) Query with Athena using SQL

**Files involved:**
- `notebooks/grid_analysis.ipynb` - Analysis notebook
- `scripts/query_results.py` - Query and download
- (Optional) Athena queries for SQL access

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
│   └── grid_analysis.ipynb           # Main analysis notebook
│
└── scripts/
    ├── upload_to_s3.py               # Upload grid data to S3
    ├── lambda_function.py            # Lambda processing function
    ├── query_results.py              # Query DynamoDB and S3
    └── __init__.py
```

---

## Cost Breakdown

**Total estimated cost: $8-14 per run**

| Service | Usage | Cost |
|---------|-------|------|
| **S3 Storage** | 5GB × 7 days | $0.35 |
| **S3 Requests** | ~2000 PUT/GET requests | $0.10 |
| **Lambda Executions** | 200 invocations × 45 sec | $2.50 |
| **Lambda Compute** | 200 invocations × 512 MB | $3.00 |
| **DynamoDB Storage** | 1GB × 7 days | $0.25 |
| **DynamoDB Read/Write** | 1000 read + 1000 write units | $1.25 |
| **SNS Notifications** | 50 email alerts | $0.05 |
| **Athena Queries** | 10 queries × 5GB scanned | $2.50 |
| **Data Transfer** | Upload + download (5GB) | $0.50 |
| **Total** | | **$10.50** |

**Cost optimization tips:**
1. Delete S3 objects after analysis ($0.35 savings)
2. Use Lambda for < 1 minute processing
3. Run Athena queries in batch (not interactive)
4. Use DynamoDB on-demand pricing (no provisioned capacity)
5. Limit SNS notifications to critical alerts

**Free Tier Usage:**
- **S3**: First 5GB free (12 months)
- **Lambda**: 1M invocations free (12 months)
- **DynamoDB**: 25GB storage free (always free)
- **SNS**: 1000 emails free (always free)
- **Athena**: First 1TB scanned free (per month)

---

## Key Learning Objectives

### AWS Services
- ✅ S3 bucket creation and management
- ✅ Lambda function deployment and triggers
- ✅ DynamoDB table design and queries
- ✅ SNS topic creation and subscriptions
- ✅ IAM role creation with least privilege
- ✅ CloudWatch monitoring and logs
- ✅ (Optional) Athena for serverless SQL queries

### Cloud Concepts
- ✅ Object storage vs NoSQL databases
- ✅ Serverless computing (no servers to manage)
- ✅ Event-driven architecture
- ✅ Real-time alerting systems
- ✅ Cost-conscious design

### Energy Systems Skills
- ✅ Load forecasting techniques
- ✅ Renewable energy integration analysis
- ✅ Grid stability assessment
- ✅ Peak demand identification
- ✅ Power quality metrics
- ✅ Energy efficiency optimization

---

## Time Estimates

**First Time Setup:**
- Read setup_guide.md: 10 minutes
- Create S3 bucket: 2 minutes
- Create DynamoDB table: 3 minutes
- Create SNS topic: 2 minutes
- Create IAM role: 5 minutes
- Deploy Lambda: 5 minutes
- Configure triggers: 3 minutes
- **Subtotal setup: 30 minutes**

**Data Processing:**
- Generate/upload data: 10-15 minutes
- Lambda processing: 10-15 minutes
- **Subtotal processing: 20-30 minutes**

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

---

## Running the Project

### Option 1: Automated (Recommended for First Time)
```bash
# Step 1: Upload data (generates sample data if needed)
python scripts/upload_to_s3.py --bucket energy-grid-{your-id}

# Step 2: Lambda processes automatically via S3 trigger
# Monitor progress:
aws logs tail /aws/lambda/optimize-energy-grid --follow

# Step 3: Query results
python scripts/query_results.py --bucket energy-grid-{your-id}

# Step 4: Analyze in Jupyter
jupyter notebook notebooks/grid_analysis.ipynb
```

### Option 2: Manual (Detailed Control)
```bash
# 1. Create S3 bucket manually
aws s3 mb s3://energy-grid-$(date +%s) --region us-east-1

# 2. Upload data
aws s3 cp sample_data/ s3://energy-grid-xxxx/raw/ --recursive

# 3. Invoke Lambda manually
aws lambda invoke \
  --function-name optimize-energy-grid \
  --payload '{"Records":[{"s3":{"bucket":{"name":"energy-grid-xxxx"},"object":{"key":"raw/grid_data.csv"}}}]}' \
  response.json

# 4. Query DynamoDB
aws dynamodb scan --table-name GridAnalysis

# 5. Run analysis notebook
jupyter notebook notebooks/grid_analysis.ipynb
```

---

## Jupyter Notebook Workflow

The main analysis notebook includes:

1. **Data Generation**
   - Create synthetic smart grid data
   - Load profiles with daily/seasonal patterns
   - Renewable generation with variability
   - Grid measurements (voltage, frequency)

2. **Upload to S3**
   - Batch upload using boto3
   - Verify uploads

3. **Trigger Processing**
   - Manual Lambda invocation
   - Monitor CloudWatch logs

4. **Query Results**
   - Query DynamoDB by time range
   - Query by location
   - Query by alert status
   - Download detailed results from S3

5. **Visualization**
   - Load curves (daily, weekly)
   - Renewable penetration over time
   - Voltage and frequency stability plots
   - Peak demand heatmaps
   - Grid efficiency metrics

6. **Analysis**
   - Load forecasting accuracy
   - Renewable integration impact
   - Grid stability correlation analysis
   - Peak demand patterns
   - Cost optimization scenarios

7. **Export**
   - Save figures (PNG, PDF)
   - Generate summary reports
   - Export data for further analysis

---

## What You'll Discover

### Grid Insights
- Daily and seasonal load patterns
- Peak demand periods and magnitudes
- Renewable energy integration challenges
- Grid stability correlations
- Voltage regulation effectiveness
- Frequency stability metrics

### Optimization Insights
- Optimal renewable penetration levels
- Load shifting opportunities
- Peak demand reduction strategies
- Grid efficiency improvements
- Energy storage requirements

### AWS Insights
- Serverless computing for real-time data
- DynamoDB for time-series queries
- S3 for cost-effective storage
- SNS for operational alerting
- Scalability from kW to GW scale

### Research Insights
- Reproducibility: Same code, same results
- Collaboration: Share workflows and results
- Scale: Process TB of grid data
- Real-time: Sub-second query responses

---

## Next Steps

### Extend This Project
1. **More Locations**: Add multiple substations/regions
2. **More Variables**: Include power factor, harmonics, transformer temps
3. **Machine Learning**: Train LSTM for load forecasting
4. **Optimization**: Implement economic dispatch algorithms
5. **Real-time**: Connect to live grid sensor APIs
6. **Battery Storage**: Model energy storage systems
7. **Demand Response**: Simulate DR programs

### Move to Tier 3 (Production)
- CloudFormation for infrastructure-as-code
- Production-grade Lambda with error handling
- Multi-region deployment for reliability
- Advanced monitoring and alerting
- Auto-scaling DynamoDB
- Cost optimization techniques
- Integration with SCADA systems

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
s3://energy-grid-$(date +%s)-yourname
```

**"Lambda timeout"**
```python
# Solution: Increase timeout in Lambda console
# Default: 3 seconds
# Recommended: 300 seconds (5 minutes) for large files
# For very large datasets: 600 seconds (10 minutes)
```

**"DynamoDB throughput exceeded"**
```bash
# Solution: Use on-demand pricing or increase provisioned capacity
aws dynamodb update-table \
  --table-name GridAnalysis \
  --billing-mode PAY_PER_REQUEST
```

**"SNS email not received"**
```bash
# Solution: Check spam folder and confirm subscription
# Resend confirmation:
aws sns subscribe \
  --topic-arn arn:aws:sns:us-east-1:ACCOUNT:grid-anomaly-alerts \
  --protocol email \
  --notification-endpoint your-email@example.com
```

**"Out of memory in Lambda"**
```python
# Solution: Increase Lambda memory allocation
# Default: 128 MB
# Recommended: 512 MB or 1024 MB for grid data processing
# Memory also increases CPU allocation
```

---

## Cleanup Guide

When finished, delete AWS resources to stop incurring charges:

```bash
# Delete S3 bucket and contents
aws s3 rm s3://energy-grid-xxxx --recursive
aws s3 rb s3://energy-grid-xxxx

# Delete Lambda function
aws lambda delete-function --function-name optimize-energy-grid

# Delete DynamoDB table
aws dynamodb delete-table --table-name GridAnalysis

# Delete SNS topic
aws sns delete-topic --topic-arn arn:aws:sns:us-east-1:ACCOUNT:grid-anomaly-alerts

# Delete IAM role
aws iam delete-role-policy --role-name lambda-grid-optimizer --policy-name lambda-policy
aws iam delete-role --role-name lambda-grid-optimizer

# Or use: python cleanup.py (automated)
```

See `cleanup_guide.md` for detailed instructions.

---

## Resources

### AWS Documentation
- [S3 Getting Started](https://docs.aws.amazon.com/s3/latest/userguide/getting-started.html)
- [Lambda Developer Guide](https://docs.aws.amazon.com/lambda/latest/dg/)
- [DynamoDB Developer Guide](https://docs.aws.amazon.com/dynamodb/latest/developerguide/)
- [SNS Developer Guide](https://docs.aws.amazon.com/sns/latest/dg/)
- [IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
- [Athena User Guide](https://docs.aws.amazon.com/athena/latest/ug/)

### Power Systems
- [NREL Grid Datasets](https://www.nrel.gov/grid/data-tools.html)
- [IEEE Power Systems Test Cases](https://icseg.iti.illinois.edu/ieee-power-systems-test-cases/)
- [Smart Grid Standards](https://www.nist.gov/engineering-laboratory/smart-grid)

### Python Libraries
- [boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- [pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Documentation](https://numpy.org/doc/)
- [SciPy Optimization](https://docs.scipy.org/doc/scipy/reference/optimize.html)

---

## Getting Help

### Project Issues
- GitHub Issues: https://github.com/research-jumpstart/research-jumpstart/issues
- Tag: `energy-systems`, `tier-2`, `aws`

### AWS Support
- AWS Support Center: https://console.aws.amazon.com/support/
- AWS Forums: https://forums.aws.amazon.com/
- Stack Overflow: Tag `amazon-aws`, `boto3`

### Energy Systems Help
- IEEE Power & Energy Society: https://www.ieee-pes.org/
- NREL Support: https://www.nrel.gov/contact/

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
| **Data Scale** | Limited to 15GB | Terabytes possible |
| **Queries** | Pandas only | SQL with Athena |
| **Persistence** | Session-based | Permanent S3/DynamoDB |
| **Alerting** | None | SNS real-time alerts |
| **Collaboration** | Limited | Full team access |

---

## What's Next?

After completing this project:

**Skill Building**
- AWS Lambda advanced features (layers, containers)
- DynamoDB advanced queries (GSI, LSI)
- Serverless architecture patterns
- Cost optimization techniques
- Security best practices

**Project Extensions**
- Real-time grid monitoring dashboard
- Automated forecasting pipelines
- Integration with SCADA/EMS systems
- Machine learning for anomaly detection
- Optimization algorithms (OPF, ED, UC)

**Tier 3 Transition**
- Production infrastructure with CloudFormation
- High-availability architecture
- Advanced monitoring and alerting
- Multi-region deployment
- Auto-scaling and load balancing

---

## Citation

If you use this project in your research:

```bibtex
@software{research_jumpstart_energy_tier2,
  title = {Smart Grid Optimization with S3 and Lambda: Tier 2},
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
