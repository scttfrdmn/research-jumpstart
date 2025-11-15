# Transportation Flow Analysis with S3 and Lambda - Tier 2 Project

**Duration:** 2-4 hours | **Cost:** $7-13 | **Platform:** AWS + Local machine

Analyze urban traffic patterns using serverless AWS services. Upload traffic data to S3, process flow metrics with Lambda, store results in DynamoDB, and query with Athena—all without managing servers.

---

## What You'll Build

A cloud-native transportation analysis pipeline that demonstrates:

1. **Data Storage** - Upload traffic sensor data (CSV/JSON) to S3
2. **Serverless Processing** - Lambda functions to calculate traffic metrics in parallel
3. **Results Storage** - Store analysis results in DynamoDB for fast queries
4. **Data Querying** - Query traffic patterns with Athena or Jupyter notebook
5. **Visualization** - Generate traffic flow maps and congestion heatmaps

This bridges the gap between local analysis (Tier 1) and production infrastructure (Tier 3).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Your Local Machine                                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Python Scripts                                             │ │
│  │ • upload_to_s3.py - Upload traffic sensor data           │ │
│  │ • lambda_function.py - Calculate traffic metrics         │ │
│  │ • query_results.py - Analyze results                     │ │
│  │ • transportation_analysis.ipynb - Jupyter notebook       │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  AWS Services                                                    │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  │  S3 Bucket       │  │ Lambda Function  │  │  DynamoDB        │
│  │                  │  │                  │  │                  │
│  │ • Raw traffic    │→ │ Traffic flow     │→ │ Analysis results │
│  │   data (CSV)     │  │ analysis         │  │ • By location    │
│  │ • Results        │  │ • Speed metrics  │  │ • By time        │
│  │                  │  │ • LOS calc       │  │ • By segment     │
│  └──────────────────┘  │ • Congestion     │  └──────────────────┘
│                        └──────────────────┘           │
│                                                        │
│                        ┌──────────────────┐           │
│                        │ Athena (SQL)     │←──────────┘
│                        │                  │
│                        │ Query traffic    │
│                        │ patterns         │
│                        └──────────────────┘
│  ┌──────────────────────────────────────────────────────────────┐
│  │  IAM Role (Permissions)                                      │
│  │  • S3 read/write                                             │
│  │  • Lambda execution                                          │
│  │  • DynamoDB read/write                                       │
│  │  • CloudWatch logging                                        │
│  └──────────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Required Knowledge
- Basic Python (pandas, boto3)
- Understanding of traffic flow concepts
- AWS fundamentals (S3, Lambda, IAM)
- Basic transportation engineering (optional but helpful)

### Required Tools
- **Python 3.8+** with:
  - boto3 (AWS SDK)
  - pandas (data manipulation)
  - numpy (numerical operations)
  - matplotlib (visualization)
  - seaborn (statistical visualization)
  - networkx (network analysis)

- **AWS Account** with:
  - S3 bucket access
  - Lambda permissions
  - IAM role creation capability
  - DynamoDB access
  - (Optional) Athena query capability

### Installation

```bash
# Clone the project
git clone https://github.com/research-jumpstart/research-jumpstart.git
cd research-jumpstart/projects/urban-planning/transportation-optimization/tier-2

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
# - S3 bucket: transportation-data-{your-id}
# - IAM role: lambda-traffic-processor
# - Lambda function: analyze-traffic-flow
# - DynamoDB table: TrafficAnalysis
```

### Step 2: Upload Sample Data (3 minutes)
```bash
python scripts/upload_to_s3.py
```

### Step 3: Process Data (2 minutes)
```bash
# Lambda is automatically triggered by S3 upload
# Or invoke manually
python scripts/invoke_lambda.py
```

### Step 4: Query Results (2 minutes)
```bash
python scripts/query_results.py
```

### Step 5: Visualize (5 minutes)
Open `notebooks/transportation_analysis.ipynb` in Jupyter and run all cells.

---

## Detailed Workflow

### 1. Data Preparation (Setup)

**What's happening:**
- Generate or upload traffic sensor data
- Data includes: vehicle counts, speeds, timestamps, GPS coordinates
- Create S3 bucket with proper permissions

**Data Format:**
```csv
timestamp,segment_id,latitude,longitude,vehicle_count,avg_speed,occupancy,congestion_level
2025-01-15T08:00:00,SEG-001,37.7749,-122.4194,450,35.2,0.65,2
2025-01-15T08:05:00,SEG-001,37.7749,-122.4194,520,28.5,0.78,3
```

**Files involved:**
- `setup_guide.md` - Step-by-step AWS setup
- `scripts/upload_to_s3.py` - Upload automation

**Time:** 20-30 minutes (includes setup_guide steps)

### 2. Lambda Processing

**What's happening:**
- Lambda reads traffic data from S3
- Calculates transportation metrics:
  - Average speeds by segment
  - Volume/Capacity (V/C) ratios
  - Level of Service (LOS) ratings
  - Congestion detection and hotspots
  - Peak hour analysis
  - Travel time reliability
- Writes results to DynamoDB

**Lambda function**:
```python
# Event: S3 upload trigger
{
    'Records': [{
        'bucket': {'name': 'transportation-data-xxxx'},
        'object': {'key': 'raw/traffic_data_2025-01-15.csv'}
    }]
}

# Processing: Calculate traffic metrics
# Output: DynamoDB record + JSON summary to S3
```

**Transportation Metrics:**
- **Level of Service (LOS)**: A-F rating based on speed/capacity
- **V/C Ratio**: Volume to capacity ratio (congestion indicator)
- **Travel Time Index**: Actual travel time vs free-flow
- **Speed Performance Index**: Average speed vs posted speed limit
- **Congestion Duration**: Time periods with LOS D or worse

**Files involved:**
- `scripts/lambda_function.py` - Process function code
- `setup_guide.md` - Lambda deployment steps

**Time:** 5-10 minutes execution

### 3. Results Storage

**What's happening:**
- Processed results stored in DynamoDB for fast queries
- Summary statistics saved as JSON in S3
- Original CSV files kept for reference
- Organized folder structure for easy access

**S3 Structure:**
```
s3://transportation-data-{your-id}/
├── raw/                          # Original traffic data
│   ├── traffic_data_2025-01-15.csv
│   ├── traffic_data_2025-01-16.csv
│   └── ...
├── results/                       # Processed JSON summaries
│   ├── analysis_2025-01-15.json
│   ├── analysis_2025-01-16.json
│   └── ...
└── logs/                          # Lambda execution logs
    └── processing_log.txt
```

**DynamoDB Schema:**
```
TrafficAnalysis Table:
- Partition Key: segment_id (String)
- Sort Key: timestamp (Number)
- Attributes:
  - avg_speed (Number)
  - vehicle_count (Number)
  - vc_ratio (Number)
  - los (String)
  - congestion_level (Number)
  - travel_time_index (Number)
  - latitude (Number)
  - longitude (Number)
```

### 4. Results Analysis

**What's happening:**
- Query DynamoDB for specific segments or time ranges
- Download results to local machine
- Analyze with Jupyter notebook
- Create transportation visualizations
- (Optional) Query with Athena using SQL

**Files involved:**
- `notebooks/transportation_analysis.ipynb` - Analysis notebook
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
│   └── transportation_analysis.ipynb # Main analysis notebook
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

**Total estimated cost: $7-13 per run**

| Service | Usage | Cost |
|---------|-------|------|
| **S3 Storage** | 2GB × 7 days | $0.15 |
| **S3 Requests** | ~500 PUT/GET requests | $0.03 |
| **Lambda Executions** | 50 invocations × 1 min | $0.75 |
| **Lambda Compute** | 50 GB-seconds | $1.00 |
| **DynamoDB Storage** | ~100MB, 1 week | $0.13 |
| **DynamoDB Queries** | 1000 read/write units | $1.25 |
| **Data Transfer** | Upload + download (2GB) | $0.20 |
| **Athena Queries** | 10 queries × 5GB scanned | $2.50 |
| **Total** | | **$10.01** |

**Cost optimization tips:**
1. Delete S3 objects after analysis ($0.15 savings)
2. Use Lambda for < 1 minute processing
3. Run Athena queries in batch (not interactive)
4. Use DynamoDB on-demand pricing for variable workloads
5. Delete DynamoDB table when not in use

**Free Tier Usage:**
- **S3**: First 5GB free (12 months)
- **Lambda**: 1M invocations free (12 months)
- **DynamoDB**: 25GB storage free (always)
- **Athena**: First 1TB scanned free (per month)

---

## Key Learning Objectives

### AWS Services
- S3 bucket creation and management
- Lambda function deployment and triggers
- IAM role creation with least privilege
- DynamoDB NoSQL database design
- CloudWatch monitoring and logs
- (Optional) Athena for serverless SQL queries

### Cloud Concepts
- Object storage vs relational databases
- Serverless computing (no servers to manage)
- Event-driven architecture
- NoSQL data modeling
- Cost-conscious design

### Transportation Engineering Skills
- Traffic flow fundamentals
- Level of Service (LOS) calculation
- Volume/Capacity ratio analysis
- Congestion hotspot identification
- Peak hour pattern detection
- Travel time reliability metrics
- Network analysis and routing

---

## Time Estimates

**First Time Setup:**
- Read setup_guide.md: 10 minutes
- Create S3 bucket: 2 minutes
- Create IAM role: 5 minutes
- Deploy Lambda: 5 minutes
- Create DynamoDB table: 3 minutes
- Configure S3 upload: 3 minutes
- **Subtotal setup: 28 minutes**

**Data Processing:**
- Upload data: 5-10 minutes (2GB, depends on connection)
- Lambda processing: 5-10 minutes
- **Subtotal processing: 10-20 minutes**

**Analysis:**
- Query results: 5 minutes
- Jupyter analysis: 30-45 minutes
- Generate figures: 10-15 minutes
- **Subtotal analysis: 45-65 minutes**

**Total time: 1.5-2 hours** (including setup)

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
- Sample traffic data provided in notebooks
- Or bring your own traffic sensor data (CSV format)
- Or use open datasets from city transportation departments

---

## Running the Project

### Option 1: Automated (Recommended for First Time)
```bash
# Step 1: Setup AWS services (follow prompts)
# See setup_guide.md for manual steps

# Step 2: Generate and upload sample data
jupyter notebook notebooks/transportation_analysis.ipynb
# Run cells 1-3 to generate sample data

# Step 3: Upload to S3
python scripts/upload_to_s3.py

# Step 4: Lambda processes automatically on upload
# Check CloudWatch logs for progress

# Step 5: Query and analyze results
python scripts/query_results.py
# Continue with notebook for visualizations
```

### Option 2: Manual (Detailed Control)
```bash
# 1. Create S3 bucket manually
aws s3 mb s3://transportation-data-$(date +%s) --region us-east-1

# 2. Upload data
aws s3 cp sample_data/ s3://transportation-data-xxxx/raw/ --recursive

# 3. Deploy Lambda (see setup_guide.md)

# 4. Create DynamoDB table
aws dynamodb create-table --cli-input-json file://dynamodb-config.json

# 5. Run analysis notebook
jupyter notebook notebooks/transportation_analysis.ipynb
```

---

## Jupyter Notebook Workflow

The main analysis notebook includes:

1. **Sample Data Generation**
   - Create synthetic traffic network
   - Generate realistic traffic patterns
   - Include peak hours and congestion events

2. **Data Upload**
   - Upload to S3 bucket
   - Trigger Lambda processing
   - Monitor execution status

3. **Results Query**
   - Query DynamoDB for analysis results
   - Filter by segment, time, congestion level
   - Aggregate statistics

4. **Analysis**
   - Calculate system-wide metrics
   - Identify bottlenecks and hotspots
   - Peak hour analysis
   - Travel time reliability

5. **Visualization**
   - Traffic flow maps
   - Congestion heatmaps
   - Time-series speed plots
   - Level of Service distribution
   - Network visualization

6. **Export**
   - Save figures
   - Generate reports
   - Export data for further analysis

---

## What You'll Discover

### Transportation Insights
- Peak hour traffic patterns and duration
- Congestion hotspots and bottlenecks
- Level of Service distribution across network
- Travel time reliability metrics
- Speed performance by road segment
- Impact of congestion on network efficiency

### AWS Insights
- Serverless computing advantages for data processing
- Event-driven architecture patterns
- NoSQL database design for time-series data
- Cost-effective cloud analysis
- Scalability from city to regional networks

### Research Insights
- Reproducibility: Same code, same results
- Collaboration: Share workflows and results
- Scale: Process data from hundreds of sensors
- Real-time: Near real-time traffic analysis possible
- Persistence: Results saved permanently in cloud

---

## Traffic Metrics Explained

### Level of Service (LOS)
Highway Capacity Manual (HCM) ratings:
- **LOS A**: Free flow, speed ≥ 55 mph, V/C ≤ 0.35
- **LOS B**: Reasonably free flow, speed ≥ 50 mph, V/C ≤ 0.54
- **LOS C**: Stable flow, speed ≥ 45 mph, V/C ≤ 0.77
- **LOS D**: Approaching unstable, speed ≥ 40 mph, V/C ≤ 0.90
- **LOS E**: Unstable flow, speed ≥ 30 mph, V/C ≤ 1.00
- **LOS F**: Forced flow, speed < 30 mph, V/C > 1.00

### Volume/Capacity (V/C) Ratio
- Ratio of actual traffic volume to road capacity
- V/C < 0.8: Good flow
- 0.8 ≤ V/C < 1.0: Congested
- V/C ≥ 1.0: Over capacity

### Travel Time Index (TTI)
- Ratio of actual travel time to free-flow travel time
- TTI = 1.0: Free flow
- TTI = 1.3: 30% longer than free flow
- TTI > 2.0: Severe congestion

---

## Next Steps

### Extend This Project
1. **Real-time Processing**: Add streaming data ingestion
2. **Incident Detection**: Identify accidents from anomalies
3. **Predictive Analytics**: Forecast congestion using ML
4. **Network Optimization**: Route optimization algorithms
5. **Notifications**: Email/SMS alerts for severe congestion (SNS)
6. **Dashboard**: Real-time traffic dashboard (QuickSight)

### Move to Tier 3 (Production)
- CloudFormation for infrastructure-as-code
- Production-grade Lambda functions with layers
- Multi-region deployment
- Advanced monitoring and alerting
- Auto-scaling DynamoDB capacity
- Cost optimization techniques
- Integration with other transportation systems

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
s3://transportation-data-$(date +%s)-yourname
```

**"Lambda timeout"**
```python
# Solution: Increase timeout in Lambda console
# Default: 3 seconds
# Recommended: 300 seconds (5 minutes)
# For larger datasets: 600 seconds (10 minutes)
```

**"DynamoDB ProvisionedThroughputExceededException"**
```python
# Solution: Use on-demand billing mode instead of provisioned
# Or increase provisioned capacity
# On-demand automatically scales with traffic
```

**"Data too large for Lambda"**
```python
# Solution: Process data in batches
# Split large CSV files into smaller chunks
# Use Lambda with 1GB+ memory for larger datasets
```

**"Athena query returns no results"**
```sql
-- Solution: Check data format and partitioning
-- Ensure CSV files have headers
-- Verify S3 path in CREATE EXTERNAL TABLE statement
```

See `docs/troubleshooting.md` for more solutions.

---

## Cleanup Guide

When finished, delete AWS resources to stop incurring charges:

```bash
# Delete S3 bucket and contents
aws s3 rm s3://transportation-data-xxxx --recursive
aws s3 rb s3://transportation-data-xxxx

# Delete Lambda function
aws lambda delete-function --function-name analyze-traffic-flow

# Delete DynamoDB table
aws dynamodb delete-table --table-name TrafficAnalysis

# Delete IAM role
aws iam detach-role-policy --role-name lambda-traffic-processor \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
aws iam detach-role-policy --role-name lambda-traffic-processor \
  --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess
aws iam detach-role-policy --role-name lambda-traffic-processor \
  --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess
aws iam delete-role --role-name lambda-traffic-processor

# Or use: python cleanup.py (automated)
```

See `cleanup_guide.md` for detailed instructions.

---

## Resources

### AWS Documentation
- [S3 Getting Started](https://docs.aws.amazon.com/s3/latest/userguide/getting-started.html)
- [Lambda Developer Guide](https://docs.aws.amazon.com/lambda/latest/dg/)
- [IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
- [DynamoDB Developer Guide](https://docs.aws.amazon.com/dynamodb/latest/developerguide/)
- [Athena User Guide](https://docs.aws.amazon.com/athena/latest/ug/)

### Transportation Data
- [FHWA Traffic Data](https://www.fhwa.dot.gov/policyinformation/travel_monitoring/tvt.cfm)
- [Highway Capacity Manual](https://www.trb.org/Main/Blurbs/175169.aspx)
- [Open Traffic Data Portal](https://www.transportation.gov/data)

### Python Libraries
- [boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- [pandas Documentation](https://pandas.pydata.org/docs/)
- [NetworkX Documentation](https://networkx.org/documentation/stable/)

---

## Getting Help

### Project Issues
- GitHub Issues: https://github.com/research-jumpstart/research-jumpstart/issues
- Tag: `urban-planning`, `tier-2`, `aws`

### AWS Support
- AWS Support Center: https://console.aws.amazon.com/support/
- AWS Forums: https://forums.aws.amazon.com/
- Stack Overflow: Tag `amazon-aws`, `boto3`

### Transportation Engineering Help
- Transportation Research Board: https://www.trb.org/
- Institute of Transportation Engineers: https://www.ite.org/

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
| **Cost** | Free | $7-13 per run |
| **Storage** | 15GB persistent | Unlimited (pay per GB) |
| **Compute** | 4GB RAM, 12-hour sessions | Scalable (pay per second) |
| **Data Scale** | Limited to local processing | Process city-wide data |
| **Parallelization** | Single notebook | Multiple Lambda functions |
| **Persistence** | Session-based | Permanent S3/DynamoDB storage |
| **Collaboration** | Limited | Full team access |
| **Real-time** | No | Yes, with event triggers |

---

## What's Next?

After completing this project:

**Skill Building**
- AWS Lambda advanced features (layers, concurrency)
- Serverless architecture patterns
- NoSQL database optimization
- Cost optimization techniques
- Security best practices

**Project Extensions**
- Real-time traffic monitoring dashboard
- Predictive congestion forecasting with ML
- Integration with traffic signal systems
- Multi-modal transportation analysis
- Environmental impact assessment

**Tier 3 Transition**
- Production infrastructure with CloudFormation
- High-availability architecture
- Advanced monitoring and alerting
- Multi-region deployment
- Auto-scaling and cost optimization
- Integration with city traffic management systems

---

## Citation

If you use this project in your research:

```bibtex
@software{research_jumpstart_transportation_tier2,
  title = {Transportation Flow Analysis with S3 and Lambda: Tier 2},
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
