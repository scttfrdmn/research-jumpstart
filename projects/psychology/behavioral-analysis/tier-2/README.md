# Behavioral Data Analysis with S3 and Lambda - Tier 2 Project

**Duration:** 2-4 hours | **Cost:** $7-13 | **Platform:** AWS + Local machine

Analyze behavioral experiment data using serverless AWS services. Upload trial-level data to S3, process with Lambda for statistical analysis and computational modeling, store results in DynamoDB, and query with Athena—all without managing servers.

---

## What You'll Build

A cloud-native behavioral data analysis pipeline that demonstrates:

1. **Data Storage** - Upload behavioral datasets (reaction time, decision making, learning) to S3
2. **Serverless Processing** - Lambda functions to calculate statistics and fit computational models
3. **Results Storage** - Store analysis results in DynamoDB for fast queries
4. **Data Querying** - Query results with Athena or retrieve with boto3
5. **Group Analysis** - Aggregate and visualize results in Jupyter notebook

This bridges the gap between local analysis (Tier 1) and production infrastructure (Tier 3).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Your Local Machine                                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Python Scripts & Jupyter                                   │ │
│  │ • upload_to_s3.py - Upload behavioral data (CSV)         │ │
│  │ • lambda_function.py - Statistical analysis & modeling   │ │
│  │ • query_results.py - Retrieve and aggregate results      │ │
│  │ • behavioral_analysis.ipynb - Visualization & stats      │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  AWS Services                                                    │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  │  S3 Bucket       │  │ Lambda Function  │  │  DynamoDB        │
│  │                  │  │                  │  │                  │
│  │ • Raw data       │→ │ Statistical      │→ │ Store results:   │
│  │   (CSV files)    │  │ analysis:        │  │ • Mean RT        │
│  │ • Participant    │  │ • Mean RT        │  │ • Accuracy       │
│  │   metadata       │  │ • Accuracy       │  │ • d-prime        │
│  │                  │  │ • Signal detect. │  │ • Model params   │
│  │                  │  │ • Comp. models   │  │                  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘
│                                                        │
│  ┌──────────────────────────────────────────────────────────────┐
│  │  Athena (SQL Queries)                                        │
│  │  • Query across participants                                 │
│  │  • Group statistics                                          │
│  │  • Performance comparisons                                   │
│  └──────────────────────────────────────────────────────────────┘
│                                                                  │
│  ┌──────────────────────────────────────────────────────────────┐
│  │  IAM Role (Permissions)                                      │
│  │  • S3 read/write                                             │
│  │  • Lambda execution                                          │
│  │  • DynamoDB write                                            │
│  │  • CloudWatch logging                                        │
│  └──────────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Required Knowledge
- Basic Python (pandas, numpy, scipy)
- Understanding of behavioral experiments (RT, accuracy)
- AWS fundamentals (S3, Lambda, IAM)
- Basic statistics (t-tests, ANOVA)

### Required Tools
- **Python 3.8+** with:
  - boto3 (AWS SDK)
  - pandas (data manipulation)
  - numpy (numerical computing)
  - scipy (statistics)
  - statsmodels (statistical modeling)
  - matplotlib & seaborn (visualization)
  - jupyter (notebooks)

- **AWS Account** with:
  - S3 bucket access
  - Lambda permissions
  - DynamoDB access
  - IAM role creation capability
  - (Optional) Athena query capability

### Installation

```bash
# Clone the project
git clone https://github.com/research-jumpstart/research-jumpstart.git
cd research-jumpstart/projects/psychology/behavioral-analysis/tier-2

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
# - S3 bucket: behavioral-data-{your-id}
# - IAM role: lambda-behavioral-processor
# - Lambda function: analyze-behavioral-data
# - DynamoDB table: BehavioralAnalysis
```

### Step 2: Generate & Upload Sample Data (3 minutes)
```bash
python scripts/upload_to_s3.py --generate-sample
```

### Step 3: Process Data (2 minutes)
```bash
# Lambda is triggered automatically on upload
# Or invoke manually:
python scripts/invoke_lambda.py
```

### Step 4: Query Results (2 minutes)
```bash
python scripts/query_results.py
```

### Step 5: Visualize (5 minutes)
Open `notebooks/behavioral_analysis.ipynb` in Jupyter and run all cells.

---

## Detailed Workflow

### 1. Data Preparation (Setup)

**What's happening:**
- Generate sample behavioral data OR use your own CSV files
- Data format: Trial-level with columns: participant_id, trial, task_type, stimulus, response, rt, accuracy
- Create S3 bucket with proper permissions
- Upload participant data files

**Files involved:**
- `setup_guide.md` - Step-by-step AWS setup
- `scripts/upload_to_s3.py` - Upload automation with sample data generation

**Time:** 20-30 minutes (includes setup_guide steps)

### 2. Lambda Processing

**What's happening:**
- Lambda reads CSV files from S3
- Performs statistical analysis per participant:
  - Mean and median reaction time (RT)
  - Accuracy by condition
  - Signal detection theory metrics (d-prime, response bias)
  - Computational model fitting (drift diffusion, reinforcement learning)
- Writes results to DynamoDB

**Lambda function analyses**:
```python
# Input: CSV file with trial-level data
# participant_id,trial,task_type,stimulus,response,rt,accuracy
# sub001,1,stroop,congruent,left,450,1
# sub001,2,stroop,incongruent,right,620,1

# Processing:
# 1. Calculate mean RT by condition
# 2. Calculate accuracy and error rates
# 3. Fit signal detection theory model
# 4. Fit drift diffusion model (if applicable)
# 5. Fit reinforcement learning model (if learning task)

# Output: DynamoDB record with participant statistics
```

**Files involved:**
- `scripts/lambda_function.py` - Processing function code
- `setup_guide.md` - Lambda deployment steps

**Time:** 2-5 minutes execution per participant

### 3. Results Storage

**What's happening:**
- Processed results stored in DynamoDB
- Each participant gets one record with all statistics
- Original CSV files kept in S3 for reference
- Results can be queried by participant, task type, or performance metrics

**S3 Structure:**
```
s3://behavioral-data-{your-id}/
├── raw/                          # Original CSV files
│   ├── sub001_stroop.csv
│   ├── sub002_decision.csv
│   ├── sub003_learning.csv
│   └── ...
├── metadata/                      # Participant metadata
│   └── participants.csv
└── logs/                          # Lambda execution logs
    └── processing_log.txt
```

**DynamoDB Schema:**
```json
{
  "participant_id": "sub001",
  "task_type": "stroop",
  "timestamp": "2025-11-14T10:30:00Z",
  "mean_rt": 542.5,
  "median_rt": 520.0,
  "accuracy": 0.94,
  "dprime": 2.15,
  "response_bias": 0.02,
  "model_params": {
    "drift_rate": 0.35,
    "threshold": 1.2,
    "non_decision_time": 0.3
  }
}
```

### 4. Results Analysis

**What's happening:**
- Query DynamoDB for participant results
- Aggregate statistics across participants
- Perform group-level statistical tests
- Create publication-quality figures
- (Optional) Use Athena for SQL-based queries

**Files involved:**
- `notebooks/behavioral_analysis.ipynb` - Analysis notebook
- `scripts/query_results.py` - Download and analyze
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
│   └── behavioral_analysis.ipynb     # Main analysis notebook
│                                      # - Generate sample data
│                                      # - Upload to S3
│                                      # - Query results
│                                      # - Group analysis
│                                      # - Statistical tests
│                                      # - Visualization
│
└── scripts/
    ├── upload_to_s3.py               # Upload data to S3
    │                                 # - Generate sample data
    │                                 # - Upload CSV files
    │                                 # - Progress tracking
    ├── lambda_function.py            # Lambda processing function
    │                                 # - Statistical analysis
    │                                 # - Signal detection theory
    │                                 # - Computational modeling
    ├── query_results.py              # Download and analyze results
    │                                 # - Query DynamoDB
    │                                 # - Aggregate statistics
    │                                 # - Export to CSV
    └── __init__.py
```

---

## Cost Breakdown

**Total estimated cost: $7-13 per run**

| Service | Usage | Cost |
|---------|-------|------|
| **S3 Storage** | 100 MB × 7 days | $0.02 |
| **S3 Requests** | ~500 PUT/GET requests | $0.03 |
| **Lambda Executions** | 50 participants × 30 sec | $1.00 |
| **Lambda Compute** | 50 invocations × 512 MB | $1.50 |
| **DynamoDB Writes** | 50 records | $0.01 |
| **DynamoDB Storage** | 10 KB × 7 days | $0.001 |
| **Data Transfer** | Upload + download (100 MB) | $0.02 |
| **Athena Queries** | 10 queries × 100 MB scanned | $5.00 |
| **Total** | | **$7.58** |

**Cost optimization tips:**
1. Delete S3 objects after analysis ($0.02 savings)
2. Use Lambda with optimized memory (512 MB sufficient)
3. Run Athena queries in batch (not interactive exploration)
4. Use DynamoDB on-demand billing for unpredictable loads
5. Export DynamoDB to S3 for long-term storage

**Free Tier Usage:**
- **S3**: First 5GB free (12 months)
- **Lambda**: 1M invocations free (12 months)
- **Lambda**: 400,000 GB-seconds compute free (12 months)
- **DynamoDB**: 25 GB storage free (always free)
- **Athena**: First 1TB scanned free (per month)

**Note:** With free tier, actual cost may be **$0-3** for first year users.

---

## Key Learning Objectives

### AWS Services
- ✅ S3 bucket creation and management
- ✅ Lambda function deployment and triggers
- ✅ IAM role creation with least privilege
- ✅ DynamoDB table design and querying
- ✅ CloudWatch monitoring and logs
- ✅ (Optional) Athena for serverless SQL queries

### Cloud Concepts
- ✅ Object storage vs database storage
- ✅ Serverless computing (no servers to manage)
- ✅ Event-driven architecture
- ✅ NoSQL database design
- ✅ Cost-conscious design
- ✅ Scalable data processing

### Behavioral Analysis Skills
- ✅ Trial-level data processing
- ✅ Reaction time analysis (mean, median, distribution)
- ✅ Accuracy and error rate calculation
- ✅ Signal detection theory (d-prime, response bias)
- ✅ Computational modeling basics (drift diffusion, RL)
- ✅ Group-level statistical testing

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
- Generate sample data: 2 minutes
- Upload data: 3-5 minutes
- Lambda processing: 2-5 minutes (50 participants)
- **Subtotal processing: 7-12 minutes**

**Analysis:**
- Query results: 2 minutes
- Jupyter analysis: 30-45 minutes
- Generate figures: 10-15 minutes
- Statistical tests: 10-15 minutes
- **Subtotal analysis: 52-77 minutes**

**Total time: 1.5-2 hours** (including setup)

---

## AWS Account Setup

### Create AWS Account
1. Go to https://aws.amazon.com/
2. Click "Create an AWS Account"
3. Follow registration steps
4. (Optional) Activate free tier: https://console.aws.amazon.com/billing/

### Create IAM User for Programmatic Access
```bash
# 1. Go to IAM Console: https://console.aws.amazon.com/iam/
# 2. Create new user with programmatic access
# 3. Attach policies: AmazonS3FullAccess, AWSLambdaFullAccess,
#    AmazonDynamoDBFullAccess, IAMFullAccess
# 4. Save Access Key ID and Secret Access Key
```

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
# Step 1: Generate and upload sample data
python scripts/upload_to_s3.py --generate-sample --bucket behavioral-data-12345

# Step 2: Deploy Lambda (follow setup_guide.md)
# Manual: Deploy scripts/lambda_function.py to Lambda console

# Step 3: Process data (Lambda triggers automatically on S3 upload)
# Or invoke manually:
aws lambda invoke \
  --function-name analyze-behavioral-data \
  --payload '{"bucket": "behavioral-data-12345", "key": "raw/sub001_stroop.csv"}' \
  response.json

# Step 4: Query results
python scripts/query_results.py --table BehavioralAnalysis

# Step 5: Analyze in notebook
jupyter notebook notebooks/behavioral_analysis.ipynb
```

### Option 2: Use Your Own Data
```bash
# 1. Prepare CSV files with columns:
#    participant_id, trial, task_type, stimulus, response, rt, accuracy

# 2. Upload to S3
python scripts/upload_to_s3.py \
  --data-dir ./my_data \
  --bucket behavioral-data-12345

# 3. Lambda processes automatically

# 4. Query and analyze
python scripts/query_results.py --table BehavioralAnalysis
jupyter notebook notebooks/behavioral_analysis.ipynb
```

---

## Jupyter Notebook Workflow

The main analysis notebook includes:

### 1. Data Generation
- Generate sample Stroop task data
- Generate decision making task data
- Generate reinforcement learning task data
- Upload to S3

### 2. Lambda Invocation
- Trigger processing for each participant
- Monitor execution status
- Check CloudWatch logs

### 3. Results Retrieval
- Query DynamoDB for all participants
- Load results into pandas DataFrame
- Export to CSV for backup

### 4. Statistical Analysis
- Descriptive statistics (mean, SD, range)
- Within-subject comparisons (paired t-tests)
- Between-subject comparisons (independent t-tests)
- ANOVA for multi-condition designs
- Non-parametric tests (Wilcoxon, Mann-Whitney)

### 5. Visualization
- Reaction time distributions
- Accuracy by condition
- Individual participant performance
- Group averages with error bars
- Model parameter distributions
- Correlations between measures

### 6. Computational Modeling
- Drift diffusion model parameters
- Response bias visualization
- Learning rate estimates (RL tasks)
- Model comparison (AIC, BIC)

---

## What You'll Discover

### Behavioral Insights
- How reaction time varies by task difficulty
- Accuracy-speed tradeoffs in decision making
- Individual differences in cognitive strategies
- Learning curves in reinforcement learning tasks
- Signal detection sensitivity and response bias

### AWS Insights
- Serverless computing advantages for batch processing
- Cost-effective analysis of multi-participant studies
- Parallel processing of independent participants
- NoSQL database design for experimental data
- Scalability from 10 to 10,000 participants

### Research Insights
- Reproducibility: Same code, same results across runs
- Collaboration: Share datasets and pipelines with team
- Scale: Process 1000 participants as easily as 10
- Persistence: Results saved permanently in cloud
- Integration: Combine with other AWS services (SageMaker, QuickSight)

---

## Behavioral Tasks Supported

### 1. Stroop Task
- **Design:** Congruent vs incongruent color-word trials
- **Measures:** RT difference, accuracy, interference effect
- **Analysis:** Repeated-measures t-test, effect size
- **Computational model:** Drift diffusion model

### 2. Decision Making
- **Design:** Two-alternative forced choice
- **Measures:** Choice probability, RT by difficulty
- **Analysis:** Signal detection theory (d-prime, criterion)
- **Computational model:** Drift diffusion model

### 3. Reinforcement Learning
- **Design:** Probabilistic reward feedback
- **Measures:** Learning curve, choice optimality
- **Analysis:** Learning rate, exploration-exploitation
- **Computational model:** Q-learning, actor-critic

### 4. Go/No-Go
- **Design:** Response inhibition task
- **Measures:** False alarm rate, miss rate, RT
- **Analysis:** Signal detection theory
- **Computational model:** Threshold model

### 5. N-Back
- **Design:** Working memory load manipulation
- **Measures:** Accuracy by load, RT by load
- **Analysis:** Load effects, capacity limits
- **Computational model:** Capacity model

---

## Next Steps

### Extend This Project
1. **More Tasks**: Add flanker task, Simon task, IAT
2. **Advanced Models**: Hierarchical drift diffusion, LBA model
3. **Demographics**: Correlate performance with age, education
4. **Longitudinal**: Track participants over time
5. **Real-time Dashboard**: Use QuickSight for live monitoring
6. **Notifications**: Email results using SNS

### Move to Tier 3 (Production)
- CloudFormation for infrastructure-as-code
- Production-grade Lambda functions with layers
- Multi-region deployment for international studies
- Advanced monitoring and alerting
- Cost optimization techniques
- Integration with REDCap or Qualtrics

### Research Applications
- Multi-site cognitive studies
- Large-scale online experiments
- Meta-analysis across studies
- Open science data sharing
- Clinical trial data management

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
# S3 buckets must be globally unique across all AWS accounts
s3://behavioral-data-$(date +%s)-yourname
```

**"Lambda timeout"**
```python
# Solution: Increase timeout in Lambda console
# Default: 3 seconds
# Recommended: 60 seconds (sufficient for most analyses)
# For complex modeling: 300 seconds (5 minutes)
```

**"Out of memory in Lambda"**
```python
# Solution: Increase Lambda memory allocation
# Default: 128 MB
# Recommended: 512 MB (sufficient for scipy/numpy)
# For large datasets: 1024 MB
# Cost increases proportionally with memory
```

**"ModuleNotFoundError: No module named 'scipy'"**
```python
# Solution: Lambda needs scipy as a layer
# 1. Create Lambda layer with scipy
# 2. Or use AWS Lambda Power Tools
# See setup_guide.md for detailed instructions
```

**"DynamoDB write throttling"**
```python
# Solution: Use on-demand billing mode
# Or increase provisioned write capacity
# On-demand is better for variable workloads
```

**"Athena query fails"**
```sql
-- Solution: Ensure S3 bucket has proper structure
-- Athena requires partitioned data or specific format
-- See setup_guide.md for Athena configuration
```

See `setup_guide.md` and `cleanup_guide.md` for more solutions.

---

## Cleanup Guide

When finished, delete AWS resources to stop incurring charges:

```bash
# Delete S3 bucket and contents
aws s3 rm s3://behavioral-data-xxxx --recursive
aws s3 rb s3://behavioral-data-xxxx

# Delete Lambda function
aws lambda delete-function --function-name analyze-behavioral-data

# Delete DynamoDB table
aws dynamodb delete-table --table-name BehavioralAnalysis

# Delete IAM role
aws iam delete-role-policy --role-name lambda-behavioral-processor \
  --policy-name lambda-policy
aws iam delete-role --role-name lambda-behavioral-processor

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

### Behavioral Analysis
- [Signal Detection Theory Tutorial](https://psycnet.apa.org/record/2002-01842-010)
- [Drift Diffusion Model Overview](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3218849/)
- [Reinforcement Learning in Psychology](https://www.nature.com/articles/nn1560)

### Python Libraries
- [boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- [pandas Documentation](https://pandas.pydata.org/docs/)
- [scipy.stats Documentation](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [statsmodels Documentation](https://www.statsmodels.org/)

---

## Getting Help

### Project Issues
- GitHub Issues: https://github.com/research-jumpstart/research-jumpstart/issues
- Tag: `psychology`, `tier-2`, `aws`, `behavioral-analysis`

### AWS Support
- AWS Support Center: https://console.aws.amazon.com/support/
- AWS Forums: https://forums.aws.amazon.com/
- Stack Overflow: Tag `amazon-aws`, `boto3`, `aws-lambda`

### Behavioral Analysis Help
- PsychoPy Discourse: https://discourse.psychopy.org/
- Cognitive Science Stack Exchange: https://cogsci.stackexchange.com/

---

## Cost Tracking

### Monitor Your Spending

```bash
# Check current AWS charges
aws ce get-cost-and-usage \
  --time-period Start=2025-11-01,End=2025-11-30 \
  --granularity MONTHLY \
  --metrics "BlendedCost"

# Set up billing alerts in AWS console:
# https://docs.aws.amazon.com/billing/latest/userguide/budgets-create.html
```

Recommended alerts:
- $5 threshold (notification)
- $10 threshold (warning)
- $20 threshold (critical)

---

## Key Differences from Tier 1

| Aspect | Tier 1 (Studio Lab) | Tier 2 (AWS) |
|--------|-------------------|------------|
| **Platform** | Free SageMaker Studio Lab | AWS account required |
| **Cost** | Free | $7-13 per run |
| **Storage** | 15GB persistent | Unlimited (pay per GB) |
| **Compute** | 4GB RAM, 12-hour sessions | Scalable (pay per second) |
| **Data Scale** | Limited to 100 participants | Thousands possible |
| **Parallelization** | Single notebook | Multiple Lambda functions |
| **Persistence** | Session-based | Permanent storage |
| **Collaboration** | Limited | Full team access via IAM |
| **Database** | Local CSV/pickle | DynamoDB + Athena |

---

## What's Next?

After completing this project:

**Skill Building**
- AWS Lambda layers for scientific libraries
- DynamoDB advanced queries and indexing
- Athena performance optimization
- CloudWatch custom metrics
- Step Functions for complex workflows

**Project Extensions**
- Real-time experiment monitoring
- Automated quality control checks
- Integration with online experiment platforms (jsPsych, Gorilla)
- Dashboard creation (QuickSight, Grafana)
- Machine learning on behavioral data (SageMaker)

**Tier 3 Transition**
- Production infrastructure with CloudFormation
- Multi-environment deployment (dev, staging, prod)
- Advanced monitoring and alerting
- CI/CD pipeline for analysis updates
- HIPAA-compliant storage for clinical data

---

## Citation

If you use this project in your research:

```bibtex
@software{research_jumpstart_behavioral_tier2,
  title = {Behavioral Data Analysis with S3 and Lambda: Tier 2},
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
