# Social Media Sentiment Analysis with AWS - Tier 2 Project

**Duration:** 2-4 hours | **Cost:** $6-10 | **Platform:** AWS + Local machine

Analyze social media sentiment at scale using serverless AWS services. Upload tweets and posts to S3, process with sentiment analysis using Lambda and AWS Comprehend, store results in DynamoDB, and query them with Athena—all without managing servers.

---

## What You'll Build

A cloud-native social media analysis pipeline that demonstrates:

1. **Data Storage** - Upload social media data (CSV/JSON) to S3
2. **Serverless Processing** - Lambda functions for sentiment analysis using AWS Comprehend
3. **NoSQL Storage** - Store analyzed posts with sentiment scores in DynamoDB
4. **Data Querying** - Query results with Athena or Python scripts
5. **Visualization** - Analyze sentiment trends and network patterns

This bridges the gap between local analysis (Tier 1) and production infrastructure (Tier 3).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Your Local Machine                                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Python Scripts & Jupyter Notebook                          │ │
│  │ • upload_to_s3.py - Upload social media data              │ │
│  │ • lambda_function.py - Sentiment analysis code            │ │
│  │ • query_results.py - Analyze results                      │ │
│  │ • social_analysis.ipynb - Main analysis notebook          │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  AWS Services                                                    │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  │  S3 Bucket       │  │ Lambda Function  │  │  AWS Comprehend  │
│  │                  │  │                  │  │                  │
│  │ • Raw data       │→ │ Process posts    │→ │ Sentiment        │
│  │ • Sample posts   │  │ Extract entities │  │ analysis API     │
│  │ • Results        │  │ Analyze hashtags │  │                  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘
│                                 │
│                                 ▼
│  ┌──────────────────┐  ┌──────────────────┐
│  │  DynamoDB Table  │  │  Athena (SQL)    │
│  │                  │  │                  │
│  │ • Post metadata  │  │ Query aggregated │
│  │ • Sentiment      │  │ sentiment data   │
│  │ • Entities       │  │ (optional)       │
│  │ • Hashtags       │  │                  │
│  └──────────────────┘  └──────────────────┘
│  ┌──────────────────────────────────────────────────────────────┐
│  │  IAM Role (Permissions)                                      │
│  │  • S3 read/write                                             │
│  │  • DynamoDB read/write                                       │
│  │  • Comprehend DetectSentiment                                │
│  │  • CloudWatch logging                                        │
│  └──────────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Required Knowledge
- Basic Python (pandas, boto3)
- Understanding of social media data structures
- AWS fundamentals (S3, Lambda, DynamoDB, IAM)
- Basic sentiment analysis concepts

### Required Tools
- **Python 3.8+** with:
  - boto3 (AWS SDK)
  - pandas (data manipulation)
  - matplotlib & seaborn (visualization)
  - networkx (network analysis)
  - jupyter (interactive analysis)

- **AWS Account** with:
  - S3 bucket access
  - Lambda permissions
  - DynamoDB permissions
  - Comprehend API access
  - IAM role creation capability
  - (Optional) Athena query capability

### Installation

```bash
# Clone the project
git clone https://github.com/research-jumpstart/research-jumpstart.git
cd research-jumpstart/projects/social-science/social-media-analysis/tier-2

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
# - S3 bucket: social-media-data-{your-id}
# - IAM role: lambda-social-analysis
# - Lambda function: analyze-sentiment
# - DynamoDB table: SocialMediaPosts
```

### Step 2: Upload Sample Data (2 minutes)
```bash
python scripts/upload_to_s3.py --s3-bucket social-media-data-{your-id}
```

### Step 3: Process Data (2 minutes)
Lambda will automatically process posts as they're uploaded to S3, or invoke manually:
```bash
# Lambda processes automatically via S3 trigger
# Or test manually via AWS Console
```

### Step 4: Query Results (2 minutes)
```bash
python scripts/query_results.py --table-name SocialMediaPosts
```

### Step 5: Visualize (5 minutes)
Open `notebooks/social_analysis.ipynb` in Jupyter and run all cells.

---

## Detailed Workflow

### 1. Data Preparation (Setup)

**What's happening:**
- Download or prepare sample social media data (tweets, posts)
- Format as CSV or JSON with fields: post_id, text, timestamp, user_id
- Create S3 bucket with proper permissions
- Create DynamoDB table for storing results

**Files involved:**
- `setup_guide.md` - Step-by-step AWS setup
- `scripts/upload_to_s3.py` - Upload automation
- Sample data provided or user-supplied

**Time:** 15-20 minutes (includes setup_guide steps)

### 2. Lambda Processing

**What's happening:**
- Lambda triggered by S3 upload events
- Reads post text from S3
- Calls AWS Comprehend DetectSentiment API
- Extracts sentiment scores (positive, negative, neutral, mixed)
- Extracts entities (people, places, organizations)
- Analyzes hashtags and mentions
- Writes results to DynamoDB

**Lambda function workflow:**
```python
# Event: S3 upload trigger
{
    'Records': [{
        'bucket': {'name': 'social-media-data-xxxx'},
        'object': {'key': 'raw/posts_batch_001.json'}
    }]
}

# Processing:
# 1. Download JSON batch from S3
# 2. For each post:
#    - Call Comprehend for sentiment
#    - Extract hashtags with regex
#    - Extract mentions
#    - Calculate engagement metrics
# 3. Write to DynamoDB: post_id, sentiment, scores, entities
# 4. Log processing metrics
```

**Files involved:**
- `scripts/lambda_function.py` - Processing function code
- `setup_guide.md` - Lambda deployment steps

**Time:** 5-15 minutes execution (depends on data size)

### 3. Results Storage

**What's happening:**
- Processed results stored in DynamoDB with schema:
  - post_id (primary key)
  - timestamp (sort key)
  - text (post content)
  - sentiment (POSITIVE/NEGATIVE/NEUTRAL/MIXED)
  - sentiment_scores (detailed scores)
  - entities (extracted entities)
  - hashtags (array of hashtags)
  - mentions (array of @mentions)

- Optional: Export to S3 for Athena queries

**DynamoDB Table Structure:**
```
Table: SocialMediaPosts
Primary Key: post_id (String)
Sort Key: timestamp (Number)

Attributes:
- text: String
- sentiment: String
- positive_score: Number
- negative_score: Number
- neutral_score: Number
- mixed_score: Number
- entities: List
- hashtags: List
- mentions: List
- user_id: String
```

### 4. Results Analysis

**What's happening:**
- Query DynamoDB for posts by sentiment, time range, hashtags
- Aggregate sentiment statistics
- Analyze trending topics
- Build network graphs of user interactions
- Create visualizations

**Files involved:**
- `notebooks/social_analysis.ipynb` - Main analysis notebook
- `scripts/query_results.py` - Query and download results
- (Optional) Athena for SQL queries

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
│   └── social_analysis.ipynb         # Main analysis notebook
│
└── scripts/
    ├── upload_to_s3.py               # Upload data to S3
    ├── lambda_function.py            # Lambda sentiment analysis
    └── query_results.py              # Query and analyze results
```

---

## Cost Breakdown

**Total estimated cost: $6-10 per run**

| Service | Usage | Cost |
|---------|-------|------|
| **S3 Storage** | 500MB × 7 days | $0.01 |
| **S3 Requests** | ~500 PUT/GET requests | $0.01 |
| **Lambda Executions** | 1000 invocations × 10 sec | $0.20 |
| **Lambda Compute** | 1000 invocations × 128MB | $0.20 |
| **AWS Comprehend** | 1000 posts × sentiment analysis | $5.00 |
| **DynamoDB** | 1000 writes + storage | $1.25 |
| **DynamoDB Reads** | ~500 queries | $0.25 |
| **Data Transfer** | Download results (100MB) | $0.01 |
| **Athena Queries** | 5 queries × 100MB scanned | $0.01 |
| **Total** | | **$6.94** |

**Cost optimization tips:**
1. Process posts in batches to reduce Lambda invocations
2. Use Comprehend batch API for 25+ posts ($0.0001/unit vs $0.0005/unit)
3. Delete S3 objects after processing
4. Use DynamoDB on-demand pricing for variable workloads
5. Set Lambda timeout to 30 seconds (not 5 minutes)

**Free Tier Usage:**
- **S3**: First 5GB free (12 months)
- **Lambda**: 1M invocations free per month
- **DynamoDB**: 25GB storage + 25 WCU/RCU free
- **Comprehend**: First 50k units free (12 months)

---

## Key Learning Objectives

### AWS Services
- S3 bucket creation and event notifications
- Lambda function deployment and S3 triggers
- IAM role creation with least privilege
- DynamoDB table design for NoSQL
- AWS Comprehend sentiment analysis API
- CloudWatch monitoring and logs
- (Optional) Athena for serverless SQL queries

### Cloud Concepts
- Event-driven architecture with S3 triggers
- Serverless computing (no servers to manage)
- NoSQL database design patterns
- API-based machine learning services
- Cost-conscious design
- Batch processing optimization

### Social Science Skills
- Sentiment analysis fundamentals
- Social network analysis basics
- Text entity extraction
- Hashtag and mention analysis
- Temporal sentiment trends
- User interaction patterns

---

## Time Estimates

**First Time Setup:**
- Read setup_guide.md: 10 minutes
- Create S3 bucket: 2 minutes
- Create DynamoDB table: 3 minutes
- Create IAM role: 5 minutes
- Deploy Lambda: 5 minutes
- Configure S3 trigger: 3 minutes
- **Subtotal setup: 28 minutes**

**Data Processing:**
- Prepare/upload data: 10-15 minutes (500MB sample data)
- Lambda processing: 5-10 minutes (1000 posts)
- **Subtotal processing: 15-25 minutes**

**Analysis:**
- Query results: 5 minutes
- Jupyter analysis: 30-45 minutes
- Generate visualizations: 10-15 minutes
- **Subtotal analysis: 45-65 minutes**

**Total time: 1.5-2 hours** (including setup)

---

## AWS Account Setup

### Create AWS Account
1. Go to: https://aws.amazon.com/
2. Click "Create an AWS Account"
3. Follow signup process
4. Add payment method (required, but free tier available)

### Configure AWS CLI
```bash
# Install AWS CLI (if needed)
pip install awscli

# Configure AWS credentials
aws configure
# Enter: Access Key ID
# Enter: Secret Access Key
# Enter: Region (us-east-1 recommended)
# Enter: Output format (json)
```

### Verify Setup
```bash
# Test AWS credentials
aws sts get-caller-identity

# Should show your account ID and user ARN
```

---

## Running the Project

### Option 1: Automated Workflow (Recommended)
```bash
# Step 1: Setup AWS services (follow setup_guide.md)
# Manual: Create S3 bucket, DynamoDB table, Lambda function, IAM role

# Step 2: Upload sample data
python scripts/upload_to_s3.py \
    --input-file sample_data/tweets.json \
    --s3-bucket social-media-data-{your-id}

# Step 3: Lambda processes automatically via S3 trigger

# Step 4: Query and analyze
python scripts/query_results.py \
    --table-name SocialMediaPosts \
    --output-dir ./results/

# Step 5: Run Jupyter analysis
jupyter notebook notebooks/social_analysis.ipynb
```

### Option 2: Manual Step-by-Step
```bash
# 1. Create S3 bucket
aws s3 mb s3://social-media-data-$(date +%s) --region us-east-1

# 2. Upload data
aws s3 cp sample_data/tweets.json s3://social-media-data-xxxx/raw/

# 3. Deploy Lambda (see setup_guide.md)
# 4. Configure S3 trigger
# 5. Run analysis notebook
jupyter notebook notebooks/social_analysis.ipynb
```

---

## Jupyter Notebook Workflow

The main analysis notebook includes:

### 1. Data Loading
- Connect to DynamoDB
- Query posts by time range
- Load into pandas DataFrame
- Basic data exploration

### 2. Sentiment Analysis
- Overall sentiment distribution
- Sentiment trends over time
- Top positive/negative posts
- Sentiment by hashtag

### 3. Entity Extraction
- Extract and count entities
- Top mentioned people
- Top mentioned organizations
- Top mentioned locations

### 4. Hashtag Analysis
- Most popular hashtags
- Hashtag co-occurrence network
- Hashtag sentiment correlation

### 5. Network Analysis
- User mention network
- Network centrality measures
- Community detection
- Influential users

### 6. Visualizations
- Sentiment timeline plots
- Word clouds by sentiment
- Network graphs
- Hashtag heatmaps

### 7. Export Results
- Save summary statistics
- Export filtered datasets
- Generate report

---

## What You'll Discover

### Social Media Insights
- Overall sentiment trends in your dataset
- Most positive and negative topics (hashtags)
- Influential users and communities
- Temporal patterns in sentiment
- Entity mentions and their sentiment

### AWS Insights
- Serverless sentiment analysis at scale
- Event-driven processing with S3 triggers
- NoSQL database design for social data
- Cost-effective ML with managed services
- Real-time processing capabilities

### Research Insights
- Reproducibility: Same code, same results
- Scalability: Process 1000 or 1M posts with same architecture
- Collaboration: Share pipelines and results
- Cost efficiency: Pay only for what you use

---

## Sample Data Format

### Input Format (JSON)
```json
[
  {
    "post_id": "12345",
    "text": "Loving the new climate research! #science #climate",
    "timestamp": 1699999999,
    "user_id": "user123",
    "username": "scientist_jane"
  },
  {
    "post_id": "12346",
    "text": "Great conference today @researcher #academic",
    "timestamp": 1699999998,
    "user_id": "user456",
    "username": "prof_john"
  }
]
```

### Input Format (CSV)
```csv
post_id,text,timestamp,user_id,username
12345,"Loving the new climate research! #science #climate",1699999999,user123,scientist_jane
12346,"Great conference today @researcher #academic",1699999998,user456,prof_john
```

### Output Format (DynamoDB)
```json
{
  "post_id": "12345",
  "timestamp": 1699999999,
  "text": "Loving the new climate research! #science #climate",
  "user_id": "user123",
  "username": "scientist_jane",
  "sentiment": "POSITIVE",
  "positive_score": 0.95,
  "negative_score": 0.01,
  "neutral_score": 0.03,
  "mixed_score": 0.01,
  "entities": [
    {"text": "climate research", "type": "OTHER", "score": 0.99}
  ],
  "hashtags": ["science", "climate"],
  "mentions": [],
  "processed_at": "2025-01-14T12:00:00Z"
}
```

---

## Next Steps

### Extend This Project
1. **More Data Sources**: Twitter API, Reddit API, Facebook Graph API
2. **Advanced Analysis**: Topic modeling (LDA), emotion detection
3. **Real-Time Processing**: Stream processing with Kinesis
4. **Dashboard**: Create live dashboard with QuickSight
5. **Alerts**: SNS notifications for sentiment anomalies
6. **Language Support**: Multi-language sentiment with Comprehend

### Move to Tier 3 (Production)
- CloudFormation for infrastructure-as-code
- Production-grade Lambda with error handling
- Multi-region deployment
- Advanced monitoring and alerting
- Cost optimization techniques
- Auto-scaling DynamoDB
- Data lake architecture

---

## Troubleshooting

### Common Issues

**"botocore.exceptions.NoCredentialsError"**
```bash
# Solution: Configure AWS credentials
aws configure
# Enter your Access Key ID, Secret Key, region, output format
```

**"AccessDeniedException: User is not authorized to perform: comprehend:DetectSentiment"**
```bash
# Solution: Add Comprehend permissions to IAM role
# In AWS Console: IAM > Roles > lambda-social-analysis
# Attach policy: ComprehendReadOnly
# Or create custom policy with comprehend:DetectSentiment
```

**"S3 bucket already exists"**
```bash
# Solution: Use a unique bucket name
s3://social-media-data-$(date +%s)-yourname
```

**"Lambda timeout"**
```python
# Solution: Increase timeout in Lambda console
# Default: 3 seconds
# Recommended: 30 seconds
# For large batches: 60 seconds
```

**"DynamoDB throughput exceeded"**
```python
# Solution: Use on-demand billing mode
# Or increase provisioned capacity
# Or reduce write rate in Lambda
```

**"Comprehend API throttling"**
```python
# Solution: Implement exponential backoff
# Use batch API for 25+ items
# Request service limit increase
# Add retry logic with time.sleep()
```

**"Cannot query DynamoDB - no results"**
```bash
# Solution: Check Lambda CloudWatch logs
# Verify Lambda is writing to correct table
# Check DynamoDB table in console
# Verify IAM permissions for DynamoDB write
```

See troubleshooting section in `setup_guide.md` for more solutions.

---

## Cleanup Guide

When finished, delete AWS resources to stop incurring charges:

```bash
# Delete S3 bucket and contents
aws s3 rm s3://social-media-data-xxxx --recursive
aws s3 rb s3://social-media-data-xxxx

# Delete DynamoDB table
aws dynamodb delete-table --table-name SocialMediaPosts

# Delete Lambda function
aws lambda delete-function --function-name analyze-sentiment

# Delete IAM role
aws iam delete-role-policy --role-name lambda-social-analysis \
  --policy-name lambda-policy
aws iam delete-role --role-name lambda-social-analysis

# Or use: python cleanup.py (automated)
```

See `cleanup_guide.md` for detailed instructions.

---

## Resources

### AWS Documentation
- [S3 Getting Started](https://docs.aws.amazon.com/s3/latest/userguide/getting-started.html)
- [Lambda Developer Guide](https://docs.aws.amazon.com/lambda/latest/dg/)
- [DynamoDB Developer Guide](https://docs.aws.amazon.com/dynamodb/)
- [Comprehend Developer Guide](https://docs.aws.amazon.com/comprehend/)
- [IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
- [Athena User Guide](https://docs.aws.amazon.com/athena/latest/ug/)

### Social Media Analysis
- [Sentiment Analysis Overview](https://en.wikipedia.org/wiki/Sentiment_analysis)
- [Social Network Analysis Tutorial](https://networkx.org/documentation/stable/tutorial.html)
- [Twitter API Documentation](https://developer.twitter.com/en/docs)

### Python Libraries
- [boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- [pandas Documentation](https://pandas.pydata.org/docs/)
- [NetworkX Documentation](https://networkx.org/)
- [Matplotlib Documentation](https://matplotlib.org/)

---

## Getting Help

### Project Issues
- GitHub Issues: https://github.com/research-jumpstart/research-jumpstart/issues
- Tag: `social-science`, `tier-2`, `aws`, `sentiment-analysis`

### AWS Support
- AWS Support Center: https://console.aws.amazon.com/support/
- AWS Forums: https://forums.aws.amazon.com/
- Stack Overflow: Tag `amazon-aws`, `boto3`, `aws-comprehend`

### Social Science Help
- Digital Methods Initiative: https://digitalmethods.net/
- Computational Social Science Forum: Various online communities

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
- $15 threshold (warning)
- $25 threshold (critical)

---

## Key Differences from Tier 1

| Aspect | Tier 1 (Studio Lab) | Tier 2 (AWS) |
|--------|-------------------|------------|
| **Platform** | Free SageMaker Studio Lab | AWS account required |
| **Cost** | Free | $6-10 per run |
| **Storage** | 15GB persistent | Unlimited (pay per GB) |
| **Compute** | 4GB RAM, 12-hour sessions | Scalable (pay per second) |
| **ML Services** | Local libraries only | AWS Comprehend API |
| **Data Scale** | Limited to 15GB | Gigabytes to terabytes |
| **Persistence** | Session-based | Permanent DynamoDB storage |
| **Collaboration** | Limited | Full team access via S3/DynamoDB |

---

## What's Next?

After completing this project:

**Skill Building**
- AWS Lambda advanced features (layers, versions)
- DynamoDB advanced queries (GSI, LSI)
- AWS Comprehend custom models
- Real-time stream processing

**Project Extensions**
- Real-time Twitter stream analysis
- Multi-language sentiment analysis
- Emotion detection (joy, anger, sadness)
- Automated reporting with SNS
- Dashboard creation with QuickSight

**Tier 3 Transition**
- Production infrastructure with CloudFormation
- High-availability architecture
- Advanced monitoring and alerting
- CI/CD pipeline with CodePipeline
- Multi-region deployment
- Cost optimization at scale

---

## Citation

If you use this project in your research:

```bibtex
@software{research_jumpstart_social_tier2,
  title = {Social Media Sentiment Analysis with AWS: Tier 2},
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
