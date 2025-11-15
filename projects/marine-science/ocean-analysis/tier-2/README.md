# Ocean Data Analysis with S3 and Lambda - Tier 2 Project

**Duration:** 2-4 hours | **Cost:** $7-13 | **Platform:** AWS + Local machine

Analyze oceanographic data using serverless AWS services. Upload ocean observations to S3, process marine parameters with Lambda, store results in DynamoDB, and query with Athena—all without managing servers.

---

## What You'll Build

A cloud-native oceanographic analysis pipeline that demonstrates:

1. **Data Storage** - Upload ocean observation data to S3
2. **Serverless Processing** - Lambda functions to analyze marine parameters
3. **Results Storage** - Store processed observations in DynamoDB
4. **Data Querying** - Query results with Athena or Python scripts
5. **Alert System** - SNS notifications for marine anomalies

This bridges the gap between local analysis (Tier 1) and production infrastructure (Tier 3).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Your Local Machine                                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Python Scripts & Jupyter Notebook                          │ │
│  │ • upload_to_s3.py - Upload oceanographic data            │ │
│  │ • lambda_function.py - Analyze ocean parameters          │ │
│  │ • query_results.py - Retrieve results                    │ │
│  │ • ocean_analysis.ipynb - Visualization                   │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  AWS Services                                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  S3 Bucket   │→ │   Lambda     │→ │  DynamoDB    │          │
│  │              │  │              │  │              │          │
│  │ • Raw ocean  │  │ • SST        │  │ • Ocean      │          │
│  │   data (CSV/ │  │   analysis   │  │   observations│          │
│  │   NetCDF)    │  │ • Salinity   │  │ • Metrics    │          │
│  │              │  │ • pH metrics │  │ • Anomalies  │          │
│  └──────────────┘  │ • Chlorophyll│  └──────────────┘          │
│                    │ • Anomaly    │                             │
│  ┌──────────────┐  │   detection  │  ┌──────────────┐          │
│  │  SNS Topic   │← └──────────────┘  │  Athena      │          │
│  │              │                     │              │          │
│  │ • Marine     │                     │ • SQL queries│          │
│  │   heatwave   │                     │   on results │          │
│  │   alerts     │                     │              │          │
│  │ • Acidify    │                     │              │          │
│  │   warnings   │                     │              │          │
│  └──────────────┘                     └──────────────┘          │
│  ┌──────────────────────────────────────────────────────────────┐
│  │  IAM Role (Permissions)                                      │
│  │  • S3 read/write                                             │
│  │  • Lambda execution                                          │
│  │  • DynamoDB write/read                                       │
│  │  • SNS publish                                               │
│  │  • CloudWatch logging                                        │
│  └──────────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Required Knowledge
- Basic Python (pandas, boto3)
- Understanding of oceanographic measurements
- AWS fundamentals (S3, Lambda, IAM)
- Marine science concepts (temperature, salinity, pH, etc.)

### Required Tools
- **Python 3.8+** with:
  - boto3 (AWS SDK)
  - pandas (data manipulation)
  - numpy (numerical computing)
  - xarray (for NetCDF files)
  - netCDF4 (NetCDF support)
  - matplotlib, seaborn (visualization)
  - cartopy (mapping)

- **AWS Account** with:
  - S3 bucket access
  - Lambda permissions
  - DynamoDB access
  - IAM role creation capability
  - SNS for notifications
  - (Optional) Athena query capability

### Installation

```bash
# Clone the project
git clone https://github.com/research-jumpstart/research-jumpstart.git
cd research-jumpstart/projects/marine-science/ocean-analysis/tier-2

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
# - S3 bucket: ocean-data-{user-id}
# - IAM role: lambda-ocean-processor
# - Lambda function: analyze-ocean-data
# - DynamoDB table: OceanObservations
# - SNS topic: ocean-anomaly-alerts
```

### Step 2: Upload Sample Data (3 minutes)
```bash
python scripts/upload_to_s3.py
```

### Step 3: Process Data (2 minutes)
Lambda automatically processes uploaded data via S3 trigger.

### Step 4: Query Results (2 minutes)
```bash
python scripts/query_results.py
```

### Step 5: Visualize (5 minutes)
Open `notebooks/ocean_analysis.ipynb` in Jupyter and run all cells.

---

## Detailed Workflow

### 1. Data Preparation (Setup)

**What's happening:**
- Generate or upload oceanographic data (CTD profiles, satellite SST)
- Support for CSV and NetCDF formats
- Create S3 bucket with proper permissions

**Data parameters:**
- **Temperature:** Sea surface temperature (SST), water column profiles
- **Salinity:** Practical Salinity Units (PSU)
- **pH:** Ocean acidification indicator
- **Dissolved Oxygen (DO):** Marine ecosystem health
- **Chlorophyll-a:** Primary productivity indicator
- **Coordinates:** GPS location, depth
- **Timestamps:** UTC datetime

**Files involved:**
- `setup_guide.md` - Step-by-step AWS setup
- `scripts/upload_to_s3.py` - Upload automation

**Time:** 20-30 minutes (includes setup_guide steps)

### 2. Lambda Processing

**What's happening:**
- Lambda reads ocean data from S3 (CSV or NetCDF)
- Calculates marine metrics and indices
- Detects anomalies and critical conditions
- Sends SNS alerts for marine heatwaves or acidification
- Writes results to DynamoDB

**Analysis performed:**
```python
# Temperature Analysis
- Calculate temperature anomalies (deviation from climatology)
- Detect marine heatwaves (>3°C above normal for >5 days)
- Analyze stratification (temperature gradient with depth)
- Identify upwelling indices

# Ocean Chemistry
- pH trends and ocean acidification metrics
- Aragonite saturation state (Ωarag)
- Dissolved oxygen levels (hypoxia detection)
- Carbon system calculations

# Biological Productivity
- Chlorophyll-a concentrations
- Primary production estimates
- Nutrient availability indicators

# Physical Oceanography
- Salinity anomalies
- Density calculations
- Mixed layer depth
- Potential temperature and potential density
```

**Lambda function event:**
```python
# Event: S3 upload trigger
{
    'Records': [{
        'bucket': {'name': 'ocean-data-xxxx'},
        'object': {'key': 'raw/ctd_profile_2024.csv'}
    }]
}

# Processing: Calculate marine metrics
# Output: DynamoDB record + optional SNS alert
```

**Files involved:**
- `scripts/lambda_function.py` - Processing function code
- `setup_guide.md` - Lambda deployment steps

**Time:** 5-10 seconds per file

### 3. Results Storage

**What's happening:**
- Processed observations stored in DynamoDB
- Original data files kept in S3 for reference
- Organized structure for efficient queries

**DynamoDB Schema:**
```
OceanObservations Table
├── observation_id (Partition Key, String)   # Unique ID
├── timestamp (Sort Key, String)             # ISO 8601 UTC
└── Attributes:
    ├── location_name (String)               # "Gulf Stream", "Station ALOHA"
    ├── coordinates (Map)                    # {lat: 40.5, lon: -70.2, depth: 100}
    ├── temperature (Number)                 # Degrees Celsius
    ├── salinity (Number)                    # PSU
    ├── ph (Number)                          # pH units
    ├── dissolved_oxygen (Number)            # mg/L
    ├── chlorophyll (Number)                 # mg/m³
    ├── temperature_anomaly (Number)         # Degrees Celsius
    ├── stratification_index (Number)        # kg/m³
    ├── aragonite_saturation (Number)        # Ωarag
    ├── primary_production (Number)          # mg C/m²/day
    ├── anomaly_status (String)              # "normal", "warning", "critical"
    ├── anomaly_type (String)                # "marine_heatwave", "acidification", etc.
    ├── alert_sent (Boolean)                 # True if SNS notification sent
    └── data_quality (String)                # "excellent", "good", "fair", "poor"
```

**S3 Structure:**
```
s3://ocean-data-{your-id}/
├── raw/                              # Original data files
│   ├── ctd_profiles/
│   │   ├── station_001_2024.csv
│   │   └── station_002_2024.csv
│   ├── satellite_sst/
│   │   ├── sst_20240101.nc
│   │   └── sst_20240102.nc
│   └── biogeochemical/
│       ├── oxygen_profiles.csv
│       └── chlorophyll_data.csv
├── processed/                         # Processed results
│   └── summary_statistics.json
└── logs/                              # Processing logs
    └── lambda_execution.log
```

### 4. Results Analysis

**What's happening:**
- Query DynamoDB for specific observations
- Download results to local machine
- Analyze with Jupyter notebook
- Create publication-quality visualizations
- (Optional) Run SQL queries with Athena

**Visualizations:**
- Depth profiles (temperature, salinity, oxygen)
- Time series (SST trends, pH changes)
- Spatial maps (ocean basins, currents)
- Anomaly plots (marine heatwave events)
- Scatter plots (T-S diagrams, pH vs temperature)

**Files involved:**
- `notebooks/ocean_analysis.ipynb` - Analysis notebook
- `scripts/query_results.py` - Query and retrieve data
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
│   └── ocean_analysis.ipynb          # Main analysis notebook
│
└── scripts/
    ├── upload_to_s3.py               # Upload ocean data to S3
    ├── lambda_function.py            # Lambda processing function
    └── query_results.py              # Download and analyze results
```

---

## Cost Breakdown

**Total estimated cost: $7-13 per run**

| Service | Usage | Cost |
|---------|-------|------|
| **S3 Storage** | 3GB × 7 days | $0.21 |
| **S3 Requests** | ~500 PUT/GET requests | $0.03 |
| **Lambda Executions** | 50 invocations × 20 sec | $0.50 |
| **Lambda Compute** | 50 GB-seconds | $1.00 |
| **DynamoDB Storage** | ~10MB for 1000 records | $0.25 |
| **DynamoDB Writes** | ~1000 write requests | $1.25 |
| **SNS Notifications** | 10 email alerts | $0.01 |
| **Data Transfer** | Upload + download (2GB) | $0.20 |
| **Athena Queries** | 5 queries × 5GB scanned | $2.50 |
| **CloudWatch Logs** | 10MB logs | Free |
| **Total** | | **$5.95-$13.00** |

**Cost optimization tips:**
1. Delete S3 objects after analysis ($0.21 savings)
2. Use Lambda for < 30 seconds processing
3. Run Athena queries in batch (not interactive)
4. Use DynamoDB on-demand pricing
5. Clean up immediately after completion

**Free Tier Usage:**
- **S3**: First 5GB free (12 months)
- **Lambda**: 1M invocations free (12 months)
- **DynamoDB**: 25GB storage free (always free)
- **SNS**: 1000 email notifications free (12 months)

---

## Key Learning Objectives

### AWS Services
- S3 bucket creation and management
- Lambda function deployment with event triggers
- DynamoDB NoSQL database design
- IAM role creation with least privilege
- SNS notification setup
- CloudWatch monitoring and logs
- (Optional) Athena for serverless SQL queries

### Cloud Concepts
- Object storage vs block storage
- Serverless computing (no servers to manage)
- Event-driven architecture (S3 → Lambda trigger)
- NoSQL database patterns
- Cost-conscious cloud design
- Monitoring and alerting

### Oceanographic Analysis
- Marine parameter calculations
- Temperature anomaly detection
- Ocean acidification metrics
- Productivity indicators
- Water column structure analysis
- Marine heatwave identification
- Data quality assessment

---

## Time Estimates

**First Time Setup:**
- Read setup_guide.md: 10 minutes
- Create S3 bucket: 2 minutes
- Create DynamoDB table: 3 minutes
- Create IAM role: 5 minutes
- Deploy Lambda: 5 minutes
- Configure SNS: 3 minutes
- **Subtotal setup: 28 minutes**

**Data Processing:**
- Generate sample data: 5 minutes
- Upload data: 5-10 minutes
- Lambda processing: 2-5 minutes
- **Subtotal processing: 12-20 minutes**

**Analysis:**
- Query results: 3 minutes
- Jupyter analysis: 30-45 minutes
- Generate figures: 10-15 minutes
- **Subtotal analysis: 43-63 minutes**

**Total time: 1.5-2 hours** (including setup)

---

## Running the Project

### Option 1: Automated (Recommended for First Time)

```bash
# Step 1: Setup AWS services (follow prompts)
# See setup_guide.md for detailed instructions

# Step 2: Generate and upload sample oceanographic data
python scripts/upload_to_s3.py --generate-sample

# Step 3: Lambda automatically processes data
# Check CloudWatch logs for progress

# Step 4: Query results
python scripts/query_results.py --location "all" --days 7

# Step 5: Analyze with Jupyter
jupyter notebook notebooks/ocean_analysis.ipynb
```

### Option 2: Manual (Detailed Control)

```bash
# 1. Create S3 bucket manually
aws s3 mb s3://ocean-data-$(whoami)-$(date +%s) --region us-east-1

# 2. Upload your own ocean data
aws s3 cp your_ocean_data.csv s3://ocean-data-xxxx/raw/

# 3. Deploy Lambda (see setup_guide.md)

# 4. Query DynamoDB directly
aws dynamodb query \
  --table-name OceanObservations \
  --key-condition-expression "observation_id = :id" \
  --expression-attribute-values '{":id":{"S":"obs-001"}}'

# 5. Run analysis notebook
jupyter notebook notebooks/ocean_analysis.ipynb
```

---

## Jupyter Notebook Workflow

The main analysis notebook includes:

1. **Data Loading**
   - Query DynamoDB for observations
   - Load into pandas DataFrame
   - Download NetCDF files from S3 if needed

2. **Exploratory Analysis**
   - Summary statistics
   - Temperature and salinity distributions
   - pH and dissolved oxygen trends
   - Chlorophyll-a patterns

3. **Ocean Metrics**
   - Calculate derived parameters
   - Stratification indices
   - Productivity estimates
   - Anomaly identification

4. **Visualization**
   - Depth profiles (CTD-style plots)
   - Time series (parameter trends)
   - Spatial maps (using cartopy)
   - T-S diagrams
   - Anomaly heatmaps

5. **Export**
   - Save processed data
   - Generate publication figures
   - Create summary reports

---

## What You'll Discover

### Marine Science Insights
- How ocean temperature varies with depth and location
- Ocean acidification trends in your data
- Marine heatwave detection and characteristics
- Productivity patterns from chlorophyll data
- Water mass characteristics from T-S relationships
- Hypoxic zones from dissolved oxygen profiles

### AWS Insights
- Serverless computing advantages for data analysis
- Event-driven processing pipelines
- Cost-effective cloud storage for ocean data
- Scale from local analysis to global datasets
- NoSQL database benefits for time series data

### Research Insights
- Reproducibility: Same code, same results
- Collaboration: Share workflows and data
- Scale: Process terabytes as easily as megabytes
- Persistence: Results saved permanently
- Automation: Continuous monitoring possible

---

## Marine Anomaly Detection

The Lambda function detects and alerts on:

### Marine Heatwaves
- Temperature > 3°C above climatological mean
- Duration > 5 days
- SNS alert: "MARINE HEATWAVE: Location X shows +4.2°C anomaly"

### Ocean Acidification
- pH < 7.8 (warning)
- pH < 7.6 (critical)
- Aragonite saturation < 1.0 (critical for shellfish)
- SNS alert: "ACIDIFICATION WARNING: pH 7.7 at Station Y"

### Hypoxia
- Dissolved oxygen < 2.0 mg/L (severe)
- Dissolved oxygen < 4.0 mg/L (warning)
- SNS alert: "HYPOXIA DETECTED: DO 1.8 mg/L at depth 200m"

### Biological Anomalies
- Chlorophyll-a > 20 mg/m³ (harmful algal bloom)
- Chlorophyll-a < 0.1 mg/m³ (biological desert)
- SNS alert: "BLOOM ALERT: Chlorophyll 25 mg/m³"

---

## Next Steps

### Extend This Project
1. **More Locations**: Add multiple oceanographic stations
2. **More Parameters**: Include nutrients, currents, waves
3. **Time Series Analysis**: Analyze multi-year trends
4. **Automation**: Daily processing of new observations
5. **Machine Learning**: Predict ocean conditions
6. **Real-time Monitoring**: Connect to live buoy data

### Move to Tier 3 (Production)
- CloudFormation for infrastructure-as-code
- Production-grade Lambda functions with layers
- Multi-region deployment
- Advanced monitoring with dashboards
- Cost optimization at scale
- API Gateway for public data access
- Step Functions for complex workflows

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
s3://ocean-data-$(whoami)-$(date +%s)
```

**"Lambda timeout"**
```python
# Solution: Increase timeout in Lambda console
# Default: 3 seconds
# Recommended: 30 seconds
# For large NetCDF files: 60 seconds
```

**"DynamoDB ProvisionedThroughputExceededException"**
```python
# Solution: Use on-demand billing mode
# AWS Console > DynamoDB > Tables > OceanObservations
# Capacity tab > Switch to "On-demand"
```

**"SNS not sending emails"**
```bash
# Solution: Confirm SNS subscription
# Check email (including spam) for confirmation link
# Verify subscription in SNS console
```

**"NetCDF read error in Lambda"**
```python
# Solution: Ensure netCDF4 library in Lambda deployment package
# Create Lambda layer with netCDF4 and dependencies
# See setup_guide.md for Lambda layer creation
```

**"Invalid ocean data values"**
```python
# Solution: Check data quality
# Valid ranges:
# - Temperature: -2°C to 40°C
# - Salinity: 0-42 PSU
# - pH: 7.0-8.5
# - DO: 0-15 mg/L
# - Chlorophyll: 0-100 mg/m³
```

---

## Cleanup Guide

When finished, delete AWS resources to stop incurring charges:

```bash
# Option 1: Use cleanup script (recommended)
python cleanup.py

# Option 2: Manual cleanup
# Delete S3 bucket and contents
aws s3 rm s3://ocean-data-xxxx --recursive
aws s3 rb s3://ocean-data-xxxx

# Delete DynamoDB table
aws dynamodb delete-table --table-name OceanObservations

# Delete Lambda function
aws lambda delete-function --function-name analyze-ocean-data

# Delete SNS topic
aws sns delete-topic --topic-arn arn:aws:sns:us-east-1:ACCOUNT:ocean-anomaly-alerts

# Delete IAM role
aws iam detach-role-policy --role-name lambda-ocean-processor \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
aws iam detach-role-policy --role-name lambda-ocean-processor \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
aws iam detach-role-policy --role-name lambda-ocean-processor \
  --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess
aws iam detach-role-policy --role-name lambda-ocean-processor \
  --policy-arn arn:aws:iam::aws:policy/AmazonSNSFullAccess
aws iam delete-role --role-name lambda-ocean-processor
```

See `cleanup_guide.md` for detailed instructions.

---

## Resources

### AWS Documentation
- [S3 Getting Started](https://docs.aws.amazon.com/s3/latest/userguide/getting-started.html)
- [Lambda Developer Guide](https://docs.aws.amazon.com/lambda/latest/dg/)
- [DynamoDB Best Practices](https://docs.aws.amazon.com/dynamodb/)
- [IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
- [SNS Developer Guide](https://docs.aws.amazon.com/sns/)

### Oceanographic Data
- [NOAA Ocean Data](https://www.nodc.noaa.gov/)
- [Copernicus Marine Service](https://marine.copernicus.eu/)
- [Argo Float Data](https://argo.ucsd.edu/)
- [World Ocean Database](https://www.ncei.noaa.gov/products/world-ocean-database)

### Python Libraries
- [boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- [xarray Documentation](https://xarray.pydata.org/)
- [pandas Documentation](https://pandas.pydata.org/docs/)
- [cartopy Documentation](https://scitools.org.uk/cartopy/)

### Marine Science Resources
- [Ocean Acidification Portal](https://www.pmel.noaa.gov/co2/story/Ocean+Acidification)
- [Marine Heatwaves](http://www.marineheatwaves.org/)
- [SeaWiFS Project (Chlorophyll)](https://oceancolor.gsfc.nasa.gov/)

---

## Getting Help

### Project Issues
- GitHub Issues: https://github.com/research-jumpstart/research-jumpstart/issues
- Tag: `marine-science`, `tier-2`, `aws`

### AWS Support
- AWS Support Center: https://console.aws.amazon.com/support/
- AWS Forums: https://forums.aws.amazon.com/
- Stack Overflow: Tag `amazon-aws`, `boto3`

### Marine Science Help
- OceanObs Community: https://www.oceanobs.org/
- Marine science mailing lists and forums

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
- $15 threshold (critical)

---

## Key Differences from Tier 1

| Aspect | Tier 1 (Studio Lab) | Tier 2 (AWS) |
|--------|-------------------|------------|
| **Platform** | Free SageMaker Studio Lab | AWS account required |
| **Cost** | Free | $7-13 per run |
| **Storage** | 15GB persistent | Unlimited (pay per GB) |
| **Compute** | 4GB RAM, 12-hour sessions | Scalable (pay per second) |
| **Data Scale** | Limited to 15GB | Petabytes possible |
| **Automation** | Manual notebook execution | Event-driven Lambda triggers |
| **Persistence** | Session-based | Permanent S3/DynamoDB |
| **Collaboration** | Individual | Full team access |
| **Alerts** | None | Real-time SNS notifications |

---

## What's Next?

After completing this project:

**Skill Building**
- AWS Lambda advanced features (layers, destinations)
- Serverless architecture patterns
- Cost optimization techniques
- Security best practices (VPC, encryption)

**Project Extensions**
- Real-time ocean sensor monitoring
- Automated analysis pipelines
- Integration with other services (Step Functions, SQS)
- Dashboard creation (QuickSight, Grafana)
- API Gateway for data sharing

**Tier 3 Transition**
- Production infrastructure with CloudFormation
- High-availability architecture
- Advanced monitoring and alerting
- Multi-region deployment
- Auto-scaling Lambda functions

---

## Citation

If you use this project in your research:

```bibtex
@software{research_jumpstart_ocean_tier2,
  title = {Ocean Data Analysis with S3 and Lambda: Tier 2},
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

## Acknowledgments

Marine science guidance and oceanographic parameter calculations based on:
- UNESCO Equation of State of Seawater (EOS-80)
- TEOS-10 Thermodynamic Equation of Seawater
- NOAA Ocean Acidification Program
- Marine Heatwaves Working Group

---

**Ready to start?** Follow the [setup_guide.md](setup_guide.md) to get started!

**Last updated:** 2025-11-14 | Research Jumpstart v1.0.0
