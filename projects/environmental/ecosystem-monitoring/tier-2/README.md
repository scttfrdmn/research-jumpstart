# Environmental Sensor Data Analysis with AWS - Tier 2 Project

**Duration:** 2-4 hours | **Cost:** $7-12 | **Platform:** AWS + Local machine

Analyze environmental sensor data using serverless AWS services. Upload IoT sensor readings to S3, process data with Lambda for air quality and water quality analysis, trigger pollution alerts via SNS, and query results using DynamoDB—all without managing servers.

---

## What You'll Build

A cloud-native environmental monitoring pipeline that demonstrates:

1. **IoT Data Storage** - Upload sensor data (air quality, water quality) to S3
2. **Serverless Processing** - Lambda functions to analyze environmental parameters in parallel
3. **Real-time Alerting** - SNS notifications for pollution threshold violations
4. **Results Storage** - Store processed metrics and alerts in DynamoDB
5. **Data Querying** - Query environmental readings with boto3 or Athena

This bridges the gap between local analysis (Tier 1) and production infrastructure (Tier 3).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Your Local Machine                                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Python Scripts                                             │ │
│  │ • upload_to_s3.py - Upload sensor data                   │ │
│  │ • lambda_function.py - Process environmental data        │ │
│  │ • query_results.py - Analyze readings                    │ │
│  │ • environmental_analysis.ipynb - Jupyter notebook        │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  AWS Services                                                    │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  │  S3 Bucket       │  │ Lambda Function  │  │  DynamoDB        │
│  │                  │  │                  │  │                  │
│  │ • Sensor data    │→ │ Environmental    │→ │ Readings &       │
│  │   (CSV/JSON)     │  │ analysis:        │  │ Metrics:         │
│  │ • Time series    │  │ - AQI calc       │  │ - Location       │
│  │   readings       │  │ - Water quality  │  │ - Timestamp      │
│  │                  │  │ - Anomalies      │  │ - Parameters     │
│  │                  │  │ - Trends         │  │ - Alert status   │
│  └──────────────────┘  └──────┬───────────┘  └──────────────────┘
│                               │                                  │
│                               ▼                                  │
│  ┌──────────────────────────────────────────────────────────────┐
│  │  SNS Topic (Pollution Alerts)                                │
│  │  • Email notifications                                        │
│  │  • Threshold violations (PM2.5, CO2, pH)                    │
│  │  • Critical alerts for environmental hazards                 │
│  └──────────────────────────────────────────────────────────────┘
│  ┌──────────────────────────────────────────────────────────────┐
│  │  IAM Role (Permissions)                                      │
│  │  • S3 read/write                                             │
│  │  • DynamoDB write                                            │
│  │  • SNS publish                                               │
│  │  • CloudWatch logging                                        │
│  └──────────────────────────────────────────────────────────────┘
│  ┌──────────────────────────────────────────────────────────────┐
│  │  Athena (Optional)                                           │
│  │  • SQL queries on time series data                           │
│  │  • Historical trend analysis                                 │
│  └──────────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Required Knowledge
- Basic Python (pandas, boto3)
- Understanding of environmental monitoring concepts
- AWS fundamentals (S3, Lambda, DynamoDB, SNS, IAM)

### Required Tools
- **Python 3.8+** with:
  - boto3 (AWS SDK)
  - pandas (data manipulation)
  - numpy (numerical computing)
  - matplotlib, seaborn (visualization)
  - jupyter (notebook environment)

- **AWS Account** with:
  - S3 bucket access
  - Lambda permissions
  - DynamoDB access
  - SNS topic creation
  - IAM role creation capability
  - (Optional) Athena query capability

### Installation

```bash
# Clone the project
git clone https://github.com/research-jumpstart/research-jumpstart.git
cd research-jumpstart/projects/environmental/ecosystem-monitoring/tier-2

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
# - S3 bucket: environmental-data-{your-id}
# - IAM role: lambda-environmental-processor
# - Lambda function: process-sensor-data
# - DynamoDB table: EnvironmentalReadings
# - SNS topic: environmental-alerts
```

### Step 2: Generate and Upload Sample Data (3 minutes)
```bash
python scripts/upload_to_s3.py --generate-sample
```

### Step 3: Process Data (2 minutes)
Lambda will be triggered automatically by S3 uploads, or invoke manually:
```bash
# Lambda processes data automatically on upload
# Or test with direct invocation (see setup_guide.md)
```

### Step 4: Query Results (2 minutes)
```bash
python scripts/query_results.py --location river-01 --days 7
```

### Step 5: Visualize (5 minutes)
Open `notebooks/environmental_analysis.ipynb` in Jupyter and run all cells.

---

## Detailed Workflow

### 1. Data Preparation

**What's happening:**
- Generate or upload environmental sensor data
- Support for multiple sensor types:
  - Air quality sensors (PM2.5, PM10, CO2, NO2, O3)
  - Weather sensors (temperature, humidity, pressure)
  - Water quality sensors (pH, dissolved oxygen, turbidity, conductivity)
  - Soil sensors (moisture, NPK, temperature)
- Time series data with GPS coordinates and timestamps

**Files involved:**
- `setup_guide.md` - Step-by-step AWS setup
- `scripts/upload_to_s3.py` - Upload automation with sample data generation

**Time:** 20-30 minutes (includes setup_guide steps)

### 2. Lambda Processing

**What's happening:**
- Lambda reads sensor data from S3 (CSV or JSON format)
- Calculates environmental indices:
  - **Air Quality Index (AQI)** - EPA standard calculation
  - **Water Quality Index (WQI)** - Comprehensive water quality score
  - **Pollution Load Index (PLI)** - Heavy metal contamination
- Detects anomalies and threshold violations
- Analyzes trends (daily, weekly patterns)
- Sends SNS alerts for critical pollution levels
- Stores results in DynamoDB with metadata

**Lambda function**:
```python
# Event: S3 upload trigger
{
    'Records': [{
        'bucket': {'name': 'environmental-data-xxxx'},
        'object': {'key': 'raw/air_quality_2025_01_14.csv'}
    }]
}

# Processing steps:
# 1. Parse sensor readings (timestamp, location, parameters)
# 2. Calculate AQI from PM2.5, PM10, O3, NO2, CO
# 3. Calculate WQI from pH, DO, turbidity, conductivity
# 4. Detect threshold violations (EPA/WHO standards)
# 5. Identify anomalies (statistical outliers)
# 6. Store in DynamoDB
# 7. Send SNS alerts if critical levels detected

# Output: DynamoDB record + SNS notification
```

**Files involved:**
- `scripts/lambda_function.py` - Processing function code
- `setup_guide.md` - Lambda deployment steps

**Time:** 5-10 minutes execution per batch

### 3. Results Storage

**What's happening:**
- Processed readings stored in DynamoDB
- Schema optimized for time series queries
- Supports queries by location, timestamp, parameter type
- Includes calculated indices (AQI, WQI) and alert status

**DynamoDB Table Structure:**
```
EnvironmentalReadings
├── partition_key: location_id (string)     # e.g., "station-01"
├── sort_key: timestamp (string)            # ISO 8601 format
├── attributes:
│   ├── reading_id: unique identifier
│   ├── sensor_type: air|water|soil|weather
│   ├── parameters: {PM25, PM10, CO2, temp, humidity, ...}
│   ├── calculated_metrics: {AQI, WQI, anomaly_score}
│   ├── alert_status: none|warning|critical
│   ├── alert_message: description if alert triggered
│   ├── coordinates: {latitude, longitude}
│   ├── processing_time_ms: Lambda execution time
│   └── data_quality_score: 0-100
```

**S3 Structure:**
```
s3://environmental-data-{your-id}/
├── raw/                          # Original sensor data
│   ├── air_quality/
│   │   ├── station_01_2025_01_14.csv
│   │   ├── station_02_2025_01_14.csv
│   │   └── ...
│   ├── water_quality/
│   │   ├── river_01_2025_01_14.csv
│   │   └── ...
│   └── weather/
│       └── weather_2025_01_14.csv
└── logs/                          # Processing logs
    └── lambda_execution.log
```

### 4. Results Analysis

**What's happening:**
- Query DynamoDB for specific locations, time ranges, or alert types
- Analyze trends and patterns in environmental data
- Create visualizations (time series plots, heatmaps, correlation matrices)
- (Optional) Run SQL queries with Athena for complex analysis

**Files involved:**
- `notebooks/environmental_analysis.ipynb` - Analysis notebook
- `scripts/query_results.py` - Command-line query tool
- (Optional) Athena for advanced SQL queries

**Time:** 30-45 minutes analysis

---

## Environmental Monitoring Capabilities

### Air Quality Monitoring

**Measured Parameters:**
- PM2.5 (particulate matter < 2.5 μm)
- PM10 (particulate matter < 10 μm)
- CO2 (carbon dioxide)
- NO2 (nitrogen dioxide)
- O3 (ozone)
- CO (carbon monoxide)
- Temperature, humidity

**Air Quality Index (AQI) Calculation:**
```python
# EPA standard AQI calculation
# Ranges: 0-50 (Good), 51-100 (Moderate), 101-150 (Unhealthy for sensitive),
#         151-200 (Unhealthy), 201-300 (Very unhealthy), 301+ (Hazardous)

def calculate_aqi(pm25, pm10, o3, no2, co):
    """Calculate AQI from multiple pollutants"""
    # Implementation in lambda_function.py
    return max(aqi_pm25, aqi_pm10, aqi_o3, aqi_no2, aqi_co)
```

**Alert Thresholds:**
- PM2.5 > 35.4 μg/m³ (24-hour average) → Warning
- PM2.5 > 55.4 μg/m³ → Critical alert
- AQI > 150 → Unhealthy alert

### Water Quality Monitoring

**Measured Parameters:**
- pH (acidity/alkalinity)
- Dissolved Oxygen (DO)
- Turbidity (clarity)
- Conductivity (salinity)
- Temperature
- Total Dissolved Solids (TDS)
- Heavy metals (optional: Pb, Cd, Hg)

**Water Quality Index (WQI) Calculation:**
```python
# Weighted WQI calculation
# Ranges: 0-25 (Excellent), 26-50 (Good), 51-75 (Poor),
#         76-100 (Very poor), 100+ (Unsuitable)

def calculate_wqi(ph, do, turbidity, conductivity, temperature):
    """Calculate WQI from multiple parameters"""
    # Implementation in lambda_function.py
    return weighted_sum / total_weight * 100
```

**Alert Thresholds:**
- pH < 6.5 or > 8.5 → Warning
- Dissolved Oxygen < 5 mg/L → Critical alert
- WQI > 75 → Poor water quality alert

### Soil Monitoring

**Measured Parameters:**
- Moisture content
- Temperature
- NPK (Nitrogen, Phosphorus, Potassium)
- pH
- Electrical conductivity

### Weather Monitoring

**Measured Parameters:**
- Temperature, humidity
- Barometric pressure
- Wind speed and direction
- Precipitation
- Solar radiation

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
│   └── environmental_analysis.ipynb  # Main analysis notebook
│
├── scripts/
│   ├── upload_to_s3.py               # Upload sensor data to S3
│   ├── lambda_function.py            # Lambda processing function
│   ├── query_results.py              # Query DynamoDB results
│   └── __init__.py
│
└── sample_data/
    └── README.md                     # Sample data documentation
```

---

## Cost Breakdown

**Total estimated cost: $7-12 per run**

| Service | Usage | Cost |
|---------|-------|------|
| **S3 Storage** | 1GB × 7 days | $0.16 |
| **S3 Requests** | ~500 PUT/GET requests | $0.03 |
| **Lambda Executions** | 200 invocations × 30s | $0.40 |
| **Lambda Compute** | 200 × 128MB × 30s | $1.00 |
| **DynamoDB Writes** | 1000 write requests | $1.25 |
| **DynamoDB Storage** | 500MB × 1 month | $0.13 |
| **SNS Notifications** | 20 email alerts | $0.00 |
| **Data Transfer** | Upload + download (1GB) | $0.10 |
| **Athena Queries (Optional)** | 5 queries × 1GB scanned | $0.03 |
| **Total** | | **$3.10 base** |

**Note:** Costs can increase with:
- More sensor locations (+$0.50 per 100 additional sensors)
- Higher frequency readings (+$0.20 per 1000 readings)
- Larger time series data (+$0.01 per additional GB)
- More SNS notifications ($0.50 per 1000 emails)

**Cost optimization tips:**
1. Delete S3 data after analysis ($0.16 savings per week)
2. Use DynamoDB on-demand billing (pay only for what you use)
3. Set Lambda timeout to 30 seconds (not 300)
4. Batch sensor readings (upload every hour, not every minute)
5. Use SNS for critical alerts only (not all readings)

**Free Tier Usage:**
- **S3**: First 5GB free (12 months)
- **Lambda**: 1M invocations free per month (always)
- **DynamoDB**: 25GB storage free (always)
- **SNS**: 1000 email notifications free per month (always)

**Realistic range for learning project: $7-12**
This includes safety margin for multiple test runs and experimentation.

---

## Key Learning Objectives

### AWS Services
- S3 bucket creation and data organization
- Lambda function deployment and event triggers
- DynamoDB schema design for time series data
- SNS topic creation and email notifications
- IAM role creation with least privilege
- CloudWatch monitoring and logs
- (Optional) Athena for serverless SQL queries

### Cloud Concepts
- Object storage for IoT data
- Serverless computing (no servers to manage)
- Event-driven architecture (S3 → Lambda)
- NoSQL databases for time series
- Pub/sub messaging with SNS
- Cost-conscious design

### Environmental Science Skills
- Air Quality Index (AQI) calculation
- Water Quality Index (WQI) calculation
- Pollution threshold detection
- Time series analysis of environmental data
- Anomaly detection in sensor readings
- Environmental data visualization

---

## Time Estimates

**First Time Setup:**
- Read setup_guide.md: 10 minutes
- Create S3 bucket: 2 minutes
- Create IAM role: 5 minutes
- Deploy Lambda: 10 minutes
- Create DynamoDB table: 3 minutes
- Create SNS topic: 3 minutes
- Configure S3 trigger: 2 minutes
- **Subtotal setup: 35 minutes**

**Data Processing:**
- Generate sample data: 5 minutes
- Upload data to S3: 5 minutes
- Lambda processing: 5-10 minutes
- Verify DynamoDB results: 3 minutes
- **Subtotal processing: 18-23 minutes**

**Analysis:**
- Query results: 5 minutes
- Jupyter analysis: 30-45 minutes
- Generate visualizations: 10-15 minutes
- **Subtotal analysis: 45-65 minutes**

**Total time: 2-4 hours** (including setup and experimentation)

---

## Running the Project

### Option 1: Automated (Recommended for First Time)

```bash
# Step 1: Setup AWS services (follow setup_guide.md)
# This is manual - no automation script provided for learning purposes

# Step 2: Generate and upload sample data
python scripts/upload_to_s3.py \
  --bucket environmental-data-xxxx \
  --generate-sample \
  --num-stations 5 \
  --days 7

# Step 3: Lambda processes automatically (S3 trigger)
# Wait 1-2 minutes for processing

# Step 4: Query results
python scripts/query_results.py \
  --table EnvironmentalReadings \
  --location station-01 \
  --days 7

# Step 5: Analyze in notebook
jupyter notebook notebooks/environmental_analysis.ipynb
```

### Option 2: Manual (Detailed Control)

```bash
# 1. Create S3 bucket manually (see setup_guide.md)
aws s3 mb s3://environmental-data-$(date +%s) --region us-east-1

# 2. Create DynamoDB table
aws dynamodb create-table \
  --table-name EnvironmentalReadings \
  --attribute-definitions \
    AttributeName=location_id,AttributeType=S \
    AttributeName=timestamp,AttributeType=S \
  --key-schema \
    AttributeName=location_id,KeyType=HASH \
    AttributeName=timestamp,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST

# 3. Create SNS topic
aws sns create-topic --name environmental-alerts

# 4. Subscribe to SNS (email)
aws sns subscribe \
  --topic-arn arn:aws:sns:us-east-1:ACCOUNT:environmental-alerts \
  --protocol email \
  --notification-endpoint your-email@example.com

# 5. Deploy Lambda (see setup_guide.md for detailed steps)

# 6. Upload data
python scripts/upload_to_s3.py --bucket environmental-data-xxxx

# 7. Run analysis notebook
jupyter notebook notebooks/environmental_analysis.ipynb
```

---

## Jupyter Notebook Workflow

The main analysis notebook includes:

### 1. Setup and Configuration
- AWS credentials configuration
- Connect to S3, DynamoDB, SNS
- Set monitoring parameters

### 2. Data Generation (Optional)
- Generate synthetic sensor data
- Multiple locations and sensor types
- Realistic time series with trends and anomalies
- Add noise and missing data

### 3. Data Upload
- Upload to S3 with progress tracking
- Verify S3 upload success
- List uploaded files

### 4. Lambda Processing
- Monitor Lambda execution via CloudWatch
- Check DynamoDB for processed results
- Verify SNS alerts sent

### 5. Data Retrieval and Analysis
- Query DynamoDB by location, time, alert type
- Calculate statistics (mean, max, trends)
- Identify pollution hotspots
- Analyze alert frequency

### 6. Visualization
- Time series plots (PM2.5, AQI, pH, DO over time)
- Heatmaps (pollution levels by location and time)
- Correlation matrices (parameter relationships)
- Geographic maps (sensor locations with color-coded alerts)
- Distribution plots (parameter distributions)

### 7. Advanced Analysis
- Trend detection (increasing/decreasing pollution)
- Seasonal patterns (daily, weekly cycles)
- Anomaly identification (outliers, sensor failures)
- Comparative analysis (location vs location)

### 8. Reporting
- Generate summary statistics
- Create alert logs
- Export results to CSV
- Save visualizations

---

## What You'll Discover

### Environmental Insights
- How air quality varies by location and time of day
- Correlation between weather and pollution levels
- Water quality patterns in different seasons
- Impact of rain on air quality (washout effect)
- Diurnal temperature variation effects

### AWS Insights
- Serverless computing advantages for IoT data
- Event-driven architecture benefits
- DynamoDB performance for time series queries
- SNS reliability for real-time alerting
- Cost-effective cloud monitoring

### Research Insights
- Reproducibility: Same code, same results
- Scalability: Process data from 10 or 10,000 sensors
- Real-time: Alerts within seconds of threshold violation
- Collaboration: Share data and alerts with team
- Persistence: Historical data always available

---

## Next Steps

### Extend This Project

1. **More Sensors**: Add more sensor locations and types
2. **Real Sensors**: Integrate with real IoT devices (Raspberry Pi, Arduino)
3. **Advanced Analytics**: Machine learning for pollution prediction
4. **Dashboards**: Create real-time dashboard with QuickSight or Grafana
5. **Mobile App**: Build mobile app for field personnel
6. **Data Export**: Export to EPA reporting formats
7. **Historical Analysis**: Analyze long-term trends (months, years)

### Move to Tier 3 (Production)

Tier 3 uses CloudFormation for automated infrastructure:
- Infrastructure-as-code templates
- One-click deployment
- Multi-region deployment
- Advanced monitoring and alerting
- Auto-scaling and high availability
- Production-grade security
- Compliance (HIPAA, SOC 2)

See `/projects/environmental/ecosystem-monitoring/tier-3/` (when available)

### Real-World Integration

- Connect to EPA Air Quality System (AQS) API
- Integrate with USGS Water Quality Portal
- Use NASA satellite data (MODIS, Sentinel)
- Combine with weather forecasts (NOAA)
- Publish to public dashboards (PurpleAir, Sensor.Community)

---

## Troubleshooting

### Common Issues

**"botocore.exceptions.NoCredentialsError"**
```bash
# Solution: Configure AWS credentials
aws configure
# Enter your Access Key ID, Secret Key, region (us-east-1), output format (json)
```

**"S3 bucket already exists"**
```bash
# Solution: Use a unique bucket name
# Bucket names must be globally unique across ALL AWS accounts
s3://environmental-data-$(whoami)-$(date +%s)
```

**"Lambda timeout"**
```python
# Solution: Increase timeout in Lambda console
# Default: 3 seconds
# Recommended: 30 seconds for sensor data processing
# Maximum: 900 seconds (15 minutes)
```

**"DynamoDB ProvisionedThroughputExceededException"**
```bash
# Solution: Use on-demand billing mode
# Or increase provisioned capacity (costs more)
aws dynamodb update-table \
  --table-name EnvironmentalReadings \
  --billing-mode PAY_PER_REQUEST
```

**"SNS email not received"**
```bash
# Solution: Check spam folder and confirm subscription
# AWS sends confirmation email - click "Confirm subscription" link
aws sns list-subscriptions-by-topic \
  --topic-arn arn:aws:sns:us-east-1:ACCOUNT:environmental-alerts
```

**"Lambda out of memory"**
```python
# Solution: Increase Lambda memory allocation
# Default: 128 MB
# Recommended: 256 MB for sensor data processing
# Cost increases proportionally with memory
```

**"Invalid AQI or WQI values"**
```python
# Solution: Check sensor data format and units
# PM2.5 should be in μg/m³
# pH should be 0-14 scale
# Dissolved oxygen in mg/L
# Review lambda_function.py for expected units
```

See `setup_guide.md` for more detailed troubleshooting.

---

## Cleanup Guide

When finished, delete AWS resources to stop incurring charges:

```bash
# Option 1: Automated cleanup
# See cleanup_guide.md for detailed step-by-step instructions

# Option 2: Quick cleanup commands
# Delete S3 bucket and contents
aws s3 rm s3://environmental-data-xxxx --recursive
aws s3 rb s3://environmental-data-xxxx

# Delete Lambda function
aws lambda delete-function --function-name process-sensor-data

# Delete DynamoDB table
aws dynamodb delete-table --table-name EnvironmentalReadings

# Delete SNS topic
aws sns delete-topic --topic-arn arn:aws:sns:us-east-1:ACCOUNT:environmental-alerts

# Delete IAM role (detach policies first)
aws iam detach-role-policy \
  --role-name lambda-environmental-processor \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
aws iam detach-role-policy \
  --role-name lambda-environmental-processor \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
aws iam detach-role-policy \
  --role-name lambda-environmental-processor \
  --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess
aws iam detach-role-policy \
  --role-name lambda-environmental-processor \
  --policy-arn arn:aws:iam::aws:policy/AmazonSNSFullAccess
aws iam delete-role --role-name lambda-environmental-processor
```

See `cleanup_guide.md` for detailed verification steps.

---

## Resources

### AWS Documentation
- [S3 Getting Started](https://docs.aws.amazon.com/s3/latest/userguide/getting-started.html)
- [Lambda Developer Guide](https://docs.aws.amazon.com/lambda/latest/dg/)
- [DynamoDB Developer Guide](https://docs.aws.amazon.com/dynamodb/latest/developerguide/)
- [SNS Developer Guide](https://docs.aws.amazon.com/sns/latest/dg/)
- [IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
- [Athena User Guide](https://docs.aws.amazon.com/athena/latest/ug/)

### Environmental Monitoring
- [EPA Air Quality Index](https://www.airnow.gov/aqi/aqi-basics/)
- [WHO Air Quality Guidelines](https://www.who.int/news-room/fact-sheets/detail/ambient-(outdoor)-air-quality-and-health)
- [Water Quality Index Standards](https://www.epa.gov/waterdata/water-quality-index)
- [USGS Water Quality Portal](https://www.waterqualitydata.us/)

### Python Libraries
- [boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- [pandas Documentation](https://pandas.pydata.org/docs/)
- [matplotlib Documentation](https://matplotlib.org/stable/index.html)
- [seaborn Documentation](https://seaborn.pydata.org/)

### IoT and Sensors
- [PurpleAir API](https://www2.purpleair.com/)
- [Sensor.Community](https://sensor.community/en/)
- [OpenAQ](https://openaq.org/)

---

## Getting Help

### Project Issues
- GitHub Issues: https://github.com/research-jumpstart/research-jumpstart/issues
- Tag: `environmental-science`, `tier-2`, `aws`, `iot`

### AWS Support
- AWS Support Center: https://console.aws.amazon.com/support/
- AWS Forums: https://forums.aws.amazon.com/
- Stack Overflow: Tag `amazon-aws`, `boto3`, `aws-lambda`

### Environmental Science Help
- EPA Tech Support: https://www.epa.gov/technical-assistance
- Environmental Data Community: https://www.environmentaldata.org/

---

## Cost Tracking

### Monitor Your Spending

```bash
# Check current AWS charges
aws ce get-cost-and-usage \
  --time-period Start=2025-01-01,End=2025-01-31 \
  --granularity MONTHLY \
  --metrics "BlendedCost"

# Check specific service costs
aws ce get-cost-and-usage \
  --time-period Start=2025-01-01,End=2025-01-31 \
  --granularity DAILY \
  --metrics "UnblendedCost" \
  --group-by Type=DIMENSION,Key=SERVICE

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
| **Cost** | Free | $7-12 per run |
| **Storage** | 15GB persistent | Unlimited (pay per GB) |
| **Compute** | 4GB RAM, 12-hour sessions | Scalable (pay per second) |
| **Data Scale** | Limited to 15GB | Terabytes possible |
| **Real-time Processing** | Manual notebook execution | Event-driven Lambda |
| **Alerting** | Manual monitoring | Automated SNS notifications |
| **Persistence** | Session-based | Permanent S3 and DynamoDB |
| **Collaboration** | Limited | Full team access |
| **IoT Integration** | Not supported | Native IoT support |

---

## What's Next?

After completing this project:

**Skill Building**
- AWS Lambda advanced features (layers, versions, aliases)
- DynamoDB advanced queries (GSI, LSI, streams)
- SNS advanced routing (filter policies, mobile push)
- CloudWatch custom metrics and dashboards
- Security best practices (encryption, VPC, IAM)

**Project Extensions**
- Real-time environmental dashboard
- Predictive pollution modeling with machine learning
- Integration with IoT devices (Raspberry Pi, Arduino)
- Mobile app for field workers
- Compliance reporting (EPA, state agencies)

**Tier 3 Transition**
- Production infrastructure with CloudFormation
- High-availability multi-region deployment
- Advanced monitoring and alerting (CloudWatch, X-Ray)
- CI/CD pipeline for Lambda functions
- Cost optimization at scale

---

## Citation

If you use this project in your research:

```bibtex
@software{research_jumpstart_environmental_tier2,
  title = {Environmental Sensor Data Analysis with AWS: Tier 2},
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
