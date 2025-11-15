# Macroeconomic Forecasting with S3 and Lambda - Tier 2 Project

**Duration:** 2-4 hours | **Cost:** $6-11 | **Platform:** AWS + Local machine

Build a cloud-native macroeconomic forecasting pipeline using AWS serverless services. Upload economic time series data to S3, process with Lambda-based forecasting models, store predictions in DynamoDB, and analyze results with Athena—all without managing servers.

---

## What You'll Build

A serverless economic forecasting system that demonstrates:

1. **Data Storage** - Upload economic time series (GDP, unemployment, inflation) to S3
2. **Serverless Forecasting** - Lambda functions running ARIMA and exponential smoothing models
3. **Results Storage** - Store predictions in DynamoDB for fast queries
4. **Data Analysis** - Query forecasts with Athena and visualize in Jupyter notebooks

This bridges the gap between local analysis (Tier 1) and production infrastructure (Tier 3).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Your Local Machine                                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Python Scripts                                             │ │
│  │ • upload_to_s3.py - Upload economic time series          │ │
│  │ • lambda_function.py - Forecast with ARIMA/ES            │ │
│  │ • query_results.py - Analyze predictions                 │ │
│  │ • economic_analysis.ipynb - Jupyter notebook             │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  AWS Services                                                    │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  │  S3 Bucket       │  │ Lambda Function  │  │  DynamoDB        │
│  │                  │  │                  │  │                  │
│  │ • Economic data  │→ │ Forecasting:     │→ │ Store forecasts: │
│  │ • CSV files      │  │ - ARIMA models   │  │ - Predictions    │
│  │ • By indicator   │  │ - Exp smoothing  │  │ - Confidence     │
│  │                  │  │ - Seasonality    │  │ - Metadata       │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘
│                                                      ▲
│  ┌──────────────────────────────────────────────────┘
│  │  Athena (SQL Queries)
│  │  • Query predictions by indicator
│  │  • Compare forecast accuracy
│  │  • Time series analysis
│  └──────────────────────────────────────────────────────────────┐
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
- Basic Python (pandas, numpy, statsmodels)
- Understanding of economic time series data
- AWS fundamentals (S3, Lambda, IAM)

### Required Tools
- **Python 3.8+** with:
  - boto3 (AWS SDK)
  - pandas (data manipulation)
  - statsmodels (forecasting models)
  - numpy (numerical computing)
  - matplotlib/seaborn (visualization)

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
cd research-jumpstart/projects/economics/macro-forecasting/tier-2

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
# - S3 bucket: economic-data-{your-id}
# - IAM role: lambda-economic-forecaster
# - Lambda function: forecast-economic-indicators
# - DynamoDB table: EconomicForecasts
```

### Step 2: Upload Economic Data (2 minutes)
```bash
python scripts/upload_to_s3.py
```

### Step 3: Run Forecasting (2 minutes)
```bash
# Lambda is triggered automatically on S3 upload
# Or invoke manually for testing
```

### Step 4: Query Predictions (1 minute)
```bash
python scripts/query_results.py
```

### Step 5: Analyze Results (5 minutes)
Open `notebooks/economic_analysis.ipynb` in Jupyter and run all cells.

---

## Detailed Workflow

### 1. Data Preparation (Setup)

**What's happening:**
- Download economic time series from FRED or World Bank
- Format as CSV with standardized schema
- Upload to S3 organized by indicator and country

**Data sources:**
- FRED (Federal Reserve Economic Data)
- World Bank Open Data
- OECD Statistics
- Sample data included in project

**Files involved:**
- `setup_guide.md` - Step-by-step AWS setup
- `scripts/upload_to_s3.py` - Upload automation

**Time:** 10-15 minutes (includes data download)

**S3 Structure:**
```
s3://economic-data-{your-id}/
├── raw/                           # Original time series
│   ├── gdp/
│   │   ├── usa_gdp_quarterly.csv
│   │   ├── chn_gdp_quarterly.csv
│   │   └── ...
│   ├── unemployment/
│   │   ├── usa_unemployment_monthly.csv
│   │   └── ...
│   └── inflation/
│       ├── usa_cpi_monthly.csv
│       └── ...
```

### 2. Lambda Forecasting

**What's happening:**
- Lambda triggered when CSV uploaded to S3
- Reads time series data
- Fits forecasting models (ARIMA, Exponential Smoothing)
- Calculates trend, seasonality, forecast intervals
- Stores predictions in DynamoDB

**Forecasting methods:**
1. **ARIMA** - Auto-regressive Integrated Moving Average
2. **Exponential Smoothing** - Simple, Double, Triple (Holt-Winters)
3. **Seasonal Decomposition** - Trend and seasonal components
4. **Confidence Intervals** - 80% and 95% prediction intervals

**Lambda function**:
```python
# Event: S3 upload trigger
{
    'Records': [{
        'bucket': {'name': 'economic-data-xxxx'},
        'object': {'key': 'raw/gdp/usa_gdp_quarterly.csv'}
    }]
}

# Processing:
# 1. Read CSV from S3
# 2. Fit ARIMA model (auto-selected parameters)
# 3. Generate 8-quarter forecast
# 4. Calculate confidence intervals
# 5. Store in DynamoDB

# Output: Forecasts stored in DynamoDB table
```

**Files involved:**
- `scripts/lambda_function.py` - Forecasting function code
- `setup_guide.md` - Lambda deployment steps

**Time:** 30-60 seconds per indicator

### 3. Results Storage

**What's happening:**
- Forecasts stored in DynamoDB with metadata
- Queryable by indicator, country, date range
- Includes model parameters and accuracy metrics

**DynamoDB Schema:**
```
EconomicForecasts Table:
  Primary Key: indicator_country (e.g., "GDP_USA")
  Sort Key: forecast_date (timestamp)

  Attributes:
    - indicator: "GDP", "unemployment", "inflation"
    - country: "USA", "CHN", "DEU", etc.
    - forecast_date: ISO 8601 timestamp
    - forecast_value: predicted value
    - confidence_80_lower: 80% CI lower bound
    - confidence_80_upper: 80% CI upper bound
    - confidence_95_lower: 95% CI lower bound
    - confidence_95_upper: 95% CI upper bound
    - model_type: "ARIMA", "ExponentialSmoothing"
    - model_params: JSON of model parameters
    - rmse: root mean squared error (if validation data)
    - mae: mean absolute error (if validation data)
    - processing_time_ms: Lambda execution time
```

### 4. Results Analysis

**What's happening:**
- Query DynamoDB for forecasts
- Download to pandas DataFrames
- Visualize predictions with confidence intervals
- Compare multiple forecasting methods
- Analyze forecast accuracy (if actual data available)

**Analysis types:**
1. **Point Forecasts** - Single best estimate
2. **Interval Forecasts** - Uncertainty quantification
3. **Model Comparison** - ARIMA vs Exponential Smoothing
4. **Accuracy Metrics** - RMSE, MAE, MAPE
5. **Policy Scenarios** - What-if analysis

**Files involved:**
- `notebooks/economic_analysis.ipynb` - Analysis notebook
- `scripts/query_results.py` - Query and download
- (Optional) Athena queries for complex SQL analysis

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
│   └── economic_analysis.ipynb       # Main analysis notebook
│
├── scripts/
│   ├── upload_to_s3.py               # Upload time series to S3
│   ├── lambda_function.py            # Lambda forecasting function
│   ├── query_results.py              # Query DynamoDB results
│   └── __init__.py
│
└── sample_data/
    ├── usa_gdp_quarterly.csv         # Sample GDP data
    ├── usa_unemployment_monthly.csv  # Sample unemployment data
    └── README.md                     # Data documentation
```

---

## Cost Breakdown

**Total estimated cost: $6-11 per run**

| Service | Usage | Cost |
|---------|-------|------|
| **S3 Storage** | 100 MB × 7 days | $0.02 |
| **S3 Requests** | ~500 PUT/GET requests | $0.05 |
| **Lambda Executions** | 50 invocations × 30 sec | $0.50 |
| **Lambda Compute** | 50 GB-seconds (512MB memory) | $0.80 |
| **DynamoDB Writes** | 500 write requests | $0.65 |
| **DynamoDB Storage** | 10 MB for 7 days | $0.03 |
| **Athena Queries** | 5 queries × 1GB scanned | $0.25 |
| **Data Transfer** | Upload + download (100MB) | $0.10 |
| **CloudWatch Logs** | 50 MB logs | $0.05 |
| **Total** | | **$2.45** |

**With safety margin and multiple runs: $6-11**

**Cost optimization tips:**
1. Delete DynamoDB items after analysis ($0.65 savings)
2. Use S3 lifecycle policies (auto-delete after 7 days)
3. Batch Lambda invocations (reduce request costs)
4. Use Athena sparingly (query DynamoDB directly)
5. Clean up CloudWatch logs regularly

**Free Tier Usage:**
- **S3**: First 5GB storage free (12 months)
- **Lambda**: 1M invocations free per month (12 months)
- **Lambda**: 400,000 GB-seconds free per month (12 months)
- **DynamoDB**: 25 GB storage + 25 WCU/RCU free (always)
- **Athena**: First 10GB scanned free per month

---

## Key Learning Objectives

### AWS Services
- S3 bucket creation and lifecycle management
- Lambda function deployment with external libraries
- IAM role creation with least privilege
- DynamoDB table design and queries
- CloudWatch monitoring and logs
- (Optional) Athena for serverless SQL queries

### Cloud Concepts
- Object storage for time series data
- Serverless computing (no servers to manage)
- Event-driven architecture (S3 triggers)
- NoSQL database design
- Cost-conscious design patterns

### Economic Forecasting Skills
- ARIMA model selection and fitting
- Exponential smoothing methods
- Seasonal decomposition
- Forecast interval calculation
- Model comparison and validation
- Policy scenario analysis

---

## Time Estimates

**First Time Setup:**
- Read setup_guide.md: 10 minutes
- Create S3 bucket: 2 minutes
- Create DynamoDB table: 3 minutes
- Create IAM role: 5 minutes
- Deploy Lambda: 10 minutes
- **Subtotal setup: 30 minutes**

**Data Processing:**
- Download sample data: 5 minutes
- Upload to S3: 2 minutes
- Lambda forecasting: 5-10 minutes (50 indicators)
- **Subtotal processing: 12-17 minutes**

**Analysis:**
- Query results: 3 minutes
- Jupyter analysis: 30-45 minutes
- Generate visualizations: 10-15 minutes
- **Subtotal analysis: 43-63 minutes**

**Total time: 1.5-2 hours** (including setup)

---

## Running the Project

### Option 1: Automated (Recommended for First Time)
```bash
# Step 1: Setup AWS services (follow prompts)
# See setup_guide.md for detailed instructions

# Step 2: Upload economic data
python scripts/upload_to_s3.py \
  --bucket economic-data-{your-id} \
  --data-dir sample_data/

# Step 3: Lambda is triggered automatically
# Monitor progress in CloudWatch Logs

# Step 4: Query results
python scripts/query_results.py \
  --table EconomicForecasts \
  --indicator GDP \
  --country USA

# Step 5: Analyze in notebook
jupyter notebook notebooks/economic_analysis.ipynb
```

### Option 2: Manual (Detailed Control)
```bash
# 1. Create S3 bucket
aws s3 mb s3://economic-data-$(date +%s) --region us-east-1

# 2. Upload data
aws s3 cp sample_data/ s3://economic-data-xxxx/raw/ --recursive

# 3. Deploy Lambda (see setup_guide.md for packaging)
# Package with statsmodels, numpy, pandas dependencies

# 4. Test Lambda manually
aws lambda invoke \
  --function-name forecast-economic-indicators \
  --payload '{"bucket":"economic-data-xxxx","key":"raw/gdp/usa_gdp_quarterly.csv"}' \
  response.json

# 5. Query DynamoDB
aws dynamodb scan \
  --table-name EconomicForecasts \
  --filter-expression "indicator = :ind" \
  --expression-attribute-values '{":ind":{"S":"GDP"}}'

# 6. Analyze results
jupyter notebook notebooks/economic_analysis.ipynb
```

---

## Jupyter Notebook Workflow

The main analysis notebook includes:

1. **Data Loading**
   - Query DynamoDB for forecasts
   - Load into pandas DataFrames
   - Compare with actual values (if available)

2. **Forecast Visualization**
   - Time series plots with confidence intervals
   - Multiple indicators on same chart
   - Country comparisons

3. **Model Comparison**
   - ARIMA vs Exponential Smoothing
   - Accuracy metrics (RMSE, MAE, MAPE)
   - Parameter sensitivity analysis

4. **Economic Analysis**
   - GDP growth projections
   - Unemployment trends
   - Inflation forecasts
   - Policy scenario analysis

5. **Export Results**
   - Save figures (publication-quality)
   - Generate forecast reports
   - Export to CSV for external tools

---

## What You'll Discover

### Economic Insights
- GDP growth projections for major economies
- Unemployment rate trends and seasonality
- Inflation forecasts with uncertainty
- Model performance across different indicators
- Leading vs lagging economic indicators

### AWS Insights
- Serverless architecture advantages
- Cost-effective cloud forecasting
- Scalability: forecast 1 or 1000 indicators
- Real-time updates as new data arrives
- Collaborative analysis (shared results)

### Research Insights
- Reproducibility: Same code, same forecasts
- Automation: New data triggers new forecasts
- Persistence: Historical forecasts stored permanently
- Comparison: Multiple models, multiple countries
- Policy tools: Scenario analysis and what-if modeling

---

## Economic Indicators Supported

### Macroeconomic Variables
1. **GDP** - Gross Domestic Product (quarterly)
2. **Unemployment Rate** - Labor market indicator (monthly)
3. **Inflation (CPI)** - Consumer Price Index (monthly)
4. **Interest Rates** - Federal Funds Rate (monthly)
5. **Industrial Production** - Manufacturing output (monthly)
6. **Retail Sales** - Consumer spending (monthly)
7. **Housing Starts** - Construction activity (monthly)
8. **Trade Balance** - Exports minus imports (monthly)

### Geographic Coverage
- United States (USA)
- China (CHN)
- Germany (DEU)
- Japan (JPN)
- United Kingdom (GBR)
- Eurozone aggregate (EU)
- Custom: Add any country from FRED or World Bank

---

## Next Steps

### Extend This Project
1. **More Indicators**: Add commodity prices, stock indices, exchange rates
2. **More Countries**: Compare forecasts across G20 nations
3. **More Models**: Add Prophet, LSTM, VAR models
4. **Real-time Data**: Integrate FRED API for automatic updates
5. **Alerts**: SNS notifications when forecasts diverge from actuals
6. **Dashboard**: QuickSight or Grafana visualization

### Move to Tier 3 (Production)
- CloudFormation for infrastructure-as-code
- Production-grade Lambda with error handling
- Multi-region deployment for global access
- Advanced monitoring and alerting
- Cost optimization with reserved capacity
- API Gateway for external access

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
s3://economic-data-$(date +%s)-yourname
```

**"Lambda timeout - statsmodels taking too long"**
```python
# Solution 1: Increase timeout in Lambda console
# Recommended: 300 seconds (5 minutes)

# Solution 2: Reduce data size (use last 5 years only)
# In lambda_function.py:
data = data[-20:]  # Last 20 quarters for GDP

# Solution 3: Increase Lambda memory (speeds up CPU)
# Recommended: 512 MB or 1024 MB
```

**"ImportError: No module named statsmodels"**
```bash
# Solution: Package statsmodels with Lambda deployment
# See setup_guide.md Section 4.3 for Lambda Layer creation

# Quick fix: Create Lambda layer with:
pip install statsmodels -t python/lib/python3.9/site-packages/
zip -r statsmodels-layer.zip python/
aws lambda publish-layer-version \
  --layer-name statsmodels \
  --zip-file fileb://statsmodels-layer.zip \
  --compatible-runtimes python3.9
```

**"DynamoDB ConditionalCheckFailedException"**
```python
# Solution: Table key schema mismatch
# Ensure indicator_country format: "GDP_USA" not "GDP-USA"
# Check scripts/lambda_function.py line 145
```

**"Athena query returns no results"**
```sql
-- Solution: DynamoDB table not registered in Athena
-- Create external table first (see setup_guide.md Step 6)
-- Or query DynamoDB directly with boto3
```

See `cleanup_guide.md` for resource deletion instructions.

---

## Cleanup Guide

When finished, delete AWS resources to stop incurring charges:

```bash
# Quick cleanup script
python scripts/cleanup.py

# Or manual cleanup:

# Delete DynamoDB table
aws dynamodb delete-table --table-name EconomicForecasts

# Delete S3 bucket and contents
aws s3 rm s3://economic-data-xxxx --recursive
aws s3 rb s3://economic-data-xxxx

# Delete Lambda function
aws lambda delete-function --function-name forecast-economic-indicators

# Delete Lambda layer (if created)
aws lambda delete-layer-version \
  --layer-name statsmodels \
  --version-number 1

# Delete IAM role
aws iam delete-role-policy \
  --role-name lambda-economic-forecaster \
  --policy-name economic-forecasting-policy
aws iam delete-role --role-name lambda-economic-forecaster
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

### Economic Data Sources
- [FRED API](https://fred.stlouisfed.org/docs/api/fred/)
- [World Bank Open Data](https://data.worldbank.org/)
- [OECD Statistics](https://stats.oecd.org/)
- [IMF Data](https://www.imf.org/en/Data)

### Python Libraries
- [boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- [statsmodels Documentation](https://www.statsmodels.org/stable/index.html)
- [pandas Documentation](https://pandas.pydata.org/docs/)
- [ARIMA Guide](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)

### Forecasting Methods
- [Exponential Smoothing](https://otexts.com/fpp2/expsmooth.html)
- [ARIMA Models](https://otexts.com/fpp2/arima.html)
- [Forecasting: Principles and Practice](https://otexts.com/fpp2/)

---

## Getting Help

### Project Issues
- GitHub Issues: https://github.com/research-jumpstart/research-jumpstart/issues
- Tag: `economics`, `tier-2`, `aws`, `forecasting`

### AWS Support
- AWS Support Center: https://console.aws.amazon.com/support/
- AWS Forums: https://forums.aws.amazon.com/
- Stack Overflow: Tag `amazon-aws`, `boto3`, `aws-lambda`

### Economic Data Help
- FRED Support: https://fred.stlouisfed.org/docs/api/fred/
- World Bank Help: https://datahelpdesk.worldbank.org/

---

## Cost Tracking

### Monitor Your Spending

```bash
# Check current AWS charges
aws ce get-cost-and-usage \
  --time-period Start=2025-11-01,End=2025-11-30 \
  --granularity MONTHLY \
  --metrics "BlendedCost" \
  --group-by Type=SERVICE

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
| **Cost** | Free | $6-11 per run |
| **Storage** | 15GB persistent | Unlimited (pay per GB) |
| **Compute** | 4GB RAM, 12-hour sessions | Scalable Lambda (pay per second) |
| **Data Scale** | Limited to 15GB | Gigabytes to terabytes |
| **Automation** | Manual notebook execution | Event-driven (S3 triggers) |
| **Persistence** | Session-based | Permanent S3/DynamoDB storage |
| **Collaboration** | Limited | Full team access to forecasts |
| **Real-time** | Manual updates | Automatic reforecasting on new data |

---

## What's Next?

After completing this project:

**Skill Building**
- Advanced time series models (VAR, VARMA)
- Machine learning forecasting (LSTM, Prophet)
- Ensemble forecasting (combine multiple models)
- Probabilistic forecasting (full distributions)

**Project Extensions**
- Real-time data ingestion from FRED API
- Automated weekly reforecasting pipeline
- Email/SMS alerts for significant changes
- Dashboard with historical forecast performance
- Scenario analysis tools (policy simulations)

**Tier 3 Transition**
- Production infrastructure with CloudFormation
- High-availability architecture
- Advanced monitoring and alerting
- Multi-region deployment
- API for external forecast access
- Integration with visualization tools

---

## Citation

If you use this project in your research:

```bibtex
@software{research_jumpstart_economics_tier2,
  title = {Macroeconomic Forecasting with S3 and Lambda: Tier 2},
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
