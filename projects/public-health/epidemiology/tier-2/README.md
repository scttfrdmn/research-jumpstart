# Disease Surveillance and Outbreak Detection with AWS - Tier 2

**Duration:** 2-4 hours | **Cost:** $6-10 | **Platform:** AWS + Local machine

Build a cloud-based epidemiological surveillance system using AWS services. Upload disease case data to S3, analyze with Lambda for outbreak detection, store results in DynamoDB, and query with Athena for real-time epidemiological insights.

---

## What You'll Build

A complete disease surveillance and outbreak detection pipeline that:

1. **Data Ingestion** - Upload case report data (CSV) to S3
2. **Automated Analysis** - Lambda processes epidemiological data on upload
3. **Outbreak Detection** - Calculate incidence rates, detect unusual patterns, estimate R0
4. **Alert System** - SNS notifications for potential outbreak signals
5. **Data Storage** - DynamoDB for case reports and aggregated metrics
6. **Analytics** - Athena queries for spatial-temporal disease patterns

This bridges the gap between local analysis (Tier 1) and production infrastructure (Tier 3).

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  Your Local Machine                                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Python Scripts                                             │ │
│  │ • upload_to_s3.py - Upload case reports to S3            │ │
│  │ • lambda_function.py - Epidemiological analysis           │ │
│  │ • query_results.py - Query surveillance data              │ │
│  │ • epidemiology_analysis.ipynb - Jupyter notebook          │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  AWS Services                                                    │
│                                                                   │
│  ┌──────────────────┐                                            │
│  │  S3 Bucket       │  Upload Trigger                           │
│  │  Case Reports    │────────────┐                              │
│  │  - Raw CSV data  │            │                              │
│  │  - Anonymized    │            ▼                              │
│  └──────────────────┘     ┌──────────────────────┐              │
│                           │  Lambda Function      │              │
│                           │  Epi Analysis         │              │
│  ┌──────────────────┐     │                       │              │
│  │  SNS Topic       │◄────│ • Incidence rates    │              │
│  │  Outbreak Alerts │     │ • Prevalence         │              │
│  │  - Email/SMS     │     │ • Attack rates       │              │
│  └──────────────────┘     │ • R0 estimation      │              │
│                           │ • Outbreak signals   │              │
│                           │ • Epidemic curves    │              │
│  ┌──────────────────┐     └──────────┬───────────┘              │
│  │  DynamoDB        │                │                          │
│  │  DiseaseReports  │◄───────────────┘                          │
│  │  - Case data     │                                            │
│  │  - Metrics       │                                            │
│  │  - Timestamps    │                                            │
│  └──────┬───────────┘                                            │
│         │                                                         │
│         ▼                                                         │
│  ┌──────────────────┐                                            │
│  │  Athena          │                                            │
│  │  SQL Queries     │                                            │
│  │  - Spatial       │                                            │
│  │  - Temporal      │                                            │
│  │  - Aggregations  │                                            │
│  └──────────────────┘                                            │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────────┐
│  │  IAM Role (Permissions)                                      │
│  │  • S3 read/write                                             │
│  │  • DynamoDB read/write                                       │
│  │  • SNS publish                                               │
│  │  • CloudWatch logging                                        │
│  └──────────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────────┘
            │
            ▼
     ┌─────────────────┐
     │  Jupyter         │
     │  Notebook        │
     │  - SIR models    │
     │  - Epidemic      │
     │    curves        │
     │  - Heatmaps      │
     │  - R0 plots      │
     └─────────────────┘
```

---

## Project Overview

This Tier 2 project demonstrates how to build a cloud-based epidemiological surveillance system using AWS services. You'll process disease case data, calculate key epidemiological metrics, detect outbreak signals, and visualize disease patterns—all using serverless AWS infrastructure.

**Key Learning:** Move from local surveillance analysis (Tier 1) to scalable cloud-based outbreak detection (Tier 2)

### What's Included

- **Epidemiological Metrics**: Incidence rates, prevalence, case fatality rates, attack rates
- **Outbreak Detection**: Statistical anomaly detection, clustering analysis, temporal patterns
- **Disease Modeling**: SIR model simulation, R0 (basic reproductive number) estimation
- **Spatial Analysis**: Geographic disease distribution, hotspot detection
- **Real-time Alerts**: SNS notifications when outbreak signals detected
- **Data Privacy**: Automated anonymization of sensitive case information

---

## Prerequisites

### Required

- AWS Account (free tier eligible)
- Python 3.8+
- Jupyter Notebook or JupyterLab
- AWS CLI configured with credentials
- boto3 library

### Knowledge

- Basic Python programming
- Understanding of epidemiological concepts (helpful but not required)
- Familiarity with AWS services (S3, Lambda)
- Basic statistics knowledge

### Installation

```bash
# Clone the project
cd research-jumpstart/projects/public-health/epidemiology/tier-2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## AWS Services Used

| Service | Purpose | Cost |
|---------|---------|------|
| **S3** | Store case report data (CSV) | $0.023 per GB/month (~$0.05-0.15) |
| **Lambda** | Serverless epidemiological analysis | $0.20 per 1M invocations (~$0.10-0.20) |
| **DynamoDB** | NoSQL storage for case data and metrics | $0.25/GB for on-demand (~$0.20-0.50) |
| **SNS** | Outbreak alert notifications | $0.50 per 1M notifications (~$0.01-0.05) |
| **Athena** | SQL queries on surveillance data | $5 per TB scanned (~$0.05-0.20) |
| **IAM** | Access control and permissions | Free |
| **CloudWatch** | Logging and monitoring | Free (basic tier) |

**Total Estimated Cost: $6-10 per run**

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure AWS

Follow the detailed setup guide:
```bash
cat setup_guide.md
```

Quick steps:
- Create S3 bucket: `epidemiology-data-{your-user-id}`
- Create IAM role for Lambda with S3, DynamoDB, and SNS permissions
- Create DynamoDB table: `DiseaseReports`
- Create SNS topic: `outbreak-alerts`
- Deploy Lambda function: `analyze-disease-surveillance`

### 3. Run the Pipeline

```bash
# Upload sample case data
python scripts/upload_to_s3.py --input-file sample_data/case_reports.csv \
                                --s3-bucket epidemiology-data-{user-id}

# Lambda automatically processes on upload
# Check CloudWatch logs for processing status

# Query results
python scripts/query_results.py --table-name DiseaseReports \
                                 --disease influenza \
                                 --start-date 2024-01-01

# Analyze in notebook
jupyter notebook notebooks/epidemiology_analysis.ipynb
```

---

## Project Structure

```
tier-2/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup_guide.md                     # Step-by-step AWS setup
├── cleanup_guide.md                   # How to delete resources
├── notebooks/
│   └── epidemiology_analysis.ipynb   # Main analysis notebook
├── scripts/
│   ├── upload_to_s3.py               # Upload case data to S3
│   ├── lambda_function.py            # Lambda epidemiological analysis
│   └── query_results.py              # Retrieve and analyze results
└── sample_data/
    └── case_reports.csv              # Sample disease case data
```

---

## Workflow Steps

### Step 1: Setup AWS Environment

- Create S3 bucket for case data
- Create IAM role for Lambda with necessary permissions
- Create DynamoDB table for surveillance data
- Create SNS topic for outbreak alerts
- (Optional) Configure Athena for SQL queries
- (Detailed instructions in `setup_guide.md`)

### Step 2: Prepare Case Data

Generate or use provided sample case reports with fields:
- Case ID (anonymized)
- Disease type
- Report date
- Location (region, zip code)
- Demographics (age group, sex)
- Outcome (recovered, hospitalized, fatal)
- Symptom onset date

### Step 3: Upload Data to S3

```bash
python scripts/upload_to_s3.py \
    --input-file sample_data/case_reports.csv \
    --s3-bucket epidemiology-data-{user-id} \
    --prefix case-data/ \
    --anonymize
```

Features:
- Data validation (required fields, date formats)
- Automatic anonymization of sensitive information
- Progress tracking for large datasets
- Error handling and logging

### Step 4: Automated Lambda Processing

Lambda function automatically triggers on S3 upload and:

1. **Data Validation**
   - Check data quality and completeness
   - Validate date ranges and geographic codes

2. **Epidemiological Metrics**
   - Incidence rate: new cases per population per time period
   - Prevalence: total cases per population at a point in time
   - Case fatality rate: deaths / total cases
   - Attack rate: cases / population at risk

3. **Outbreak Detection**
   - Moving average analysis
   - Standard deviation thresholds (2σ, 3σ)
   - Seasonal baseline comparison
   - Geographic clustering (spatial scan statistics)

4. **Disease Modeling**
   - Basic reproductive number (R0) estimation
   - Epidemic curve generation
   - Doubling time calculation
   - Growth rate analysis

5. **Alert Generation**
   - Send SNS notification if outbreak signal detected
   - Include: disease, location, severity, recommended actions

6. **Data Storage**
   - Store processed results in DynamoDB
   - Log metadata: processing time, case counts, alerts

### Step 5: Query Results

```bash
# Query by disease type
python scripts/query_results.py --disease influenza --limit 100

# Query by region
python scripts/query_results.py --region northeast --start-date 2024-01-01

# Query by time period
python scripts/query_results.py --start-date 2024-01-01 --end-date 2024-01-31

# Export to CSV
python scripts/query_results.py --disease influenza --output results.csv
```

### Step 6: Advanced Analytics with Athena (Optional)

```sql
-- Setup Athena external table (see setup_guide.md)
CREATE EXTERNAL TABLE disease_surveillance (
    case_id STRING,
    disease STRING,
    report_date DATE,
    region STRING,
    age_group STRING,
    outcome STRING
)
STORED AS PARQUET
LOCATION 's3://epidemiology-data-{user-id}/case-data/';

-- Query: Cases by region and disease
SELECT region, disease, COUNT(*) as case_count
FROM disease_surveillance
WHERE report_date >= DATE '2024-01-01'
GROUP BY region, disease
ORDER BY case_count DESC;

-- Query: Weekly incidence
SELECT
    DATE_TRUNC('week', report_date) as week,
    disease,
    COUNT(*) as cases
FROM disease_surveillance
GROUP BY DATE_TRUNC('week', report_date), disease
ORDER BY week, disease;
```

### Step 7: Visualization and Analysis

Open `notebooks/epidemiology_analysis.ipynb` to:

1. **Generate Sample Data**
   - Synthetic case reports with realistic patterns
   - Inject outbreak signals for testing

2. **Upload and Process**
   - Upload to S3
   - Monitor Lambda processing
   - Check for alert notifications

3. **Query and Analyze**
   - Download results from DynamoDB
   - Calculate epidemiological metrics
   - Compare with baseline expectations

4. **Visualizations**
   - **Epidemic curves**: Cases over time
   - **Geographic heatmaps**: Disease distribution by region
   - **SIR model plots**: Susceptible-Infected-Recovered dynamics
   - **R0 estimation**: Reproductive number over time
   - **Attack rate maps**: Disease impact by demographic groups

---

## Expected Results

After completing this project, you will have:

### 1. Cloud-Based Surveillance System
- Automated case data processing
- Real-time outbreak detection
- Scalable to millions of case reports
- Geographic and temporal analysis

### 2. Epidemiological Analytics
- Incidence and prevalence tracking
- Case fatality rate monitoring
- R0 estimation for outbreak assessment
- Epidemic curve generation

### 3. Outbreak Detection Capabilities
- Statistical anomaly detection
- Geographic clustering analysis
- Temporal pattern recognition
- Automated alert notifications

### 4. Data Management Skills
- Privacy-preserving data handling
- NoSQL database design for surveillance
- S3 data organization
- ETL pipeline development

### 5. Cost-Effective Infrastructure
- Pay-per-use serverless processing
- Minimal operational overhead
- Clear cost tracking
- Foundation for production scaling (Tier 3)

---

## Cost Breakdown

### Detailed Cost Estimate

**Assumptions:**
- 10,000 case reports (CSV, ~5 MB)
- Lambda: 100 invocations, 15 seconds each
- DynamoDB: 10,000 writes, 1,000 reads
- SNS: 5 outbreak alerts
- Athena: 10 queries scanning 1 GB total
- Run duration: 1 week

**Costs:**

| Service | Usage | Cost |
|---------|-------|------|
| S3 Storage | 5 MB for 1 week | $0.00001 per GB-hour = $0.01 |
| S3 Requests | 100 PUT, 200 GET | $0.005 per 1000 = $0.002 |
| Lambda Compute | 100 × 15s @ 256MB | ~$0.15 |
| Lambda Requests | 100 requests | ~$0.00002 |
| DynamoDB Writes | 10,000 on-demand | $1.25 per 1M = $0.01 |
| DynamoDB Reads | 1,000 on-demand | $0.25 per 1M = $0.0003 |
| SNS Notifications | 5 email alerts | $0.50 per 1M = ~$0.00 |
| Athena Queries | 1 GB scanned | $5 per TB = $0.005 |
| Data Transfer | ~10 MB out | ~$0.00 |
| **Total** | | **$0.43** |

**Note:** This is a minimal learning run. Typical costs $6-10 include:
- Multiple test runs (5-10 iterations)
- Larger datasets (100,000+ cases)
- More frequent queries
- Development and debugging time
- Safety margin for experiments

**Cost Optimization Tips:**
1. Delete S3 objects after analysis
2. Use on-demand DynamoDB pricing (auto-scales)
3. Batch Lambda invocations when possible
4. Limit Athena query scans with partitioning
5. Clean up test data regularly (see cleanup_guide.md)

**Free Tier Benefits (First 12 Months):**
- Lambda: 1M invocations/month free
- DynamoDB: 25 GB storage, 25 WCU, 25 RCU free
- S3: 5 GB storage, 20,000 GET, 2,000 PUT free
- SNS: 1,000 notifications free

---

## Learning Objectives

### Technical Skills
- [x] Create and configure AWS services (S3, Lambda, DynamoDB, SNS)
- [x] Write boto3 code for AWS automation
- [x] Deploy Lambda functions with triggers
- [x] Design NoSQL schemas for surveillance data
- [x] Query DynamoDB with boto3
- [x] Set up SNS notifications
- [x] Use Athena for SQL analytics
- [x] Monitor costs and optimize spending

### Epidemiological Concepts
- [x] Calculate incidence, prevalence, and attack rates
- [x] Estimate basic reproductive number (R0)
- [x] Generate epidemic curves
- [x] Detect outbreak signals with statistical methods
- [x] Apply SIR disease models
- [x] Analyze spatial disease patterns
- [x] Interpret case fatality rates

### Cloud Architecture
- [x] Event-driven serverless architecture
- [x] Data pipeline design
- [x] NoSQL vs SQL tradeoffs
- [x] Real-time alert systems
- [x] Privacy-preserving data processing
- [x] Scalable surveillance infrastructure

---

## Troubleshooting

### Common Issues

**Problem:** "NoCredentialsError" when running Python scripts
- **Solution:** Configure AWS CLI with `aws configure` and provide access keys
- Verify credentials: `aws sts get-caller-identity`
- See setup_guide.md Step 1 for detailed instructions

**Problem:** Lambda function timeout during processing
- **Solution:** Increase timeout in Lambda configuration to 60 seconds
- For large datasets, consider batch processing
- Check CloudWatch logs for specific bottlenecks

**Problem:** "AccessDenied" errors with S3 or DynamoDB
- **Solution:** Check IAM role permissions in setup_guide.md
- Ensure Lambda execution role has:
  - `s3:GetObject`, `s3:PutObject` on S3 bucket
  - `dynamodb:PutItem`, `dynamodb:Query` on DynamoDB table
  - `sns:Publish` on SNS topic

**Problem:** SNS alerts not received
- **Solution:** Verify SNS subscription confirmed via email
- Check Lambda CloudWatch logs for SNS publish errors
- Ensure SNS topic ARN correctly set in Lambda environment

**Problem:** High DynamoDB costs
- **Solution:** Use on-demand billing instead of provisioned
- Review query patterns to minimize scans
- Delete old test data regularly

**Problem:** Athena query errors
- **Solution:** Ensure data format matches table schema
- Use partitioning for large datasets
- Limit scans with WHERE clauses on partitioned columns

**Problem:** Invalid case data format
- **Solution:** Use provided CSV template in sample_data/
- Validate with `upload_to_s3.py --validate-only` before uploading
- Check CloudWatch logs for specific validation errors

See `cleanup_guide.md` for information on removing test resources.

---

## Epidemiological Metrics Explained

### Incidence Rate
Number of new cases over a time period divided by population at risk.
- **Formula**: (New cases / Population) × 100,000
- **Use**: Measure disease burden, track trends

### Prevalence
Total cases (existing + new) at a point in time divided by total population.
- **Formula**: (Total cases / Population) × 100
- **Use**: Understand disease burden at a specific time

### Case Fatality Rate (CFR)
Proportion of cases that result in death.
- **Formula**: (Deaths / Total cases) × 100
- **Use**: Measure disease severity

### Attack Rate
Proportion of population that becomes ill during an outbreak.
- **Formula**: (Cases / Population at risk) × 100
- **Use**: Assess outbreak scope

### Basic Reproductive Number (R0)
Average number of secondary infections from one infected individual.
- **R0 < 1**: Outbreak will die out
- **R0 = 1**: Endemic steady state
- **R0 > 1**: Epidemic growth
- **Use**: Predict outbreak trajectory, evaluate interventions

### Epidemic Curve
Histogram of cases by date of symptom onset.
- **Shape indicates**: Point source, continuous common source, or propagated outbreak
- **Use**: Identify outbreak pattern and timing

---

## Next Steps

After completing this Tier 2 project:

### Option 1: Advanced Tier 2 Features
- Add CloudWatch dashboard for real-time monitoring
- Implement SQS queue for handling data spikes
- Add more sophisticated outbreak detection (machine learning)
- Integrate external data sources (weather, mobility)
- Implement contact tracing analysis
- Add forecasting models (ARIMA, Prophet)

### Option 2: Move to Tier 3 (Production)
Tier 3 uses CloudFormation for automated infrastructure:
- Infrastructure-as-code templates
- One-click deployment
- Multi-region availability
- Production-ready security policies
- Auto-scaling and high availability
- Advanced monitoring and alerting
- Integration with public health APIs

See `/projects/public-health/epidemiology/tier-3/` for production deployment

### Option 3: Real-World Applications
- Integrate with CDC or WHO data feeds
- Connect to electronic health record systems
- Build public-facing surveillance dashboard
- Implement syndromic surveillance
- Add phylogenetic analysis for genomic epidemiology
- Deploy mobile data collection apps

---

## Key Differences from Tier 1

| Aspect | Tier 1 (Studio Lab) | Tier 2 (AWS) |
|--------|-------------------|------------|
| **Platform** | Free SageMaker Studio Lab | AWS account required |
| **Cost** | Free | $6-10 per run |
| **Storage** | 15GB temporary | Unlimited S3 (pay per GB) |
| **Compute** | 4GB RAM, 12-hour sessions | Scalable Lambda functions |
| **Data Scale** | Limited to 15GB | Millions of case reports |
| **Automation** | Manual notebook runs | Event-driven processing |
| **Alerts** | Manual monitoring | Automated SNS notifications |
| **Persistence** | Session-based | Permanent DynamoDB storage |
| **Collaboration** | Limited | Full team access, APIs |

---

## Public Health Context

### Disease Surveillance
Public health surveillance is the ongoing, systematic collection, analysis, and interpretation of health data. This project demonstrates key surveillance activities:

1. **Case-based surveillance**: Individual case reports
2. **Syndromic surveillance**: Early detection using symptom patterns
3. **Outbreak detection**: Statistical methods for anomaly detection
4. **Epidemic investigation**: Analysis of outbreak characteristics

### Applications
- Seasonal influenza monitoring
- COVID-19 case tracking
- Foodborne illness outbreak detection
- Vaccine-preventable disease surveillance
- Healthcare-associated infection monitoring
- Emerging infectious disease early warning

### Data Privacy
This project emphasizes privacy-preserving approaches:
- Automatic anonymization of identifiers
- Geographic aggregation to prevent re-identification
- Secure cloud storage with encryption
- Access controls via IAM roles
- HIPAA considerations for real-world applications

---

## References

### AWS Documentation
- [S3 Documentation](https://docs.aws.amazon.com/s3/)
- [Lambda Documentation](https://docs.aws.amazon.com/lambda/)
- [DynamoDB Documentation](https://docs.aws.amazon.com/dynamodb/)
- [SNS Documentation](https://docs.aws.amazon.com/sns/)
- [Athena Documentation](https://docs.aws.amazon.com/athena/)
- [boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)

### Epidemiology Resources
- [CDC Principles of Epidemiology](https://www.cdc.gov/csels/dsepd/ss1978/)
- [WHO Disease Surveillance](https://www.who.int/teams/surveillance-prevention-control)
- [Epidemiologic Methods for Outbreak Investigation](https://www.cdc.gov/eis/field-epi-manual/)
- [R0 Estimation Methods](https://doi.org/10.1371/journal.pcbi.1003908)

### Python Libraries
- [pandas Documentation](https://pandas.pydata.org/docs/)
- [numpy Documentation](https://numpy.org/doc/)
- [scipy Statistics](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [seaborn Gallery](https://seaborn.pydata.org/examples/index.html)

### Public Health Data Sources
- [CDC National Notifiable Diseases Surveillance System](https://www.cdc.gov/nndss/)
- [WHO Global Outbreak Alert and Response](https://www.who.int/emergencies/outbreak-toolkit)
- [ECDC Surveillance Atlas](https://atlas.ecdc.europa.eu/)
- [HealthMap Real-time Disease Surveillance](https://www.healthmap.org/)

---

## Support

### Getting Help
1. Check troubleshooting section above
2. Review AWS service error messages in CloudWatch logs
3. Consult boto3 documentation for API details
4. Check AWS service quotas (may need increase for scale)

### For Issues
- Review setup_guide.md for configuration problems
- Check IAM permissions if access errors occur
- Verify S3 bucket naming (globally unique)
- Confirm DynamoDB table exists before running scripts
- Test SNS subscriptions are confirmed

---

## Citation

If you use this project in your research:

```bibtex
@software{research_jumpstart_epidemiology_tier2,
  title = {Disease Surveillance and Outbreak Detection with AWS: Tier 2},
  author = {Research Jumpstart Community},
  year = {2025},
  url = {https://github.com/research-jumpstart/research-jumpstart},
  note = {Accessed: [date]}
}
```

---

## License

This project is part of the Research Jumpstart curriculum and is provided for educational purposes.

Apache License 2.0 - See [LICENSE](../../../../LICENSE) for details.

---

## Author Notes

This is a Tier 2 (AWS Starter) project. It bridges Tier 1 (Studio Lab free tier) and Tier 3 (Production CloudFormation).

**Time to Complete:** 2-4 hours
**Cost:** $6-10 per run
**Difficulty:** Intermediate (requires AWS account setup)

For questions about the project structure, see `TIER_2_SPECIFICATIONS.md` in the project root.

**Public Health Note:** This is an educational project. For real-world disease surveillance, consult with public health authorities and ensure compliance with applicable regulations (HIPAA, GDPR, etc.).

---

**Ready to start?** Follow the [setup_guide.md](setup_guide.md) to get started!

**Last updated:** 2025-11-14 | Research Jumpstart v1.0.0
