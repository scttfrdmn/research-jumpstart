# Astronomical Image Processing with S3 and Lambda

**Duration:** 2-4 hours
**Cost:** $7-10
**Platform:** AWS (requires account)
**Services:** S3, Lambda, Athena, IAM
**Data:** FITS images (~500MB - 1GB)

## Project Overview

Build a complete astronomical image processing pipeline on AWS. Upload FITS images to S3, use Lambda functions to detect astronomical sources, store results in a searchable catalog, and query your data with SQL using Athena.

This project bridges the gap between free Studio Lab (Tier 1) and production infrastructure (Tier 3), introducing you to:

- **Object storage** with S3
- **Serverless computing** with Lambda
- **SQL querying** with Athena
- **Identity & access** with IAM
- **Cost-effective cloud processing** of astronomical data

### What You'll Learn

1. Upload FITS astronomical images to AWS S3
2. Create and deploy a Lambda function for source detection
3. Store results as queryable catalogs in S3
4. Use Athena for SQL queries on your catalog
5. Visualize and analyze results in a Jupyter notebook

### Architecture Diagram

```
FITS Images (Local)
        │
        ▼
    ┌─────────────────────────────┐
    │  Python Upload Script        │
    │  (boto3)                     │
    └────────────┬─────────────────┘
                 │
                 ▼
         ┌──────────────┐
         │   S3 Bucket  │
         │ (Raw Images) │
         └──────┬───────┘
                │
    ┌───────────┴────────────┐
    │                        │
    ▼                        ▼
┌─────────────────┐  ┌──────────────────┐
│ Lambda Function │  │ Athena (SQL)     │
│ (Source         │  │ Queries Results  │
│  Detection)     │  │                  │
└────────┬────────┘  └──────────────────┘
         │
         ▼
    ┌──────────────┐
    │   S3 Bucket  │
    │  (Catalog)   │
    └──────┬───────┘
           │
           ▼
    ┌──────────────────┐
    │ Jupyter Notebook │
    │ (Visualization)  │
    └──────────────────┘
```

## Prerequisites

### Before You Start

1. **AWS Account** - Free tier eligible (but will incur ~$7-10)
2. **AWS CLI** - Configured with credentials
3. **Python 3.8+** - Local development environment
4. **boto3** - AWS SDK for Python
5. **jupyter** - For the analysis notebook

### Setup AWS CLI (5 minutes)

```bash
# Install AWS CLI (if not already installed)
# macOS
brew install awscli

# Linux
pip install awscli

# Verify installation
aws --version

# Configure credentials (get from AWS Console)
aws configure
# Enter: AWS Access Key ID
# Enter: AWS Secret Access Key
# Enter: Default region (e.g., us-east-1)
# Enter: Default output format (json)

# Verify configuration
aws s3 ls
```

### Install Python Dependencies

```bash
# Install requirements
pip install -r requirements.txt

# Create a working directory
mkdir -p ~/astronomy-tier2-project
cd ~/astronomy-tier2-project
```

## Step-by-Step Setup Guide

### Step 1: Create IAM Role for Lambda (10 minutes)

Lambda needs permissions to read from S3 and write logs. Create an IAM role:

1. Go to [IAM Console](https://console.aws.amazon.com/iam/)
2. Click "Roles" → "Create role"
3. Select "AWS Lambda" as trusted entity
4. Click "Next: Permissions"
5. Attach policies:
   - `AmazonS3FullAccess` (for S3 read/write)
   - `CloudWatchLogsFullAccess` (for logging)
6. Name the role: `lambda-astronomy-role`
7. Copy the **Role ARN** (format: `arn:aws:iam::ACCOUNT_ID:role/lambda-astronomy-role`)

**Alternative: Using AWS CLI**

```bash
# Create trust policy file
cat > lambda-trust-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create role
aws iam create-role \
  --role-name lambda-astronomy-role \
  --assume-role-policy-document file://lambda-trust-policy.json

# Attach S3 policy
aws iam attach-role-policy \
  --role-name lambda-astronomy-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# Attach CloudWatch logs policy
aws iam attach-role-policy \
  --role-name lambda-astronomy-role \
  --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess

# Get role ARN
aws iam get-role --role-name lambda-astronomy-role --query 'Role.Arn'
```

### Step 2: Create S3 Buckets (5 minutes)

```bash
# Define a unique bucket name (S3 bucket names must be globally unique)
BUCKET_NAME="astronomy-tier2-$(date +%s)"

# Create bucket for raw FITS images
aws s3 mb s3://${BUCKET_NAME}-raw --region us-east-1

# Create bucket for catalogs
aws s3 mb s3://${BUCKET_NAME}-catalog --region us-east-1

# Verify buckets
aws s3 ls

# Save bucket names for later
echo "BUCKET_RAW=s3://${BUCKET_NAME}-raw" >> ~/.astronomy_env
echo "BUCKET_CATALOG=s3://${BUCKET_NAME}-catalog" >> ~/.astronomy_env
```

### Step 3: Upload Sample FITS Images (10 minutes)

```bash
# Download sample FITS images (Sloan Digital Sky Survey)
python scripts/download_sample_fits.py

# Upload images to S3
python scripts/upload_to_s3.py

# Verify upload
aws s3 ls s3://${BUCKET_NAME}-raw/
```

### Step 4: Create and Deploy Lambda Function (15 minutes)

```bash
# Package Lambda function
cd scripts
zip lambda_function.zip lambda_function.py

# Get your AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Get the role ARN you created
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/lambda-astronomy-role"

# Deploy Lambda function
aws lambda create-function \
  --function-name astronomy-source-detection \
  --runtime python3.11 \
  --role ${ROLE_ARN} \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://lambda_function.zip \
  --timeout 300 \
  --memory-size 1024 \
  --environment Variables="{BUCKET_RAW=${BUCKET_NAME}-raw,BUCKET_CATALOG=${BUCKET_NAME}-catalog}"

cd ..
```

### Step 5: Run Source Detection (20 minutes)

```bash
# Invoke Lambda for each FITS image
python scripts/invoke_lambda.py

# Check Lambda logs
aws logs tail /aws/lambda/astronomy-source-detection --follow
```

### Step 6: Query Results with Athena (15 minutes)

```bash
# Create Athena table and run queries
python scripts/query_with_athena.py

# Or use the Jupyter notebook for interactive queries
jupyter notebook notebooks/sky_analysis.ipynb
```

## Project Files

```
tier-2/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── setup_guide.md                 # AWS setup step-by-step (detailed)
│
├── notebooks/
│   └── sky_analysis.ipynb        # Interactive analysis notebook
│
├── scripts/
│   ├── download_sample_fits.py   # Download test data from SDSS
│   ├── upload_to_s3.py           # Upload FITS to S3
│   ├── lambda_function.py        # Source detection code
│   ├── invoke_lambda.py          # Trigger Lambda for all images
│   └── query_with_athena.py      # SQL queries on results
│
├── data/
│   └── README.md                 # Data directory (git-ignored)
│
└── cleanup_guide.md              # How to delete all resources
```

## Running the Project

### Quick Start (Automated)

```bash
# 1. Setup AWS (one-time)
bash scripts/setup_aws.sh

# 2. Download and upload sample data
python scripts/download_sample_fits.py
python scripts/upload_to_s3.py

# 3. Run source detection
python scripts/invoke_lambda.py

# 4. Query and visualize
jupyter notebook notebooks/sky_analysis.ipynb
```

### Manual Steps

```bash
# Terminal 1: Upload images
source ~/.astronomy_env
python scripts/upload_to_s3.py

# Terminal 2: Monitor Lambda
aws logs tail /aws/lambda/astronomy-source-detection --follow

# Terminal 3: Interactive analysis
jupyter notebook notebooks/sky_analysis.ipynb
```

## What You'll Discover

### Sample Output

After running the source detection pipeline, you'll get:

**Detected Sources in Image:**
```
Source ID | RA (°)  | Dec (°) | Flux (μJy) | SNR
1         | 185.234 | 15.567  | 1245      | 45.3
2         | 185.289 | 15.612  | 892       | 32.1
3         | 185.301 | 15.489  | 1567      | 52.8
```

**SQL Query Results:**
```sql
SELECT COUNT(*) as total_sources,
       AVG(flux) as mean_flux,
       MAX(snr) as max_snr
FROM sources
WHERE flux > 500;

-- Results:
-- total_sources: 2847
-- mean_flux: 1124.3
-- max_snr: 89.5
```

### Scientific Questions You Can Answer

1. **How many astronomical sources are in my images?**
   - Count objects by type and brightness
   - Find the faintest detectable sources

2. **What are the properties of detected sources?**
   - Position, brightness, morphology
   - Spatial distribution across sky

3. **Are there any interesting objects?**
   - Find rare high-flux sources
   - Identify extended sources (galaxies vs stars)

4. **How does detection efficiency vary?**
   - With image quality
   - With position on the detector
   - With source brightness

## Cost Breakdown

**Detailed cost estimate for this project:**

| Service | Usage | Cost |
|---------|-------|------|
| **S3 Storage** | 1GB images + 100MB catalog | $0.50 |
| **S3 Requests** | ~5000 read/write ops | $0.25 |
| **Lambda Invocations** | 50 invocations × 30 sec | $1.00 |
| **Lambda Compute** | 50 × 30 sec on 1GB = 1500 GB-sec | $3.00 |
| **Athena Queries** | 10 queries scanning 100MB | $0.50 |
| **CloudWatch Logs** | ~100MB ingestion | $0.50 |
| **Data Transfer** | ~1GB out of AWS | $1.00 |
| **Miscellaneous** | Buffer for rounding | $0.25 |
| **TOTAL** | | **$7-10** |

**Cost Optimization:**
- Use AWS Free Tier where applicable (1GB S3 storage)
- Run once and analyze multiple times
- Use lifecycle policies to delete old logs
- Cleanup resources when done (see cleanup_guide.md)

## Architecture Details

### Lambda Function

The Lambda function performs source detection on each FITS image:

1. **Download image** from S3
2. **Read FITS** using astropy
3. **Detect sources** using SEP (Source Extraction Program)
4. **Calculate properties**: position, flux, morphology
5. **Save catalog** to S3 as Parquet
6. **Log results** to CloudWatch

```python
# Example detection code
from astropy.io import fits
import sep
import numpy as np

# Load FITS
data = fits.getdata('image.fits').astype(float)

# Detect sources (5-sigma above background)
objects = sep.extract(data, thresh=5.0, err=np.ones_like(data))

# Measure properties
print(f"Detected {len(objects)} sources")
```

### Athena Queries

Once results are stored in S3 as Parquet format, Athena lets you query them with SQL:

```sql
-- Find brightest sources
SELECT ra, dec, flux, snr
FROM sources
WHERE flux > 1000
ORDER BY flux DESC
LIMIT 10;

-- Count by SNR ranges
SELECT
  CASE
    WHEN snr > 50 THEN 'Very High'
    WHEN snr > 20 THEN 'High'
    WHEN snr > 5 THEN 'Medium'
    ELSE 'Low'
  END as snr_class,
  COUNT(*) as count
FROM sources
GROUP BY snr_class;
```

## Troubleshooting

### Lambda Timeout

If Lambda times out (default 30 seconds):

```bash
# Increase timeout to 5 minutes
aws lambda update-function-configuration \
  --function-name astronomy-source-detection \
  --timeout 300
```

### Permission Denied

If you get "Access Denied" errors:

```bash
# Check IAM role has S3 permissions
aws iam get-role-policy --role-name lambda-astronomy-role --policy-name <POLICY>

# Check bucket policy
aws s3api get-bucket-policy --bucket your-bucket-name
```

### Images Not Uploading

```bash
# Verify S3 bucket exists
aws s3 ls

# Check your credentials
aws sts get-caller-identity

# Verify you can write to bucket
aws s3 cp test.txt s3://your-bucket-name/
```

### Athena Table Not Found

```bash
# Create Glue table manually
aws glue create-table \
  --database-name default \
  --table-input file://table-definition.json
```

## Next Steps

### Beginner Extensions (30 minutes - 1 hour)

1. **Process More Images**
   - Download images from different sky regions
   - Compare source detection across fields

2. **Analyze Source Properties**
   - Create color-magnitude diagrams
   - Identify stars vs galaxies

3. **Monitor Costs**
   - Track spending in AWS Cost Explorer
   - Optimize Lambda memory and timeout

### Intermediate Extensions (1-2 hours)

4. **Automated Pipeline**
   - Set up S3 event triggers for Lambda
   - Automatically process new uploads

5. **Add More Catalogs**
   - Cross-match with SDSS or Gaia
   - Find matches in external surveys

6. **Advanced Queries**
   - Spatial SQL queries in Athena
   - Statistical analysis of source population

### Advanced Extensions (2-4 hours)

7. **Scale to Full Sky**
   - Process thousands of images in parallel
   - Use AWS Batch for distributed processing

8. **Real-Time Processing**
   - Set up SNS alerts for interesting sources
   - Build dashboard with QuickSight

9. **Machine Learning**
   - Train classifier to identify AGN or quasars
   - Deploy model as Lambda layer

## Key Concepts Learned

### AWS Services

| Service | Purpose | What We Did |
|---------|---------|-------------|
| **S3** | Object storage | Stored FITS images and catalogs |
| **Lambda** | Serverless compute | Ran source detection on each image |
| **Athena** | SQL queries on S3 | Queried catalog results |
| **IAM** | Access control | Created role for Lambda |
| **CloudWatch** | Monitoring | Viewed Lambda logs |

### Astronomical Concepts

| Concept | Explanation | Our Application |
|---------|-------------|-----------------|
| **FITS** | Flexible Image Transport System (astronomy standard) | Stored astronomical images |
| **Source Detection** | Finding objects in astronomical images | Lambda function task |
| **Photometry** | Measuring brightness and positions | Lambda function output |
| **SNR** | Signal-to-Noise Ratio (quality metric) | Filter results by detection quality |
| **WCS** | World Coordinate System (sky coordinates) | Convert image pixels to RA/Dec |

## Performance Benchmarks

**Source Detection Performance:**

| Image Size | Pixel Count | Detection Time | Sources Found |
|------------|------------|-----------------|---------------|
| 1024×1024 | 1M | 5 sec | ~100-500 |
| 2048×1536 | 3M | 15 sec | ~500-2000 |
| 4096×4096 | 16M | 45 sec | ~2000-10000 |

**Athena Query Performance:**

| Query Type | Data Scanned | Time | Cost |
|------------|-------------|------|------|
| Simple COUNT | 100MB | 2-3 sec | $0.0005 |
| Aggregation | 500MB | 5-10 sec | $0.0025 |
| Large JOIN | 1GB+ | 15-30 sec | $0.0050 |

## Comparison: Tier 1 vs Tier 2 vs Tier 3

| Feature | Tier 1 (Studio Lab) | Tier 2 (AWS Lambda) | Tier 3 (CloudFormation) |
|---------|-------------------|-------------------|----------------------|
| **Cost** | Free | $5-15 | $50-500/month |
| **Setup Time** | 5 min | 45 min | 1-2 hours |
| **Data Size** | Up to 15GB | Unlimited | Petabytes |
| **Compute** | 1 GPU / 4 CPU | Massively parallel | Dedicated infrastructure |
| **Duration** | 4-12 hours | Per-invocation (5 min max) | 24/7 |
| **Persistence** | Notebook + storage | S3 + databases | Full data lake |
| **Learning Curve** | Gentle | Moderate | Steep |

## Resources

### AWS Documentation
- [S3 Documentation](https://docs.aws.amazon.com/s3/)
- [Lambda Developer Guide](https://docs.aws.amazon.com/lambda/)
- [Athena Documentation](https://docs.aws.amazon.com/athena/)
- [IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)

### Astronomy Packages
- [Astropy](https://www.astropy.org/) - Core astronomy library
- [SEP](https://sep.readthedocs.io/) - Source Extraction Program
- [Photutils](https://photutils.readthedocs.io/) - Photometry utilities
- [FITS Format](https://fits.gsfc.nasa.gov/) - FITS standard

### AWS Free Tier
- [AWS Free Tier](https://aws.amazon.com/free/) - Check eligibility
- [Cost Calculator](https://calculator.aws/) - Estimate your costs

### Example Datasets
- [SDSS](https://www.sdss.org/) - Sloan Digital Sky Survey
- [Pan-STARRS](https://panstarrs.stsci.edu/) - Panoramic Survey
- [Legacy Survey](https://www.legacysurvey.org/) - Public imaging

## Appendix: Sample Commands

### List all Lambda functions
```bash
aws lambda list-functions --region us-east-1
```

### View Lambda function code
```bash
aws lambda get-function --function-name astronomy-source-detection
```

### Manually invoke Lambda
```bash
aws lambda invoke \
  --function-name astronomy-source-detection \
  --payload '{"bucket":"my-bucket","key":"image.fits"}' \
  response.json
```

### Check CloudWatch logs
```bash
aws logs describe-log-groups
aws logs describe-log-streams --log-group-name /aws/lambda/astronomy-source-detection
aws logs get-log-events --log-group-name /aws/lambda/astronomy-source-detection --log-stream-name <stream-name>
```

### Query Athena
```bash
aws athena start-query-execution \
  --query-string "SELECT COUNT(*) FROM sources" \
  --result-configuration OutputLocation=s3://your-bucket/athena-results/
```

---

**Tier 2 Project Status:** Production-ready
**Estimated Setup Time:** 45-60 minutes
**Processing Time:** 1-2 hours (per run)
**Cost:** $7-10 per execution

Ready to explore AWS? Start with Step 1 of the setup guide above!
