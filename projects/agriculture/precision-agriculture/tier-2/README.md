# Satellite Imagery Analysis with S3 and Lambda

**Tier 2 AWS Starter Project**

Analyze Sentinel-2 satellite imagery for crop health monitoring using AWS serverless services.

**Duration:** 2-4 hours
**Cost:** $8-11 per run
**AWS Services:** S3, Lambda, Athena, IAM
**Platform:** AWS account + local machine or Studio Lab

## Project Overview

Learn how to process satellite imagery at scale using AWS services. This project demonstrates:
- Uploading Sentinel-2 data to S3
- Computing NDVI (Normalized Difference Vegetation Index) with Lambda
- Querying field-level crop health metrics with Athena
- Visualizing results in Jupyter notebooks

**Use Case:** Monitor crop health across multiple fields, detect stress patterns, and track seasonal changes using satellite data.

## Architecture

```
Internet
   ↓
Sentinel-2 Imagery (10m resolution)
   ↓
[S3 Bucket: raw/] ← Upload script
   ↓
[Lambda Function] ← NDVI calculation, crop health metrics
   ↓
[S3 Bucket: results/] ← NDVI GeoTIFF, metrics CSV
   ↓
[Athena] ← SQL queries on field-level metrics
   ↓
[Jupyter Notebook] ← Visualization & analysis
```

## What You'll Learn

### AWS Concepts
- S3 bucket policies and object storage
- Lambda functions for serverless computing
- IAM roles with principle of least privilege
- Athena for querying structured data
- CloudWatch logging and monitoring

### Remote Sensing
- NDVI calculation from multispectral data
- Crop stress detection using vegetation indices
- Temporal analysis of field health
- Geospatial data handling with rasterio

### Python Tools
- boto3 for AWS SDK interactions
- rasterio for GeoTIFF processing
- pandas for tabular data analysis
- matplotlib/folium for visualization

## Prerequisites

### Before You Start
- AWS account with credit card (free tier usage recommended)
- Python 3.8+ installed locally
- Basic familiarity with AWS console (see setup guide)
- ~30 minutes to set up AWS resources

### Python Environment
```bash
# Clone or download this project
cd precision-agriculture/tier-2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. AWS Setup (15 minutes)
Follow the detailed steps in [setup_guide.md](setup_guide.md):
- Create S3 bucket for data storage
- Create IAM role for Lambda
- Deploy Lambda function
- Configure Athena

### 2. Upload Sample Data (5 minutes)
```bash
python scripts/upload_to_s3.py \
  --bucket your-bucket-name \
  --file sample_data/sentinel2_field.tif
```

### 3. Trigger Lambda Processing (2 minutes)
```bash
python scripts/query_results.py \
  --bucket your-bucket-name \
  --field-id field_001
```

### 4. Analyze Results (30 minutes)
```bash
jupyter notebook notebooks/crop_analysis.ipynb
```

## File Structure

```
tier-2/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── setup_guide.md                 # Step-by-step AWS setup
├── cleanup_guide.md               # Resource deletion guide
│
├── scripts/
│   ├── upload_to_s3.py           # Upload Sentinel-2 imagery
│   ├── lambda_function.py         # NDVI calculation code (deploy to Lambda)
│   └── query_results.py           # Query Athena, retrieve results
│
├── notebooks/
│   └── crop_analysis.ipynb        # Visualization and analysis
│
└── sample_data/
    ├── README.md                  # Data source documentation
    └── sentinel2_field.tif        # Example Sentinel-2 tile
```

## AWS Setup Summary

| Resource | Name | Purpose |
|----------|------|---------|
| S3 Bucket | `satellite-imagery-{user-id}` | Store imagery and results |
| Lambda | `process-ndvi-calculation` | Compute NDVI from imagery |
| IAM Role | `lambda-ndvi-processor` | Permissions for Lambda |
| Athena Table | `field_metrics` | Query crop health metrics |
| CloudWatch Log | `/aws/lambda/process-ndvi-calculation` | Monitor Lambda execution |

## Key Features

### NDVI Calculation
Vegetation Index formula: NDVI = (NIR - Red) / (NIR + Red)
- Values: -1.0 to 1.0
- Green vegetation: 0.4 - 0.8
- Stressed plants: < 0.3
- Water/clouds: < 0

### Lambda Processing
- **Trigger:** S3 event when image uploaded
- **Computation:** ~2-5 seconds per image
- **Output:** NDVI GeoTIFF + CSV metrics
- **Cost:** ~$0.0000002 per invocation

### Athena Queries
```sql
-- Example: Find stressed fields
SELECT field_id, date, avg_ndvi, min_ndvi
FROM field_metrics
WHERE avg_ndvi < 0.4
ORDER BY avg_ndvi ASC;
```

## Cost Breakdown

**Typical run costs:**

| Service | Usage | Cost |
|---------|-------|------|
| S3 Storage | 10 images × 20MB = 200MB | $0.01 |
| S3 Data Transfer | Upload + download | $0.02 |
| Lambda Invocations | 10 runs × 3 sec | $0.05 |
| Lambda Duration | 10 × 3 sec = 30 sec | $0.006 |
| Athena Queries | 10 queries × 5MB | $0.05 |
| **Total** | | **$0.18 - $0.25** |

**Multiplied by full project:** ~$8-11 (includes setup, testing, cleanup)

**Cost optimization:**
- AWS Free Tier: First 1M Lambda invocations free per month
- Use S3 Intelligent-Tiering to reduce storage costs
- Set Lambda timeout to 5 minutes (safety limit)
- Delete test resources after completion

## Running the Project

### Step 1: Configure AWS Credentials
```bash
# Set up AWS CLI credentials
aws configure
# Enter: AWS Access Key ID
# Enter: AWS Secret Access Key
# Enter: Default region (us-east-1)
# Enter: Default output (json)
```

### Step 2: Update Configuration
Edit `scripts/config.py` with your AWS resources:
```python
AWS_REGION = "us-east-1"
S3_BUCKET = "your-bucket-name"
LAMBDA_FUNCTION = "process-ndvi-calculation"
ATHENA_OUTPUT_LOCATION = "s3://your-bucket-name/athena-results/"
```

### Step 3: Run Upload Script
```bash
python scripts/upload_to_s3.py \
  --bucket your-bucket-name \
  --input sample_data/
```

### Step 4: Check Lambda Execution
```bash
# View Lambda logs
aws logs tail /aws/lambda/process-ndvi-calculation --follow
```

### Step 5: Query Results
```bash
python scripts/query_results.py \
  --bucket your-bucket-name \
  --output results/
```

### Step 6: Visualize Analysis
```bash
jupyter notebook notebooks/crop_analysis.ipynb
```

## Expected Results

After completing the project, you'll have:

1. **NDVI Images** - Visualization of crop health
   - Green areas: Healthy vegetation (NDVI > 0.5)
   - Yellow areas: Moderate stress (NDVI 0.3-0.5)
   - Red areas: Severe stress (NDVI < 0.3)

2. **Field Metrics CSV**
   ```
   field_id, date, avg_ndvi, min_ndvi, max_ndvi, vegetation_coverage
   field_001, 2024-06-15, 0.65, 0.42, 0.81, 0.87
   field_002, 2024-06-15, 0.52, 0.21, 0.73, 0.71
   field_003, 2024-06-15, 0.71, 0.55, 0.89, 0.93
   ```

3. **Athena Query Results** - SQL analysis on field metrics
   - Identify stressed fields
   - Track temporal trends
   - Compare field performance

4. **Jupyter Analysis** - Interactive visualization
   - NDVI maps with folium
   - Time series plots
   - Statistical summaries

## Troubleshooting

### Lambda Timeout
If Lambda times out on large images:
- Increase timeout in AWS Console (up to 15 minutes)
- Split large GeoTIFFs into smaller tiles
- Process in parallel with multiple Lambda functions

### S3 Upload Errors
```bash
# Check bucket permissions
aws s3 ls s3://your-bucket-name/

# Check IAM credentials
aws sts get-caller-identity
```

### Athena Query Errors
- Ensure CSV has proper column headers
- Check data types match table schema
- Verify S3 output location is writable

### Python Import Errors
```bash
# Verify dependencies installed
pip list | grep rasterio

# Reinstall if needed
pip install --force-reinstall rasterio
```

## Cleanup

**IMPORTANT:** Delete AWS resources after completion to avoid charges.

See [cleanup_guide.md](cleanup_guide.md) for detailed instructions:
1. Delete S3 bucket and contents
2. Delete Lambda function
3. Delete IAM role
4. Delete Athena table
5. Delete CloudWatch logs

**Quick cleanup:**
```bash
python scripts/cleanup.py --bucket your-bucket-name
```

## Next Steps

After mastering Tier 2:

1. **Expand to Tier 3:** Infrastructure as Code with CloudFormation
   - Automated resource deployment
   - Multi-region replication
   - Cost monitoring and alerts
   - Production-grade security

2. **Advanced Capabilities:**
   - Process multiple satellite sensors (Landsat, MODIS)
   - Real-time alerts for crop stress
   - Machine learning for yield prediction
   - Integration with IoT sensor data

3. **Scaling to Production:**
   - Process 10,000+ fields in parallel
   - Cost: $50-500/month
   - Serverless architecture handles scale automatically

## Resources

### AWS Documentation
- [S3 User Guide](https://docs.aws.amazon.com/s3/)
- [Lambda Developer Guide](https://docs.aws.amazon.com/lambda/)
- [Athena User Guide](https://docs.aws.amazon.com/athena/)
- [IAM Best Practices](https://docs.aws.amazon.com/iam/latest/userguide/best-practices.html)

### Remote Sensing
- [Sentinel-2 on AWS Registry](https://registry.opendata.aws/sentinel-2/)
- [USGS NDVI Information](https://www.usgs.gov/core-science-systems/nli/landsat/normalized-difference-vegetation-index)
- [Rasterio Documentation](https://rasterio.readthedocs.io/)

### Data Sources
- [AWS Open Data Registry](https://registry.opendata.aws/)
- [Copernicus Sentinel Hub](https://www.sentinelhub.com/)
- [Planet Labs](https://www.planet.com/)

## Support

**Issues or questions?**
- Check troubleshooting section above
- Review AWS service documentation
- See setup_guide.md for step-by-step help

## License

This project is part of Research Jumpstart and is released under the MIT License.

---

**Next:** [AWS Setup Guide](setup_guide.md)

**Built with AWS and Python | Part of Research Jumpstart**
