# Quick Start Guide - Tier 2 Astronomy Project

Get up and running in 60 minutes!

## Prerequisites (5 minutes)

- AWS Account with credentials configured
- Python 3.8+
- Terminal/bash access

```bash
# Verify setup
aws sts get-caller-identity
python --version
```

## Step-by-Step (45 minutes)

### 1. Install Dependencies (2 minutes)

```bash
pip install -r requirements.txt
```

### 2. Set Up AWS Resources (25 minutes)

Follow the detailed setup in `setup_guide.md`:

```bash
# Quick version:
# 1. Create IAM role (manual or CLI)
# 2. Create S3 buckets
# 3. Create Lambda function
# 4. Set up Athena

# Save bucket names
export BUCKET_RAW="astronomy-tier2-XXXXX-raw"
export BUCKET_CATALOG="astronomy-tier2-XXXXX-catalog"
echo "export BUCKET_RAW=${BUCKET_RAW}" >> ~/.astronomy_env
echo "export BUCKET_CATALOG=${BUCKET_CATALOG}" >> ~/.astronomy_env
source ~/.astronomy_env
```

### 3. Download and Upload (10 minutes)

```bash
# Download sample FITS images (~50 MB)
python scripts/download_sample_fits.py

# Upload to S3
python scripts/upload_to_s3.py

# Verify
aws s3 ls s3://${BUCKET_RAW}/images/
```

### 4. Run Source Detection (5 minutes)

```bash
# Invoke Lambda for each image
python scripts/invoke_lambda.py

# Watch logs
aws logs tail /aws/lambda/astronomy-source-detection --follow
```

### 5. Analyze Results (5 minutes)

```bash
# Query with Athena
python scripts/query_with_athena.py

# Or use Jupyter
jupyter notebook notebooks/sky_analysis.ipynb
```

## Verify Success

All of these should work:

```bash
# 1. Check S3 buckets
aws s3 ls | grep astronomy

# 2. Check Lambda function
aws lambda list-functions | grep astronomy-source-detection

# 3. Check CloudWatch logs
aws logs describe-log-groups | grep lambda

# 4. Check Athena results
aws s3 ls s3://${BUCKET_CATALOG}/sources/ --recursive

# 5. Run Athena query
python scripts/query_with_athena.py
```

## Typical Output

**Lambda Invocation:**
```
✓ Found 3 FITS files
Processing images...
  ✓ images/sdss_test_frame_g.fits: 145 sources
  ✓ images/sdss_test_frame_r.fits: 189 sources
  ✓ images/sdss_test_frame_i.fits: 167 sources
Total: 501 sources detected
```

**Athena Query:**
```
Query: SELECT COUNT(*) as total_sources FROM astronomy.sources;

Results:
total_sources
501
```

## Common Issues & Fixes

| Issue | Solution |
|-------|----------|
| "No AWS credentials" | Run `aws configure` |
| "Bucket already exists" | Use unique name with timestamp |
| "Lambda timeout" | Increase timeout: `aws lambda update-function-configuration --timeout 300` |
| "Table not found" | Create table manually in setup_guide.md |
| "Permission denied" | Check IAM role has S3 and CloudWatch permissions |

## Next Steps

1. **Explore data** - Use Jupyter notebook to visualize results
2. **Custom queries** - Write your own Athena SQL queries
3. **Scale up** - Process more images, larger datasets
4. **Extend** - Add cross-matching with other surveys
5. **Production** - Move to Tier 3 with CloudFormation

## Cost Tracking

During execution, check costs:

```bash
# View CloudWatch metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/Lambda \
  --metric-name Invocations \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-01-02T00:00:00Z \
  --period 3600 \
  --statistics Sum

# Check AWS Cost Explorer
# https://console.aws.amazon.com/cost-management/
```

## Cleanup

When finished, delete resources to avoid charges:

```bash
# Manual cleanup
aws lambda delete-function --function-name astronomy-source-detection
aws iam delete-role --role-name lambda-astronomy-role
aws s3 rm s3://${BUCKET_RAW} --recursive
aws s3 rb s3://${BUCKET_RAW}
aws s3 rm s3://${BUCKET_CATALOG} --recursive
aws s3 rb s3://${BUCKET_CATALOG}

# Or use automated script
bash scripts/cleanup_all.sh
```

Or see detailed instructions in `cleanup_guide.md`.

## Getting Help

- **AWS Documentation:** https://docs.aws.amazon.com/
- **Astronomy packages:** https://www.astropy.org/
- **AWS CLI:** `aws help`
- **Project issues:** See README.md troubleshooting section

## Key Files

| File | Purpose |
|------|---------|
| `README.md` | Complete documentation |
| `setup_guide.md` | Detailed AWS setup (45 min) |
| `cleanup_guide.md` | How to delete resources |
| `requirements.txt` | Python dependencies |
| `notebooks/sky_analysis.ipynb` | Analysis and visualization |
| `scripts/download_sample_fits.py` | Download test data |
| `scripts/upload_to_s3.py` | Upload to S3 |
| `scripts/lambda_function.py` | Source detection code |
| `scripts/invoke_lambda.py` | Trigger Lambda |
| `scripts/query_with_athena.py` | Query results |

## Expected Timeline

- Setup: 45 minutes (one-time)
- Data download: 5 minutes
- Data upload: 2 minutes
- Lambda execution: 5 minutes
- Analysis: 10 minutes
- **Total: ~60 minutes**

Ready to start? Run `setup_guide.md` now!
