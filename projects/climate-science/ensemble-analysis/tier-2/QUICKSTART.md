# Quick Start Guide - 5 Minutes to Climate Analysis

Get up and running with AWS climate data analysis in 5 steps.

---

## Prerequisites (Check First!)

- ✅ AWS account with billing enabled
- ✅ Python 3.8+ installed
- ✅ AWS CLI configured (`aws configure`)
- ✅ ~$10-15 budget for testing

```bash
# Verify setup
aws sts get-caller-identity
python --version
```

---

## The 5-Step Path

### Step 1: Clone and Setup (5 minutes)

```bash
# Navigate to project
cd tier-2

# Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify
python -c "import boto3; print('✓ boto3 imported')"
```

### Step 2: Create AWS Resources (10 minutes)

Follow the detailed steps in `setup_guide.md`:

**Quick summary:**
1. Create S3 bucket: `aws s3 mb s3://climate-data-$(date +%s)`
2. Create IAM role (console or CLI)
3. Create Lambda function (console or CLI)
4. Save bucket name to `.env` file

```bash
# Create .env file
cat > .env << 'EOF'
AWS_REGION=us-east-1
AWS_S3_BUCKET=climate-data-xxxx
AWS_LAMBDA_FUNCTION=process-climate-data
EOF
```

### Step 3: Upload Sample Data (5-10 minutes)

```bash
# Set bucket name
export AWS_S3_BUCKET="climate-data-xxxx"

# Generate sample data (if you don't have real data)
python -c "
import numpy as np
import xarray as xr
lat = np.linspace(-90, 90, 180)
lon = np.linspace(-180, 180, 360)
time = np.arange(1950, 2100)
temp = np.random.randn(len(time), len(lat), len(lon)) * 2 + 288
ds = xr.Dataset({'tas': (['time', 'lat', 'lon'], temp)},
                 coords={'time': time, 'lat': lat, 'lon': lon})
ds['tas'].attrs['units'] = 'K'
ds.to_netcdf('sample_data/test_temperature.nc')
print('✓ Created sample data')
"

# Upload data
python scripts/upload_to_s3.py \
  --bucket $AWS_S3_BUCKET \
  --data-dir sample_data/
```

### Step 4: Process Data (5 minutes)

```bash
# Deploy Lambda function (copy lambda_function.py to AWS Lambda console)
# See setup_guide.md Step 4 for detailed instructions

# Or invoke Lambda manually (for testing)
python -c "
import boto3
lam = boto3.client('lambda')
response = lam.invoke(
    FunctionName='process-climate-data',
    InvocationType='RequestResponse',
    Payload='{\"Records\":[{\"s3\":{\"bucket\":{\"name\":\"climate-data-xxxx\"},\"object\":{\"key\":\"raw/test_temperature.nc\"}}}]}'
)
print(response['StatusCode'])
"
```

### Step 5: Analyze Results (10 minutes)

```bash
# Download results
python scripts/query_results.py \
  --bucket $AWS_S3_BUCKET \
  --output-dir ./results

# Analyze in Jupyter
jupyter notebook notebooks/climate_analysis.ipynb

# Or use Python directly
python -c "
import json
import pandas as pd
with open('results/analysis_summary.json') as f:
    summary = json.load(f)
print(json.dumps(summary, indent=2))
"
```

---

## Running Each Step Individually

### Just Upload Data

```bash
python scripts/upload_to_s3.py --bucket climate-data-xxxx --data-dir sample_data/
```

### Just Download Results

```bash
python scripts/query_results.py --bucket climate-data-xxxx --output-dir ./results
```

### Just List Files

```bash
aws s3 ls s3://climate-data-xxxx/raw/
aws s3 ls s3://climate-data-xxxx/results/
```

### Just Check Costs

```bash
aws ce get-cost-and-usage \
  --time-period Start=$(date -d '1 day ago' +%Y-%m-%d),End=$(date +%Y-%m-%d) \
  --granularity DAILY \
  --metrics UnblendedCost \
  --group-by Type=DIMENSION,Key=SERVICE
```

---

## Troubleshooting

**Problem: "NoCredentialsError"**
```bash
aws configure
# Enter: Access Key ID, Secret Key, region (us-east-1), format (json)
```

**Problem: "Bucket already exists"**
```bash
# Use unique name with timestamp
aws s3 mb s3://climate-data-$(date +%s%N)
```

**Problem: Lambda not processing**
1. Check S3 trigger is enabled (setup_guide.md Step 6)
2. Check IAM role has S3 permissions
3. Monitor Lambda logs: `aws logs tail /aws/lambda/process-climate-data --follow`

**Problem: High costs**
1. Delete old data: `aws s3 rm s3://climate-data-xxxx/raw/ --recursive`
2. Increase Lambda timeout to avoid retries
3. Follow cleanup_guide.md when done

---

## Key Commands Reference

```bash
# AWS Resources
aws s3 ls                                    # List buckets
aws s3 ls s3://bucket-name --recursive      # List bucket contents
aws lambda list-functions                    # List Lambda functions
aws iam list-roles | grep lambda             # List IAM roles

# Upload/Download
aws s3 cp file.nc s3://bucket/raw/
aws s3 cp s3://bucket/results/ ./ --recursive

# Lambda
aws lambda invoke --function-name process-climate-data response.json
aws logs tail /aws/lambda/process-climate-data --follow

# Cleanup (when done!)
aws s3 rm s3://bucket --recursive
aws s3 rb s3://bucket
aws lambda delete-function --function-name process-climate-data
aws iam delete-role --role-name lambda-climate-processor
```

---

## Expected Timeline

| Step | Time | Notes |
|------|------|-------|
| Setup environment | 5 min | Virtual environment, pip install |
| Create AWS resources | 10 min | S3, IAM, Lambda (manual setup) |
| Upload data | 5-10 min | Depends on file size |
| Process (Lambda) | 5 min | Usually faster with S3 trigger |
| Download results | 5 min | Depends on file size |
| Analysis | 10 min | Jupyter notebook |
| **Total** | **40-45 min** | For first run |

---

## What Happens Next?

### Option A: Continue Exploring
- Modify `lambda_function.py` to calculate different statistics
- Add more climate variables (precipitation, humidity)
- Compare multiple climate models
- Create more sophisticated visualizations

### Option B: Clean Up and Stop
- Follow `cleanup_guide.md`
- Delete S3 bucket and Lambda function
- Stop AWS charges

### Option C: Move to Production (Tier 3)
- Use CloudFormation for infrastructure-as-code
- Deploy multi-region architecture
- Advanced monitoring and alerting
- Production-grade configurations

---

## Cost Estimate

For the quick start (5GB data, 100 Lambda invocations):

| Service | Cost |
|---------|------|
| S3 storage (5GB, 7 days) | $0.35 |
| S3 requests | $0.05 |
| Lambda executions | $1.50 |
| Lambda compute | $2.00 |
| Data transfer | $0.50 |
| **Total** | **~$4.40** |

Free tier covers most of this!

---

## Need Help?

**For AWS issues:** See `setup_guide.md` troubleshooting section

**For project issues:** See `README.md` troubleshooting section

**For cleanup:** See `cleanup_guide.md`

**For detailed docs:** Read `setup_guide.md` (comprehensive guide)

---

## Next: Read the Full Guides

- **setup_guide.md** - Detailed AWS setup with all options
- **README.md** - Full project overview and architecture
- **cleanup_guide.md** - How to delete resources and stop charges

---

**Ready to start?** Run Step 1 above! It'll take just 5 minutes.
