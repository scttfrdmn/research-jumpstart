# AWS Setup Guide - Climate Data Analysis Tier 2

This guide walks you through setting up all AWS resources needed for the climate data analysis project. Estimated time: 30-45 minutes.

---

## Prerequisites

Before starting, ensure you have:
- ✅ AWS account with billing enabled
- ✅ AWS CLI installed (`aws --version`)
- ✅ AWS credentials configured (`aws configure`)
- ✅ Python 3.8+ installed
- ✅ $20 budget available for testing

---

## Step 1: Create S3 Bucket

S3 stores your climate data, processed results, and Lambda function logs.

### Option A: Using AWS Console (Recommended for beginners)

1. Go to [AWS S3 Console](https://console.aws.amazon.com/s3/)
2. Click "Create bucket"
3. Enter bucket name: `climate-data-{your-name}-{date}`
   - Example: `climate-data-alice-20250114`
   - Must be globally unique
   - Must be lowercase
   - No special characters
4. Select region: **us-east-1** (same as CMIP6 data)
5. Keep all default settings
6. Click "Create bucket"

### Option B: Using AWS CLI

```bash
# Create S3 bucket
BUCKET_NAME="climate-data-$(whoami)-$(date +%s)"
aws s3 mb "s3://$BUCKET_NAME" --region us-east-1

# Verify bucket created
aws s3 ls | grep climate-data

# Save bucket name for later
echo "BUCKET_NAME=$BUCKET_NAME" > .env
```

### Verify Bucket Created

```bash
# List your buckets
aws s3 ls

# Should see your new climate-data-xxxx bucket
```

**Save your bucket name!** You'll use it in later steps.

---

## Step 2: Create S3 Folder Structure

Lambda will expect data organized in specific folders.

### Using AWS Console

1. Open your climate-data bucket
2. Create folders:
   - Click "Create folder" → name it `raw` → Create
   - Click "Create folder" → name it `results` → Create
   - Click "Create folder" → name it `logs` → Create

### Using AWS CLI

```bash
# Create folder structure
aws s3api put-object --bucket "$BUCKET_NAME" --key "raw/"
aws s3api put-object --bucket "$BUCKET_NAME" --key "results/"
aws s3api put-object --bucket "$BUCKET_NAME" --key "logs/"

# Verify
aws s3 ls "s3://$BUCKET_NAME/" --recursive
```

---

## Step 3: Create IAM Role for Lambda

Lambda needs permissions to read/write S3 and write logs.

### Option A: Using AWS Console

1. Go to [IAM Console](https://console.aws.amazon.com/iam/home)
2. Click "Roles" in left menu
3. Click "Create role"
4. Choose "Lambda" as trusted entity
5. Click "Next: Permissions"
6. Search for and select these policies:
   - `AWSLambdaBasicExecutionRole` (CloudWatch logs)
   - `AmazonS3FullAccess` (S3 read/write)
7. Click "Next: Tags"
8. Add tag: Key=`Environment` Value=`tier-2`
9. Role name: `lambda-climate-processor`
10. Click "Create role"

### Option B: Using AWS CLI

```bash
# Create IAM role
ROLE_JSON=$(cat <<'EOF'
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
)

aws iam create-role \
  --role-name lambda-climate-processor \
  --assume-role-policy-document "$ROLE_JSON"

# Attach policies
aws iam attach-role-policy \
  --role-name lambda-climate-processor \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam attach-role-policy \
  --role-name lambda-climate-processor \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# Get role ARN (you'll need this)
ROLE_ARN=$(aws iam get-role --role-name lambda-climate-processor \
  --query 'Role.Arn' --output text)
echo "ROLE_ARN=$ROLE_ARN" >> .env
```

### Verify Role Created

```bash
aws iam get-role --role-name lambda-climate-processor
```

**Save your role ARN!** Format: `arn:aws:iam::ACCOUNT_ID:role/lambda-climate-processor`

---

## Step 4: Create Lambda Function

Lambda processes your climate data files.

### Option A: Using AWS Console

1. Go to [Lambda Console](https://console.aws.amazon.com/lambda/)
2. Click "Create function"
3. Function name: `process-climate-data`
4. Runtime: **Python 3.11**
5. Architecture: **x86_64**
6. Permissions: Choose existing role
   - Select `lambda-climate-processor`
7. Click "Create function"

#### Configure Function:

1. In function page, scroll down to "Code source"
2. Delete default code
3. Copy entire contents of `scripts/lambda_function.py`
4. Paste into editor
5. Click "Deploy"

#### Increase Timeout:

1. Scroll to "Configuration" tab
2. Click "General configuration" → "Edit"
3. Timeout: Change to **300 seconds** (5 minutes)
4. Memory: Keep at **128 MB** (or increase to 512 MB for large files)
5. Click "Save"

#### Set Environment Variables:

1. In Configuration tab
2. Scroll to "Environment variables"
3. Click "Edit"
4. Add variable: `BUCKET_NAME` = `climate-data-xxxx`
5. Add variable: `PROCESS_MODE` = `calculate_statistics`
6. Click "Save"

### Option B: Using AWS CLI

```bash
# Create Lambda function from zip file
# First, create deployment package
cd scripts
zip lambda_function.zip lambda_function.py
cd ..

# Wait for role to be available (usually 10-30 seconds)
sleep 15

# Create function
LAMBDA_ARN=$(aws lambda create-function \
  --function-name process-climate-data \
  --runtime python3.11 \
  --role "$ROLE_ARN" \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://scripts/lambda_function.zip \
  --timeout 300 \
  --memory-size 512 \
  --environment Variables="{BUCKET_NAME=$BUCKET_NAME,PROCESS_MODE=calculate_statistics}" \
  --query 'FunctionArn' \
  --output text)

echo "LAMBDA_ARN=$LAMBDA_ARN" >> .env

# Verify function created
aws lambda get-function --function-name process-climate-data
```

### Test Lambda Function

```bash
# Create test event
TEST_EVENT=$(cat <<'EOF'
{
  "Records": [
    {
      "s3": {
        "bucket": {
          "name": "climate-data-xxxx"
        },
        "object": {
          "key": "raw/sample_data.nc"
        }
      }
    }
  ]
}
EOF
)

# Note: Replace climate-data-xxxx with your actual bucket name

# Invoke function with test event
aws lambda invoke \
  --function-name process-climate-data \
  --payload "$TEST_EVENT" \
  --cli-binary-format raw-in-base64-out \
  response.json

# View response
cat response.json
```

---

## Step 5: Upload Sample Data

Upload climate data to test the pipeline.

### Option A: Using Python Script (Recommended)

```bash
# Configure bucket name
export AWS_S3_BUCKET="climate-data-xxxx"

# Run upload script
python scripts/upload_to_s3.py

# Verify upload
aws s3 ls "s3://$AWS_S3_BUCKET/raw/" --recursive
```

### Option B: Using AWS Console

1. Go to your S3 bucket
2. Open `raw/` folder
3. Click "Upload"
4. Select files from `sample_data/`
5. Click "Upload"

### Option C: Using AWS CLI

```bash
# Upload sample data
aws s3 cp sample_data/ \
  "s3://$BUCKET_NAME/raw/" \
  --recursive

# Verify upload
aws s3 ls "s3://$BUCKET_NAME/raw/" --recursive
```

---

## Step 6: Configure S3 Event Trigger (Optional)

Auto-trigger Lambda when new files are uploaded to S3.

### Option A: Using AWS Console

1. Open S3 bucket
2. Go to `raw/` folder properties
3. Click "Create event notification"
4. Name: `lambda-trigger`
5. Events: Select "All object create events"
6. Destination: Lambda function
7. Function: `process-climate-data`
8. Click "Create event notification"

### Option B: Using AWS CLI

```bash
# Create Lambda permission for S3
aws lambda add-permission \
  --function-name process-climate-data \
  --principal s3.amazonaws.com \
  --action lambda:InvokeFunction \
  --source-arn "arn:aws:s3:::$BUCKET_NAME"

# Create S3 event notification
NOTIFICATION_JSON=$(cat <<'EOF'
{
  "LambdaFunctionConfigurations": [
    {
      "LambdaFunctionArn": "LAMBDA_ARN",
      "Events": ["s3:ObjectCreated:*"],
      "Filter": {
        "Key": {
          "FilterRules": [
            {
              "Name": "prefix",
              "Value": "raw/"
            },
            {
              "Name": "suffix",
              "Value": ".nc"
            }
          ]
        }
      }
    }
  ]
}
EOF
)

# Replace LAMBDA_ARN with your actual ARN
aws s3api put-bucket-notification-configuration \
  --bucket "$BUCKET_NAME" \
  --notification-configuration "$NOTIFICATION_JSON"
```

**Note:** S3 triggers can take 2-3 minutes to activate.

---

## Step 7: Test the Pipeline

Test that data flows through successfully.

### Test 1: Manual Lambda Invocation

```bash
# Upload test file
aws s3 cp sample_data/test_data.nc "s3://$BUCKET_NAME/raw/test_data.nc"

# Invoke Lambda manually
aws lambda invoke \
  --function-name process-climate-data \
  --payload '{"Records":[{"s3":{"bucket":{"name":"'$BUCKET_NAME'"},"object":{"key":"raw/test_data.nc"}}}]}' \
  response.json

# Check response
cat response.json

# Check results in S3
aws s3 ls "s3://$BUCKET_NAME/results/" --recursive
```

### Test 2: Check CloudWatch Logs

```bash
# Find log group
LOG_GROUP="/aws/lambda/process-climate-data"

# Get latest log streams
aws logs describe-log-streams \
  --log-group-name "$LOG_GROUP" \
  --order-by LastEventTime \
  --descending

# View logs (replace STREAM_NAME with actual stream)
aws logs get-log-events \
  --log-group-name "$LOG_GROUP" \
  --log-stream-name "STREAM_NAME"
```

### Test 3: Verify Results

```bash
# Download processed result
aws s3 cp "s3://$BUCKET_NAME/results/" ./results/ --recursive

# View results
cat results/test_data_processed.json | python -m json.tool
```

**Expected result:** JSON file with regional statistics:
```json
{
  "file": "test_data.nc",
  "timestamp": "2025-01-14T10:30:00Z",
  "statistics": {
    "temperature_mean": 15.2,
    "temperature_std": 2.1,
    "precipitation_total": 850.5
  }
}
```

---

## Step 8: Set Up Local Environment

Configure your local machine for the analysis.

### Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt

# Verify AWS CLI works
aws s3 ls
```

### Configure .env File

```bash
# Create .env file
cat > .env << 'EOF'
AWS_REGION=us-east-1
AWS_S3_BUCKET=climate-data-xxxx
AWS_LAMBDA_FUNCTION=process-climate-data
AWS_PROFILE=default
EOF

# Load environment variables (optional)
source .env
```

### Configure Credentials

```bash
# If not already configured
aws configure

# Enter:
# AWS Access Key ID: [your key]
# AWS Secret Access Key: [your secret]
# Default region: us-east-1
# Default output format: json

# Verify credentials work
aws sts get-caller-identity
```

---

## Step 9: Run Full Pipeline

Now run the complete analysis workflow.

### Script 1: Upload Data

```bash
# Upload all climate data to S3
python scripts/upload_to_s3.py

# Monitor upload
watch -n 2 'aws s3 ls s3://$BUCKET_NAME/raw/ --recursive'
```

### Script 2: Process Data

```bash
# Process files with Lambda
python scripts/lambda_function.py

# Or manually trigger:
for file in $(aws s3 ls s3://$BUCKET_NAME/raw/ --recursive | awk '{print $NF}'); do
  aws lambda invoke \
    --function-name process-climate-data \
    --payload "{\"Records\":[{\"s3\":{\"bucket\":{\"name\":\"$BUCKET_NAME\"},\"object\":{\"key\":\"$file\"}}}]}" \
    response.json
done

# Monitor Lambda logs
aws logs tail /aws/lambda/process-climate-data --follow
```

### Script 3: Query Results

```bash
# Download and analyze results
python scripts/query_results.py

# Results saved to ./results/
ls -lah results/
```

### Script 4: Jupyter Analysis

```bash
# Launch Jupyter notebook
jupyter notebook notebooks/climate_analysis.ipynb

# Run all cells for full analysis
```

---

## Step 10: Monitor Costs

Track spending to avoid surprises.

### Set Billing Alerts

1. Go to [AWS Billing Console](https://console.aws.amazon.com/billing/)
2. Click "Billing Preferences"
3. Enable "Receive Billing Alerts"
4. Create budget:
   - Name: `Climate Tier 2 Budget`
   - Budget type: Monthly
   - Limit: $20
   - Alert threshold: 80% and 100%

### Check Current Costs

```bash
# View costs for current month
aws ce get-cost-and-usage \
  --time-period Start=$(date -d '30 days ago' +%Y-%m-%d),End=$(date +%Y-%m-%d) \
  --granularity DAILY \
  --metrics "UnblendedCost" \
  --group-by Type=DIMENSION,Key=SERVICE

# Or use AWS Cost Explorer GUI
# https://console.aws.amazon.com/cost-management/
```

### Cost Optimization

1. **Delete unused data**: `aws s3 rm s3://$BUCKET_NAME/raw/ --recursive` ($0.35 savings)
2. **Use S3 Intelligent-Tiering**: Automatic cost optimization
3. **Set Lambda timeout to 5 min**: Prevents runaway costs
4. **Clean up Lambda versions**: Only keep latest
5. **Use spot instances**: 60-70% cheaper (future upgrade)

---

## Troubleshooting

### Problem: "Access Denied" when creating S3 bucket

**Cause:** IAM permissions issue

**Solution:**
```bash
# Check your AWS user permissions
aws iam get-user

# Ensure you have S3 full access
aws iam attach-user-policy \
  --user-name your-username \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

### Problem: Lambda timeout errors

**Cause:** Files too large or timeout too short

**Solution:**
```bash
# Increase timeout to 10 minutes
aws lambda update-function-configuration \
  --function-name process-climate-data \
  --timeout 600

# Or increase memory allocation
aws lambda update-function-configuration \
  --function-name process-climate-data \
  --memory-size 1024
```

### Problem: "botocore.exceptions.NoCredentialsError"

**Cause:** AWS credentials not configured

**Solution:**
```bash
# Configure credentials
aws configure

# Or set environment variables
export AWS_ACCESS_KEY_ID=your_key_id
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

### Problem: S3 trigger not working

**Cause:** Lambda permission not set or S3 notification not configured

**Solution:**
```bash
# 1. Add Lambda permission
aws lambda add-permission \
  --function-name process-climate-data \
  --principal s3.amazonaws.com \
  --action lambda:InvokeFunction \
  --source-arn arn:aws:s3:::$BUCKET_NAME

# 2. Verify S3 notification
aws s3api get-bucket-notification-configuration \
  --bucket $BUCKET_NAME
```

### Problem: High AWS costs

**Cause:** Lambda running too long, data transfer, or unterminated resources

**Solution:**
```bash
# 1. Check Lambda duration
aws logs get-log-events \
  --log-group-name /aws/lambda/process-climate-data \
  --log-stream-name 'latest' \
  | grep Duration

# 2. Check S3 data size
aws s3api list-object-versions \
  --bucket $BUCKET_NAME \
  --output json | python -c \
  "import json,sys; data=json.load(sys.stdin); \
   print(f'Size: {sum(v.get(\"Size\",0) for v in data.get(\"Versions\",[]))/1e9:.2f}GB')"

# 3. Delete unused resources
python cleanup.py
```

---

## Security Best Practices

### Least Privilege Access

Your IAM role should only have permissions it needs:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject"
      ],
      "Resource": "arn:aws:s3:::climate-data-*/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*"
    }
  ]
}
```

### Bucket Encryption

```bash
# Enable default encryption on bucket
aws s3api put-bucket-encryption \
  --bucket "$BUCKET_NAME" \
  --server-side-encryption-configuration '{
    "Rules": [{
      "ApplyServerSideEncryptionByDefault": {
        "SSEAlgorithm": "AES256"
      }
    }]
  }'
```

### Block Public Access

```bash
# Block all public access
aws s3api put-public-access-block \
  --bucket "$BUCKET_NAME" \
  --public-access-block-configuration \
  "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
```

---

## Next Steps

After successful setup:

1. ✅ Review `notebooks/climate_analysis.ipynb` for analysis workflows
2. ✅ Read `cleanup_guide.md` for resource deletion
3. ✅ Explore extending the project (more models, variables, etc.)
4. ✅ Check AWS Cost Explorer regularly
5. ✅ Move to Tier 3 for production infrastructure

---

## Quick Reference

### Key Commands

```bash
# List your S3 buckets
aws s3 ls

# Upload file to S3
aws s3 cp file.nc s3://bucket-name/folder/

# Download from S3
aws s3 cp s3://bucket-name/file output.json

# Monitor Lambda logs
aws logs tail /aws/lambda/process-climate-data --follow

# Check Lambda metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/Lambda \
  --metric-name Duration \
  --dimensions Name=FunctionName,Value=process-climate-data \
  --start-time 2025-01-14T00:00:00Z \
  --end-time 2025-01-15T00:00:00Z \
  --period 3600 \
  --statistics Average,Maximum

# List IAM roles
aws iam list-roles

# Get bucket size
aws s3api list-object-versions \
  --bucket climate-data-xxxx \
  --output json | python -c \
  "import json,sys; d=json.load(sys.stdin); \
   print(f'{sum(v.get(\"Size\",0) for v in d.get(\"Versions\",[]))/1e9:.2f}GB')"
```

---

## Support

- **Documentation:** See README.md
- **Issues:** https://github.com/research-jumpstart/research-jumpstart/issues
- **AWS Support:** https://console.aws.amazon.com/support/

---

**Next:** Follow the main workflow in [README.md](README.md) or clean up with [cleanup_guide.md](cleanup_guide.md)
