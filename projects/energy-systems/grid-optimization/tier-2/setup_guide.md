# AWS Setup Guide - Smart Grid Optimization Tier 2

This guide walks you through setting up all AWS resources needed for the smart grid optimization project. Estimated time: 30-45 minutes.

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

S3 stores your grid data, processed results, and Lambda function logs.

### Option A: Using AWS Console (Recommended for beginners)

1. Go to [AWS S3 Console](https://console.aws.amazon.com/s3/)
2. Click "Create bucket"
3. Enter bucket name: `energy-grid-{your-name}-{date}`
   - Example: `energy-grid-alice-20250114`
   - Must be globally unique
   - Must be lowercase
   - No special characters except hyphens
4. Select region: **us-east-1** (recommended for lowest cost)
5. Keep all default settings (Block all public access: ON)
6. Click "Create bucket"

### Option B: Using AWS CLI

```bash
# Create S3 bucket with unique name
BUCKET_NAME="energy-grid-$(whoami)-$(date +%s)"
aws s3 mb "s3://$BUCKET_NAME" --region us-east-1

# Verify bucket created
aws s3 ls | grep energy-grid

# Save bucket name for later
echo "BUCKET_NAME=$BUCKET_NAME" > .env
```

### Verify Bucket Created

```bash
# List your buckets
aws s3 ls

# Should see your new energy-grid-xxxx bucket
```

**Save your bucket name!** You'll use it in later steps.

---

## Step 2: Create S3 Folder Structure

Lambda will expect data organized in specific folders.

### Using AWS Console

1. Open your energy-grid bucket
2. Create folders:
   - Click "Create folder" → name it `raw` → Create
   - Click "Create folder" → name it `results` → Create
   - Click "Create folder" → name it `logs` → Create

### Using AWS CLI

```bash
# Create folder structure (using empty objects)
aws s3api put-object --bucket "$BUCKET_NAME" --key "raw/"
aws s3api put-object --bucket "$BUCKET_NAME" --key "results/"
aws s3api put-object --bucket "$BUCKET_NAME" --key "logs/"

# Verify folders created
aws s3 ls "s3://$BUCKET_NAME/"
```

---

## Step 3: Create DynamoDB Table

DynamoDB stores grid analysis metrics for fast queries.

### Option A: Using AWS Console

1. Go to [DynamoDB Console](https://console.aws.amazon.com/dynamodb/)
2. Click "Create table"
3. Table name: `GridAnalysis`
4. Partition key: `location` (String)
5. Sort key: `timestamp` (String)
6. Table settings: **Default settings**
7. Read/write capacity: **On-demand**
8. Click "Create table"

### Option B: Using AWS CLI

```bash
# Create DynamoDB table with on-demand billing
aws dynamodb create-table \
  --table-name GridAnalysis \
  --attribute-definitions \
    AttributeName=location,AttributeType=S \
    AttributeName=timestamp,AttributeType=S \
  --key-schema \
    AttributeName=location,KeyType=HASH \
    AttributeName=timestamp,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST \
  --region us-east-1

# Wait for table to be active (takes ~30 seconds)
aws dynamodb wait table-exists --table-name GridAnalysis

# Verify table created
aws dynamodb describe-table --table-name GridAnalysis \
  --query 'Table.[TableName,TableStatus]' \
  --output table
```

### Verify Table Created

```bash
# List DynamoDB tables
aws dynamodb list-tables

# Should see GridAnalysis in the list
```

---

## Step 4: Create SNS Topic

SNS sends email alerts for grid anomalies.

### Option A: Using AWS Console

1. Go to [SNS Console](https://console.aws.amazon.com/sns/)
2. Click "Topics" in left menu
3. Click "Create topic"
4. Type: **Standard**
5. Name: `grid-anomaly-alerts`
6. Display name: `Grid Anomaly Alerts`
7. Click "Create topic"
8. **Subscribe to topic:**
   - Click "Create subscription"
   - Protocol: **Email**
   - Endpoint: `your-email@example.com`
   - Click "Create subscription"
9. **Confirm subscription:**
   - Check your email
   - Click confirmation link

### Option B: Using AWS CLI

```bash
# Create SNS topic
TOPIC_ARN=$(aws sns create-topic \
  --name grid-anomaly-alerts \
  --region us-east-1 \
  --query 'TopicArn' \
  --output text)

echo "TOPIC_ARN=$TOPIC_ARN" >> .env

# Subscribe your email
aws sns subscribe \
  --topic-arn "$TOPIC_ARN" \
  --protocol email \
  --notification-endpoint your-email@example.com

# Check your email and confirm subscription!
echo "Check your email and confirm SNS subscription"
```

### Verify SNS Topic

```bash
# List SNS topics
aws sns list-topics

# List subscriptions
aws sns list-subscriptions-by-topic --topic-arn "$TOPIC_ARN"
```

---

## Step 5: Create IAM Role for Lambda

Lambda needs permissions to access S3, DynamoDB, SNS, and CloudWatch.

### Option A: Using AWS Console

1. Go to [IAM Console](https://console.aws.amazon.com/iam/home)
2. Click "Roles" in left menu
3. Click "Create role"
4. Trusted entity: **Lambda**
5. Click "Next"
6. Search for and select these policies:
   - `AWSLambdaBasicExecutionRole` (CloudWatch logs)
   - `AmazonS3FullAccess` (S3 read/write)
   - `AmazonDynamoDBFullAccess` (DynamoDB read/write)
   - `AmazonSNSFullAccess` (SNS publish)
7. Click "Next"
8. Role name: `lambda-grid-optimizer`
9. Description: `Lambda role for grid optimization with S3, DynamoDB, SNS access`
10. Click "Create role"

### Option B: Using AWS CLI

```bash
# Create trust policy for Lambda
TRUST_POLICY=$(cat <<'EOF'
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

# Create IAM role
aws iam create-role \
  --role-name lambda-grid-optimizer \
  --assume-role-policy-document "$TRUST_POLICY" \
  --description "Lambda role for grid optimization"

# Attach managed policies
aws iam attach-role-policy \
  --role-name lambda-grid-optimizer \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam attach-role-policy \
  --role-name lambda-grid-optimizer \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam attach-role-policy \
  --role-name lambda-grid-optimizer \
  --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess

aws iam attach-role-policy \
  --role-name lambda-grid-optimizer \
  --policy-arn arn:aws:iam::aws:policy/AmazonSNSFullAccess

# Get role ARN (you'll need this for Lambda)
ROLE_ARN=$(aws iam get-role --role-name lambda-grid-optimizer \
  --query 'Role.Arn' --output text)
echo "ROLE_ARN=$ROLE_ARN" >> .env
```

### Verify Role Created

```bash
# Get role details
aws iam get-role --role-name lambda-grid-optimizer

# List attached policies
aws iam list-attached-role-policies --role-name lambda-grid-optimizer
```

**Save your role ARN!** Format: `arn:aws:iam::ACCOUNT_ID:role/lambda-grid-optimizer`

---

## Step 6: Create Lambda Function

Lambda processes grid data and performs optimization analysis.

### Option A: Using AWS Console

1. Go to [Lambda Console](https://console.aws.amazon.com/lambda/)
2. Click "Create function"
3. Choose "Author from scratch"
4. Function name: `optimize-energy-grid`
5. Runtime: **Python 3.11**
6. Architecture: **x86_64**
7. Permissions: **Use an existing role**
   - Select `lambda-grid-optimizer`
8. Click "Create function"

#### Upload Function Code:

1. In function page, scroll to "Code source"
2. Click "Upload from" → ".zip file"
3. **First, create the zip file locally:**
   ```bash
   cd scripts
   zip lambda_function.zip lambda_function.py
   cd ..
   ```
4. Upload `scripts/lambda_function.zip`
5. Click "Save"

#### Configure Function Settings:

1. Go to "Configuration" tab
2. Click "General configuration" → "Edit"
3. **Memory:** 512 MB
4. **Timeout:** 300 seconds (5 minutes)
5. Click "Save"

#### Set Environment Variables:

1. In Configuration tab → "Environment variables"
2. Click "Edit"
3. Add variables:
   - `BUCKET_NAME` = `energy-grid-xxxx` (your bucket)
   - `DYNAMODB_TABLE` = `GridAnalysis`
   - `SNS_TOPIC_ARN` = `arn:aws:sns:...` (your topic ARN)
4. Click "Save"

### Option B: Using AWS CLI

```bash
# Create deployment package
cd scripts
zip lambda_function.zip lambda_function.py
cd ..

# Wait for IAM role to propagate (10-30 seconds)
sleep 15

# Create Lambda function
aws lambda create-function \
  --function-name optimize-energy-grid \
  --runtime python3.11 \
  --role "$ROLE_ARN" \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://scripts/lambda_function.zip \
  --timeout 300 \
  --memory-size 512 \
  --environment Variables="{BUCKET_NAME=$BUCKET_NAME,DYNAMODB_TABLE=GridAnalysis,SNS_TOPIC_ARN=$TOPIC_ARN}" \
  --region us-east-1

# Verify function created
aws lambda get-function --function-name optimize-energy-grid
```

### Test Lambda Function

```bash
# Create test event
TEST_EVENT=$(cat <<EOF
{
  "Records": [{
    "s3": {
      "bucket": {"name": "$BUCKET_NAME"},
      "object": {"key": "raw/test_data.csv"}
    }
  }]
}
EOF
)

# Invoke Lambda (will fail if test_data.csv doesn't exist, but tests deployment)
aws lambda invoke \
  --function-name optimize-energy-grid \
  --payload "$TEST_EVENT" \
  --cli-binary-format raw-in-base64-out \
  response.json

# View response
cat response.json
```

---

## Step 7: Configure S3 Event Trigger (Optional)

Automatically trigger Lambda when new grid data is uploaded to S3.

### Option A: Using AWS Console

1. Open Lambda function `optimize-energy-grid`
2. Click "Add trigger"
3. Select trigger: **S3**
4. Bucket: `energy-grid-xxxx`
5. Event type: **All object create events**
6. Prefix: `raw/`
7. Suffix: `.csv`
8. Click "Add"

### Option B: Using AWS CLI

```bash
# Add Lambda permission for S3 to invoke
aws lambda add-permission \
  --function-name optimize-energy-grid \
  --principal s3.amazonaws.com \
  --action lambda:InvokeFunction \
  --source-arn "arn:aws:s3:::$BUCKET_NAME" \
  --statement-id s3-trigger-permission

# Create S3 notification configuration
NOTIFICATION_CONFIG=$(cat <<EOF
{
  "LambdaFunctionConfigurations": [{
    "LambdaFunctionArn": "$(aws lambda get-function --function-name optimize-energy-grid --query 'Configuration.FunctionArn' --output text)",
    "Events": ["s3:ObjectCreated:*"],
    "Filter": {
      "Key": {
        "FilterRules": [
          {"Name": "prefix", "Value": "raw/"},
          {"Name": "suffix", "Value": ".csv"}
        ]
      }
    }
  }]
}
EOF
)

# Apply notification configuration
aws s3api put-bucket-notification-configuration \
  --bucket "$BUCKET_NAME" \
  --notification-configuration "$NOTIFICATION_CONFIG"

# Verify notification configured
aws s3api get-bucket-notification-configuration --bucket "$BUCKET_NAME"
```

---

## Step 8: Set Up Athena (Optional)

Athena allows SQL queries on S3 data.

### Using AWS Console

1. Go to [Athena Console](https://console.aws.amazon.com/athena/)
2. Click "Get Started"
3. **Set up query result location:**
   - Click "Settings"
   - Query result location: `s3://energy-grid-xxxx/athena-results/`
   - Click "Save"
4. **Create database:**
   ```sql
   CREATE DATABASE energy_grid_db;
   ```
5. **Create table for grid data:**
   ```sql
   CREATE EXTERNAL TABLE energy_grid_db.grid_data (
     timestamp STRING,
     location STRING,
     load_mw DOUBLE,
     generation_mw DOUBLE,
     voltage_kv DOUBLE,
     frequency_hz DOUBLE,
     solar_mw DOUBLE,
     wind_mw DOUBLE
   )
   ROW FORMAT DELIMITED
   FIELDS TERMINATED BY ','
   STORED AS TEXTFILE
   LOCATION 's3://energy-grid-xxxx/raw/';
   ```

### Using AWS CLI

```bash
# Create Athena result location
aws s3api put-object --bucket "$BUCKET_NAME" --key "athena-results/"

# Note: Table creation requires running SQL via console or SDK
echo "Create Athena tables using AWS Console or see notebooks/grid_analysis.ipynb"
```

---

## Step 9: Upload Sample Data

Test the pipeline with sample grid data.

### Option A: Using Python Script (Recommended)

```bash
# Set environment variable
export AWS_S3_BUCKET="energy-grid-xxxx"

# Run upload script (generates and uploads sample data)
python scripts/upload_to_s3.py --bucket "$AWS_S3_BUCKET" --generate

# Verify upload
aws s3 ls "s3://$AWS_S3_BUCKET/raw/" --recursive
```

### Option B: Using AWS Console

1. Go to your S3 bucket
2. Open `raw/` folder
3. Click "Upload"
4. Create a CSV file locally first:
   ```csv
   timestamp,location,load_mw,generation_mw,voltage_kv,frequency_hz,solar_mw,wind_mw
   2025-01-14T00:00:00,substation_001,125.5,130.2,13.8,60.01,15.3,8.7
   2025-01-14T00:15:00,substation_001,128.3,132.1,13.79,60.00,12.1,9.2
   ```
5. Upload the file
6. Click "Upload"

---

## Step 10: Test the Complete Pipeline

Verify all components work together.

### Test 1: Manual Lambda Invocation

```bash
# Upload test data
python scripts/upload_to_s3.py --bucket "$AWS_S3_BUCKET" --generate

# If S3 trigger is configured, Lambda runs automatically
# Monitor CloudWatch logs:
aws logs tail /aws/lambda/optimize-energy-grid --follow

# Or invoke manually:
aws lambda invoke \
  --function-name optimize-energy-grid \
  --payload '{"Records":[{"s3":{"bucket":{"name":"'$BUCKET_NAME'"},"object":{"key":"raw/grid_data_sample.csv"}}}]}' \
  --cli-binary-format raw-in-base64-out \
  response.json

cat response.json
```

### Test 2: Check DynamoDB Results

```bash
# Scan DynamoDB table
aws dynamodb scan --table-name GridAnalysis --max-items 10

# Query specific location
aws dynamodb query \
  --table-name GridAnalysis \
  --key-condition-expression "location = :loc" \
  --expression-attribute-values '{":loc":{"S":"substation_001"}}'
```

### Test 3: Check S3 Results

```bash
# List processed results
aws s3 ls "s3://$BUCKET_NAME/results/" --recursive

# Download a result file
aws s3 cp "s3://$BUCKET_NAME/results/grid_data_sample_analysis.json" ./test_result.json

# View results
cat test_result.json | python -m json.tool
```

### Test 4: Verify SNS Alert

```bash
# Check if alert was sent (check your email)
# Or list SNS messages:
aws sns publish \
  --topic-arn "$TOPIC_ARN" \
  --message "Test: Grid optimization pipeline is working!" \
  --subject "Grid Alert Test"

# Check your email for the test message
```

---

## Step 11: Set Up Local Environment

Configure your local machine for analysis.

### Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt

# Verify installations
python -c "import boto3; print('boto3:', boto3.__version__)"
python -c "import pandas; print('pandas:', pandas.__version__)"
```

### Configure AWS Credentials

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

### Create .env File

```bash
# Create .env file with your resource names
cat > .env << EOF
AWS_REGION=us-east-1
AWS_S3_BUCKET=energy-grid-xxxx
AWS_LAMBDA_FUNCTION=optimize-energy-grid
AWS_DYNAMODB_TABLE=GridAnalysis
AWS_SNS_TOPIC_ARN=arn:aws:sns:us-east-1:ACCOUNT:grid-anomaly-alerts
AWS_PROFILE=default
EOF

# Note: Replace xxxx and ACCOUNT with your actual values
```

---

## Step 12: Run Full Analysis Pipeline

Execute the complete workflow.

### Option A: Using Jupyter Notebook (Recommended)

```bash
# Launch Jupyter
jupyter notebook notebooks/grid_analysis.ipynb

# Run all cells in the notebook
# The notebook will:
# 1. Generate sample grid data
# 2. Upload to S3
# 3. Trigger Lambda processing
# 4. Query results from DynamoDB
# 5. Visualize grid metrics
```

### Option B: Using Scripts

```bash
# Step 1: Upload data
python scripts/upload_to_s3.py --bucket energy-grid-xxxx --generate

# Step 2: Wait for Lambda processing (if auto-trigger enabled)
# Or manually trigger:
python scripts/lambda_function.py --test

# Step 3: Query results
python scripts/query_results.py --bucket energy-grid-xxxx

# Results will be saved to ./results/ directory
```

---

## Step 13: Monitor Costs

Set up cost tracking to avoid surprises.

### Set Billing Alerts

1. Go to [AWS Billing Console](https://console.aws.amazon.com/billing/)
2. Click "Billing Preferences"
3. Enable "Receive Billing Alerts"
4. Click "Budgets" → "Create budget"
5. Budget details:
   - Name: `Energy Grid Tier 2 Budget`
   - Budget type: Monthly
   - Limit: $20
   - Alert thresholds: 50%, 80%, 100%
   - Email: your-email@example.com

### Check Current Costs

```bash
# View costs for last 7 days
aws ce get-cost-and-usage \
  --time-period Start=$(date -v-7d +%Y-%m-%d),End=$(date +%Y-%m-%d) \
  --granularity DAILY \
  --metrics "UnblendedCost" \
  --group-by Type=DIMENSION,Key=SERVICE

# Or use AWS Cost Explorer GUI
# https://console.aws.amazon.com/cost-management/
```

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

**Cause:** Processing takes longer than timeout setting

**Solution:**
```bash
# Increase timeout to 10 minutes
aws lambda update-function-configuration \
  --function-name optimize-energy-grid \
  --timeout 600

# Also increase memory (more memory = more CPU)
aws lambda update-function-configuration \
  --function-name optimize-energy-grid \
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

### Problem: DynamoDB write errors

**Cause:** Table doesn't exist or permissions issue

**Solution:**
```bash
# Verify table exists
aws dynamodb describe-table --table-name GridAnalysis

# If doesn't exist, create it (see Step 3)
# If exists, check Lambda role has DynamoDB permissions
aws iam list-attached-role-policies --role-name lambda-grid-optimizer
```

### Problem: SNS emails not received

**Cause:** Subscription not confirmed

**Solution:**
```bash
# Resend confirmation
aws sns subscribe \
  --topic-arn "$TOPIC_ARN" \
  --protocol email \
  --notification-endpoint your-email@example.com

# Check spam folder
# Click confirmation link in email
```

### Problem: S3 trigger not working

**Cause:** Lambda permission not set

**Solution:**
```bash
# Add Lambda permission for S3
aws lambda add-permission \
  --function-name optimize-energy-grid \
  --principal s3.amazonaws.com \
  --action lambda:InvokeFunction \
  --source-arn "arn:aws:s3:::$BUCKET_NAME" \
  --statement-id s3-invoke-permission

# Verify notification configuration
aws s3api get-bucket-notification-configuration --bucket "$BUCKET_NAME"
```

---

## Security Best Practices

### Least Privilege IAM Policy

Instead of using managed policies, create a custom policy with minimum permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject"
      ],
      "Resource": "arn:aws:s3:::energy-grid-*/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:PutItem",
        "dynamodb:GetItem",
        "dynamodb:Query"
      ],
      "Resource": "arn:aws:dynamodb:*:*:table/GridAnalysis"
    },
    {
      "Effect": "Allow",
      "Action": "sns:Publish",
      "Resource": "arn:aws:sns:*:*:grid-anomaly-alerts"
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

### Enable S3 Encryption

```bash
# Enable default encryption
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
# Ensure bucket is private
aws s3api put-public-access-block \
  --bucket "$BUCKET_NAME" \
  --public-access-block-configuration \
  "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
```

---

## Next Steps

After successful setup:

1. ✅ Review `notebooks/grid_analysis.ipynb` for analysis workflows
2. ✅ Read `cleanup_guide.md` for resource deletion
3. ✅ Explore extending the project with more grid metrics
4. ✅ Check AWS Cost Explorer regularly
5. ✅ Move to Tier 3 for production infrastructure

---

## Quick Reference

### Key Commands

```bash
# List S3 buckets
aws s3 ls

# Upload to S3
aws s3 cp file.csv s3://bucket-name/raw/

# Invoke Lambda
aws lambda invoke --function-name optimize-energy-grid response.json

# Query DynamoDB
aws dynamodb scan --table-name GridAnalysis

# Monitor Lambda logs
aws logs tail /aws/lambda/optimize-energy-grid --follow

# Check costs
aws ce get-cost-and-usage --time-period Start=2025-01-01,End=2025-01-31 \
  --granularity MONTHLY --metrics "BlendedCost"
```

---

## Support

- **Documentation:** See README.md
- **Issues:** https://github.com/research-jumpstart/research-jumpstart/issues
- **AWS Support:** https://console.aws.amazon.com/support/

---

**Next:** Follow the main workflow in [README.md](README.md) or start with [scripts/upload_to_s3.py](scripts/upload_to_s3.py)
