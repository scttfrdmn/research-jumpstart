# AWS Setup Guide for Transportation Flow Analysis Pipeline

This guide provides step-by-step instructions to set up the AWS environment for the Transportation Optimization Tier 2 project.

**Total Setup Time:** 40-50 minutes

## Prerequisites

Before starting, ensure you have:
- [ ] AWS Account (with payment method registered)
- [ ] AWS CLI installed and configured
- [ ] Python 3.8+ installed
- [ ] Access to AWS Management Console
- [ ] Basic understanding of AWS services

### Install AWS CLI

If you haven't installed AWS CLI:

```bash
# macOS (using Homebrew)
brew install awscli

# Or using pip
pip install awscli

# Verify installation
aws --version
```

### Configure AWS Credentials

1. Log in to AWS Management Console
2. Go to **IAM** > **Users** > **Your User** > **Security Credentials**
3. Create an access key (if you don't have one)
4. Run:

```bash
aws configure
```

You'll be prompted for:
- AWS Access Key ID
- AWS Secret Access Key
- Default region (use `us-east-1`)
- Default output format (use `json`)

Verify configuration:
```bash
aws sts get-caller-identity
```

You should see your account details.

---

## Step 1: Create S3 Bucket

The S3 bucket will store raw traffic data and processed results.

### Using AWS Management Console

1. Go to **S3** > **Buckets** > **Create Bucket**
2. Enter bucket name: `transportation-data-{your-user-id}`
   - **Important:** Bucket names must be globally unique
   - Use lowercase letters and hyphens only
   - Example: `transportation-data-scttfrdmn-001`
3. Region: Select your region (e.g., `us-east-1`)
4. **Block Public Access:** Keep all settings enabled (default)
5. **Bucket Versioning:** Enable (optional, for data recovery)
6. Click **Create Bucket**

### Using AWS CLI

```bash
# Replace {your-user-id} with your name/ID
aws s3 mb s3://transportation-data-{your-user-id} --region us-east-1

# Verify bucket creation
aws s3 ls | grep transportation-data
```

### Create Bucket Folders

Create folder structure:

```bash
# Raw traffic data folder
aws s3api put-object --bucket transportation-data-{your-user-id} --key raw/

# Processed results folder
aws s3api put-object --bucket transportation-data-{your-user-id} --key results/

# Logs folder
aws s3api put-object --bucket transportation-data-{your-user-id} --key logs/

# Verify folder structure
aws s3 ls s3://transportation-data-{your-user-id}/
```

---

## Step 2: Create DynamoDB Table

The DynamoDB table stores traffic analysis results for fast queries.

### Using AWS Management Console

1. Go to **DynamoDB** > **Tables** > **Create Table**
2. Table name: `TrafficAnalysis`
3. Partition Key: `segment_id` (String)
4. Sort Key: `timestamp` (Number)
5. Table Settings: **Customize settings**
6. Billing Mode: **On-demand** (auto-scales, easier for learning)
7. Secondary Indexes: None (we'll add via queries if needed)
8. Click **Create Table**
9. Wait for table to become ACTIVE (1-2 minutes)

### Using AWS CLI

```bash
aws dynamodb create-table \
    --table-name TrafficAnalysis \
    --attribute-definitions \
        AttributeName=segment_id,AttributeType=S \
        AttributeName=timestamp,AttributeType=N \
    --key-schema \
        AttributeName=segment_id,KeyType=HASH \
        AttributeName=timestamp,KeyType=RANGE \
    --billing-mode PAY_PER_REQUEST \
    --region us-east-1

# Verify table creation
aws dynamodb list-tables --region us-east-1
```

### Create Global Secondary Index (Optional)

For querying by congestion level:

```bash
aws dynamodb update-table \
    --table-name TrafficAnalysis \
    --attribute-definitions \
        AttributeName=congestion_level,AttributeType=N \
        AttributeName=timestamp,AttributeType=N \
    --global-secondary-index-updates \
    '[{
        "Create": {
            "IndexName": "congestion-index",
            "KeySchema": [
                {"AttributeName": "congestion_level", "KeyType": "HASH"},
                {"AttributeName": "timestamp", "KeyType": "RANGE"}
            ],
            "Projection": {"ProjectionType": "ALL"}
        }
    }]'
```

### Verify Table

```bash
aws dynamodb describe-table --table-name TrafficAnalysis --region us-east-1
```

Check that status is `ACTIVE`.

---

## Step 3: Create IAM Role for Lambda

Lambda needs permissions to access S3 and DynamoDB.

### Using AWS Management Console

1. Go to **IAM** > **Roles** > **Create Role**
2. Trusted entity type: **AWS service**
3. Use case: **Lambda**
4. Click **Next**
5. Add permissions:
   - Search for and select: `AmazonS3FullAccess`
   - Search for and select: `AmazonDynamoDBFullAccess`
   - Search for and select: `CloudWatchLogsFullAccess`
   - Click **Next**
6. Role name: `lambda-traffic-processor`
7. Description: `Lambda role for traffic flow analysis`
8. Click **Create Role**

### Using AWS CLI

```bash
# Create trust policy file
cat > trust-policy.json << 'EOF'
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
    --role-name lambda-traffic-processor \
    --assume-role-policy-document file://trust-policy.json

# Attach policies
aws iam attach-role-policy \
    --role-name lambda-traffic-processor \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam attach-role-policy \
    --role-name lambda-traffic-processor \
    --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess

aws iam attach-role-policy \
    --role-name lambda-traffic-processor \
    --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess

# Get role ARN (you'll need this for Lambda)
aws iam get-role --role-name lambda-traffic-processor --query 'Role.Arn'
```

**Save the role ARN** - you'll need it for Lambda deployment.

Example: `arn:aws:iam::123456789012:role/lambda-traffic-processor`

---

## Step 4: Deploy Lambda Function

### Create Lambda Function

1. Go to **Lambda** > **Functions** > **Create Function**
2. Choose: **Author from scratch**
3. Function name: `analyze-traffic-flow`
4. Runtime: **Python 3.11** (or latest)
5. Architecture: **x86_64**
6. Execution role: **Use an existing role**
7. Existing role: Select `lambda-traffic-processor` from dropdown
8. Click **Create Function**

### Upload Code

In the Lambda console:

1. Go to the **Code** tab
2. Copy the code from `scripts/lambda_function.py`
3. Paste into the Lambda editor (lambda_function.py)
4. Click **Deploy**

Or using AWS CLI:

```bash
# Navigate to scripts directory
cd scripts/

# Zip the lambda function
zip lambda_function.zip lambda_function.py

# Deploy to Lambda (replace YOUR_ACCOUNT_ID with your AWS account ID)
aws lambda create-function \
    --function-name analyze-traffic-flow \
    --runtime python3.11 \
    --role arn:aws:iam::YOUR_ACCOUNT_ID:role/lambda-traffic-processor \
    --handler lambda_function.lambda_handler \
    --zip-file fileb://lambda_function.zip \
    --timeout 300 \
    --memory-size 512 \
    --region us-east-1

# If function already exists, update it
aws lambda update-function-code \
    --function-name analyze-traffic-flow \
    --zip-file fileb://lambda_function.zip
```

### Configure Lambda Settings

1. Go to **Configuration** tab
2. Set:
   - **Timeout:** 300 seconds (5 minutes)
   - **Memory:** 512 MB (sufficient for traffic data processing)
   - **Ephemeral storage:** 512 MB

```bash
# Using AWS CLI
aws lambda update-function-configuration \
    --function-name analyze-traffic-flow \
    --timeout 300 \
    --memory-size 512 \
    --ephemeral-storage Size=512
```

### Add Environment Variables

In Lambda console, add environment variables:

1. Go to **Configuration** > **Environment variables**
2. Add:
   - `DYNAMODB_TABLE`: `TrafficAnalysis`
   - `S3_BUCKET`: `transportation-data-{your-user-id}`
   - `RESULTS_PREFIX`: `results/`

```bash
# Using AWS CLI
aws lambda update-function-configuration \
    --function-name analyze-traffic-flow \
    --environment "Variables={DYNAMODB_TABLE=TrafficAnalysis,S3_BUCKET=transportation-data-{your-user-id},RESULTS_PREFIX=results/}"
```

### Test Lambda Function

Create a test event in the Lambda console or use CLI:

```bash
# Create test event file
cat > test-event.json << 'EOF'
{
  "Records": [
    {
      "s3": {
        "bucket": {
          "name": "transportation-data-{your-user-id}"
        },
        "object": {
          "key": "raw/traffic_sample.csv"
        }
      }
    }
  ]
}
EOF

# Test function (after uploading test data)
# aws lambda invoke --function-name analyze-traffic-flow --payload file://test-event.json response.json
```

---

## Step 5: Set Up S3 Event Notifications

Configure S3 to automatically trigger Lambda when traffic data is uploaded.

### Using AWS Management Console

1. Go to **S3** > Your bucket > **Properties** > **Event notifications**
2. Click **Create event notification**
3. Event name: `trigger-traffic-analysis`
4. Prefix: `raw/`
5. Events: Select **All object create events** (s3:ObjectCreated:*)
6. Destination: **Lambda function**
7. Lambda function: Select `analyze-traffic-flow`
8. Click **Save changes**

### Using AWS CLI

```bash
# Create Lambda permission to be invoked by S3
aws lambda add-permission \
    --function-name analyze-traffic-flow \
    --statement-id AllowS3Invoke \
    --action lambda:InvokeFunction \
    --principal s3.amazonaws.com \
    --source-arn arn:aws:s3:::transportation-data-{your-user-id}

# Configure S3 bucket notification (replace YOUR_ACCOUNT_ID)
cat > s3-event-config.json << 'EOF'
{
  "LambdaFunctionConfigurations": [
    {
      "LambdaFunctionArn": "arn:aws:lambda:us-east-1:YOUR_ACCOUNT_ID:function:analyze-traffic-flow",
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
              "Value": ".csv"
            }
          ]
        }
      }
    }
  ]
}
EOF

aws s3api put-bucket-notification-configuration \
    --bucket transportation-data-{your-user-id} \
    --notification-configuration file://s3-event-config.json

# Verify notification configuration
aws s3api get-bucket-notification-configuration \
    --bucket transportation-data-{your-user-id}
```

---

## Step 6: Set Up Athena (Optional)

Athena allows SQL queries on S3 data without loading into a database.

### Create Athena Database

1. Go to **Athena** console
2. Choose **Query editor**
3. Click **Settings** > **Manage**
4. Set query result location: `s3://transportation-data-{your-user-id}/athena-results/`
5. Click **Save**

Run this query to create database:

```sql
CREATE DATABASE IF NOT EXISTS traffic_db;
```

### Create External Table

```sql
CREATE EXTERNAL TABLE IF NOT EXISTS traffic_db.traffic_data (
    timestamp STRING,
    segment_id STRING,
    latitude DOUBLE,
    longitude DOUBLE,
    vehicle_count INT,
    avg_speed DOUBLE,
    occupancy DOUBLE,
    congestion_level INT
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION 's3://transportation-data-{your-user-id}/raw/'
TBLPROPERTIES ('skip.header.line.count'='1');
```

### Test Query

```sql
SELECT
    segment_id,
    AVG(avg_speed) as avg_speed,
    AVG(vehicle_count) as avg_count,
    AVG(congestion_level) as avg_congestion
FROM traffic_db.traffic_data
GROUP BY segment_id
ORDER BY avg_congestion DESC
LIMIT 10;
```

---

## Step 7: Install Python Dependencies

### Create Virtual Environment (Recommended)

```bash
# Navigate to project directory
cd /path/to/transportation-optimization/tier-2

# Create virtual environment
python3 -m venv venv

# Activate it
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Verify installation:
```bash
python -c "import boto3; print(f'boto3 version: {boto3.__version__}')"
python -c "import pandas; print(f'pandas version: {pandas.__version__}')"
python -c "import networkx; print(f'networkx version: {networkx.__version__}')"
```

---

## Step 8: Environment Variables Setup

Create a `.env` file with your AWS configuration:

```bash
cd /path/to/tier-2/

cat > .env << 'EOF'
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here

# S3 Configuration
S3_BUCKET_NAME=transportation-data-{your-user-id}
S3_RAW_PREFIX=raw/
S3_RESULTS_PREFIX=results/

# DynamoDB Configuration
DYNAMODB_TABLE_NAME=TrafficAnalysis
DYNAMODB_REGION=us-east-1

# Lambda Configuration
LAMBDA_FUNCTION_NAME=analyze-traffic-flow
LAMBDA_REGION=us-east-1

# Analysis Parameters
SPEED_LIMIT_DEFAULT=55
CAPACITY_DEFAULT=2000
PEAK_HOURS=7,8,9,17,18,19
EOF
```

**Important:** Never commit `.env` to version control (add to `.gitignore`)

```bash
# Add to .gitignore
echo ".env" >> .gitignore
```

---

## Step 9: Verify Setup

Run verification script to ensure everything is configured:

```bash
python -c "
import boto3
import os
from dotenv import load_dotenv

load_dotenv()

print('Verifying AWS setup...\n')

# Check S3
s3 = boto3.client('s3', region_name=os.getenv('AWS_REGION'))
try:
    s3.head_bucket(Bucket=os.getenv('S3_BUCKET_NAME'))
    print('✓ S3 bucket accessible')
except Exception as e:
    print(f'✗ S3 bucket not accessible: {e}')

# Check DynamoDB
dynamodb = boto3.client('dynamodb', region_name=os.getenv('DYNAMODB_REGION'))
try:
    response = dynamodb.describe_table(TableName=os.getenv('DYNAMODB_TABLE_NAME'))
    status = response['Table']['TableStatus']
    print(f'✓ DynamoDB table accessible (status: {status})')
except Exception as e:
    print(f'✗ DynamoDB table not accessible: {e}')

# Check Lambda
lambda_client = boto3.client('lambda', region_name=os.getenv('LAMBDA_REGION'))
try:
    response = lambda_client.get_function(FunctionName=os.getenv('LAMBDA_FUNCTION_NAME'))
    runtime = response['Configuration']['Runtime']
    print(f'✓ Lambda function accessible (runtime: {runtime})')
except Exception as e:
    print(f'✗ Lambda function not accessible: {e}')

print('\nSetup verification complete!')
"
```

---

## Troubleshooting Setup Issues

### "NoCredentialsError"
- Solution: Run `aws configure` and enter your access keys
- Verify with `aws sts get-caller-identity`

### "Access Denied" for S3
- Check IAM user has S3FullAccess policy
- Check S3 bucket policy (should be open to your account)

### "Access Denied" for DynamoDB
- Ensure Lambda role has DynamoDB permissions
- Verify role is attached to Lambda function
- Check table exists: `aws dynamodb list-tables`

### "Bucket already exists"
- Bucket names are globally unique across AWS
- Choose a different name with a unique suffix
- Example: `transportation-data-yourname-001`

### Lambda timeout issues
- Increase timeout to 300 seconds in configuration
- Check Lambda logs in CloudWatch for errors
- Consider increasing memory allocation (more memory = faster processing)

### S3 event notification not triggering
- Verify Lambda permission was added with `add-permission`
- Check S3 event configuration with `get-bucket-notification-configuration`
- Ensure filter prefix matches uploaded objects (raw/)
- Check CloudWatch logs for Lambda invocations

### DynamoDB write errors
- Check IAM role has PutItem permissions
- Verify table schema matches data structure
- Check for missing required attributes (segment_id, timestamp)

### Athena query fails
- Ensure S3 path in CREATE TABLE statement is correct
- Check CSV files have headers
- Verify data types match schema
- Set up query result location in Athena settings

---

## Cost Monitoring

To monitor costs as you use the pipeline:

### Using AWS Console
1. Go to **Billing** > **Cost Explorer**
2. Set date range for your project
3. Filter by service (S3, Lambda, DynamoDB)
4. Check daily and hourly breakdowns

### Set Up Billing Alerts

1. Go to **Billing** > **Budgets**
2. Click **Create budget**
3. Budget type: **Cost budget**
4. Set budget amount: $15 (for this project)
5. Set up alert thresholds:
   - 80% ($12)
   - 100% ($15)
6. Enter email for notifications
7. Click **Create budget**

### Using AWS CLI
```bash
aws ce get-cost-and-usage \
    --time-period Start=2025-01-01,End=2025-01-08 \
    --granularity DAILY \
    --metrics BlendedCost \
    --group-by Type=DIMENSION,Key=SERVICE
```

---

## Next Steps

After setup verification:

1. **Generate sample data:** Open `notebooks/transportation_analysis.ipynb`
2. **Run upload script:** `python scripts/upload_to_s3.py`
3. **Monitor Lambda:** Check CloudWatch logs for processing
4. **Query results:** `python scripts/query_results.py`
5. **Analyze in notebook:** Continue with notebook cells for visualization

See `README.md` for workflow instructions.

---

## Cleanup

After completing the project, follow `cleanup_guide.md` to delete AWS resources and avoid charges.

**Important:** Incomplete cleanup can result in ongoing charges, especially for:
- S3 storage (charged per GB per month)
- DynamoDB on-demand (charged per read/write)
- Lambda invocations (free tier: 1M requests/month)

---

## Quick Reference

### AWS Resource Names
- **S3 Bucket**: `transportation-data-{your-user-id}`
- **DynamoDB Table**: `TrafficAnalysis`
- **Lambda Function**: `analyze-traffic-flow`
- **IAM Role**: `lambda-traffic-processor`
- **Athena Database**: `traffic_db`

### Key AWS Regions
- **us-east-1**: N. Virginia (lowest cost, most services)
- **us-west-2**: Oregon
- **eu-west-1**: Ireland

### Python Commands
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run upload script
python scripts/upload_to_s3.py

# Query results
python scripts/query_results.py

# Start Jupyter
jupyter notebook
```

### AWS CLI Quick Commands
```bash
# List S3 buckets
aws s3 ls

# List S3 bucket contents
aws s3 ls s3://transportation-data-{your-user-id}/ --recursive

# Invoke Lambda manually
aws lambda invoke --function-name analyze-traffic-flow out.json

# Query DynamoDB
aws dynamodb scan --table-name TrafficAnalysis

# View Lambda logs
aws logs tail /aws/lambda/analyze-traffic-flow --follow
```

---

**Setup complete!** You're ready to start analyzing traffic data with AWS.

Return to [README.md](README.md) for the project workflow.
