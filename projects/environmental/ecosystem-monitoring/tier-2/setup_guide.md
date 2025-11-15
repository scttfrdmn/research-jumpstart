# AWS Setup Guide - Environmental Sensor Data Analysis Tier 2

This guide walks you through setting up all AWS resources needed for the environmental monitoring project. Estimated time: 35-45 minutes.

---

## Prerequisites

Before starting, ensure you have:
- AWS account with billing enabled
- AWS CLI installed (`aws --version`)
- AWS credentials configured (`aws configure`)
- Python 3.8+ installed
- $15 budget available for testing

---

## Setup Overview

We'll create these AWS resources:
1. **S3 Bucket** - Store sensor data
2. **IAM Role** - Lambda permissions
3. **Lambda Function** - Process environmental data
4. **DynamoDB Table** - Store processed readings
5. **SNS Topic** - Send pollution alerts
6. **S3 Event Trigger** - Automatically invoke Lambda

**Total setup time:** 35-45 minutes

---

## Step 1: Create S3 Bucket

S3 stores your sensor data and serves as the entry point for the pipeline.

### Option A: Using AWS Console (Recommended for beginners)

1. Go to [AWS S3 Console](https://console.aws.amazon.com/s3/)
2. Click "Create bucket"
3. Enter bucket name: `environmental-data-{your-name}-{date}`
   - Example: `environmental-data-alice-20250114`
   - Must be globally unique across ALL AWS accounts
   - Must be lowercase
   - Only letters, numbers, and hyphens
4. Select region: **us-east-1** (recommended)
5. **Block Public Access**: Keep all boxes checked (block all public access)
6. **Bucket Versioning**: Disabled (to save costs)
7. **Default Encryption**: Enabled (SSE-S3)
8. Click "Create bucket"

### Option B: Using AWS CLI

```bash
# Create S3 bucket with unique name
BUCKET_NAME="environmental-data-$(whoami)-$(date +%s)"
aws s3 mb "s3://$BUCKET_NAME" --region us-east-1

# Verify bucket created
aws s3 ls | grep environmental-data

# Save bucket name for later steps
echo "export BUCKET_NAME=$BUCKET_NAME" >> ~/.bashrc
echo "BUCKET_NAME=$BUCKET_NAME" > .env
```

### Create Folder Structure

Create folders to organize your data:

**Using AWS Console:**
1. Open your environmental-data bucket
2. Click "Create folder" → name it `raw` → Create
3. Click "Create folder" → name it `logs` → Create

**Using AWS CLI:**
```bash
# Create folder structure (folders are just prefixes in S3)
aws s3api put-object --bucket "$BUCKET_NAME" --key "raw/"
aws s3api put-object --bucket "$BUCKET_NAME" --key "logs/"

# Verify
aws s3 ls "s3://$BUCKET_NAME/" --recursive
```

**Save your bucket name!** You'll use it throughout this project.

---

## Step 2: Create DynamoDB Table

DynamoDB stores processed sensor readings for fast queries.

### Option A: Using AWS Console

1. Go to [DynamoDB Console](https://console.aws.amazon.com/dynamodb/)
2. Click "Create table"
3. Table name: `EnvironmentalReadings`
4. Partition key: `location_id` (String)
5. Sort key: `timestamp` (String)
6. Table settings: **Customize settings**
7. Table class: **DynamoDB Standard**
8. Capacity mode: **On-demand** (pay per request)
9. Encryption: **Default** (AWS owned key)
10. Click "Create table"

Wait 30-60 seconds for table to become active.

### Option B: Using AWS CLI

```bash
# Create DynamoDB table
aws dynamodb create-table \
  --table-name EnvironmentalReadings \
  --attribute-definitions \
    AttributeName=location_id,AttributeType=S \
    AttributeName=timestamp,AttributeType=S \
  --key-schema \
    AttributeName=location_id,KeyType=HASH \
    AttributeName=timestamp,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST \
  --region us-east-1

# Wait for table to become active
aws dynamodb wait table-exists --table-name EnvironmentalReadings

# Verify table created
aws dynamodb describe-table --table-name EnvironmentalReadings \
  --query 'Table.[TableName,TableStatus,ItemCount]' --output table
```

### Table Schema

The table uses this schema for efficient time series queries:

```
EnvironmentalReadings
├── location_id (Partition Key, String)    # e.g., "station-01"
├── timestamp (Sort Key, String)           # ISO 8601: "2025-01-14T10:30:00Z"
└── Attributes:
    ├── reading_id (String)                # Unique identifier
    ├── sensor_type (String)               # "air", "water", "soil", "weather"
    ├── parameters (Map)                   # {PM25: 35.4, temp: 22.5, ...}
    ├── calculated_metrics (Map)           # {AQI: 101, WQI: 85, ...}
    ├── alert_status (String)              # "none", "warning", "critical"
    ├── alert_message (String)             # Description if alert
    ├── coordinates (Map)                  # {lat: 40.7128, lon: -74.0060}
    ├── processing_time_ms (Number)        # Lambda execution time
    └── data_quality_score (Number)        # 0-100
```

---

## Step 3: Create SNS Topic for Alerts

SNS sends email alerts when pollution thresholds are exceeded.

### Option A: Using AWS Console

1. Go to [SNS Console](https://console.aws.amazon.com/sns/)
2. Click "Topics" in left menu
3. Click "Create topic"
4. Type: **Standard**
5. Name: `environmental-alerts`
6. Display name: `Environmental Alerts`
7. Keep all default settings
8. Click "Create topic"

#### Subscribe to Email Alerts:

1. In the topic page, click "Create subscription"
2. Protocol: **Email**
3. Endpoint: Your email address
4. Click "Create subscription"
5. **Check your email** for confirmation message
6. Click "Confirm subscription" link in email

### Option B: Using AWS CLI

```bash
# Create SNS topic
TOPIC_ARN=$(aws sns create-topic --name environmental-alerts \
  --query 'TopicArn' --output text)

echo "export TOPIC_ARN=$TOPIC_ARN" >> ~/.bashrc
echo "TOPIC_ARN=$TOPIC_ARN" >> .env

# Subscribe your email
aws sns subscribe \
  --topic-arn "$TOPIC_ARN" \
  --protocol email \
  --notification-endpoint your-email@example.com

# Check your email and confirm subscription!
# You MUST confirm before alerts will be sent
```

**Important:** Check your email (including spam folder) and confirm the subscription!

---

## Step 4: Create IAM Role for Lambda

Lambda needs permissions to access S3, DynamoDB, SNS, and CloudWatch.

### Option A: Using AWS Console

1. Go to [IAM Console](https://console.aws.amazon.com/iam/)
2. Click "Roles" in left menu
3. Click "Create role"
4. Trusted entity: **AWS service**
5. Use case: **Lambda**
6. Click "Next"

#### Attach Policies:

Search for and select these policies:
- `AWSLambdaBasicExecutionRole` (CloudWatch logs)
- `AmazonS3FullAccess` (S3 read/write)
- `AmazonDynamoDBFullAccess` (DynamoDB write)
- `AmazonSNSFullAccess` (SNS publish)

7. Click "Next"
8. Role name: `lambda-environmental-processor`
9. Description: `Lambda role for environmental sensor data processing`
10. Click "Create role"

#### Get Role ARN:

1. Click on the role name `lambda-environmental-processor`
2. Copy the **ARN** at the top (you'll need this)
   - Format: `arn:aws:iam::123456789012:role/lambda-environmental-processor`

### Option B: Using AWS CLI

```bash
# Create trust policy document
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
  --role-name lambda-environmental-processor \
  --assume-role-policy-document "$TRUST_POLICY"

# Attach policies
aws iam attach-role-policy \
  --role-name lambda-environmental-processor \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam attach-role-policy \
  --role-name lambda-environmental-processor \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam attach-role-policy \
  --role-name lambda-environmental-processor \
  --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess

aws iam attach-role-policy \
  --role-name lambda-environmental-processor \
  --policy-arn arn:aws:iam::aws:policy/AmazonSNSFullAccess

# Get role ARN (save this!)
ROLE_ARN=$(aws iam get-role --role-name lambda-environmental-processor \
  --query 'Role.Arn' --output text)

echo "export ROLE_ARN=$ROLE_ARN" >> ~/.bashrc
echo "ROLE_ARN=$ROLE_ARN" >> .env

echo "Role ARN: $ROLE_ARN"
```

**Important:** Wait 10 seconds for IAM role to propagate before creating Lambda.

---

## Step 5: Create Lambda Function

Lambda processes environmental sensor data and triggers alerts.

### Option A: Using AWS Console

#### Create Function:

1. Go to [Lambda Console](https://console.aws.amazon.com/lambda/)
2. Click "Create function"
3. Select **Author from scratch**
4. Function name: `process-sensor-data`
5. Runtime: **Python 3.11**
6. Architecture: **x86_64**
7. Permissions: **Use an existing role**
   - Select `lambda-environmental-processor`
8. Click "Create function"

#### Upload Function Code:

1. In the function page, scroll to "Code source"
2. Delete default code in `lambda_function.py`
3. Open `scripts/lambda_function.py` from this project
4. Copy entire contents
5. Paste into Lambda code editor
6. Click "Deploy" button (wait for success message)

#### Configure Function Settings:

1. Click "Configuration" tab
2. Click "General configuration" → "Edit"
3. **Memory**: 256 MB
4. **Timeout**: 30 seconds
5. **Description**: Process environmental sensor data and send alerts
6. Click "Save"

#### Set Environment Variables:

1. Still in Configuration tab
2. Click "Environment variables" in left menu
3. Click "Edit"
4. Add these variables:

| Key | Value |
|-----|-------|
| `BUCKET_NAME` | `environmental-data-{your-bucket}` |
| `DYNAMODB_TABLE` | `EnvironmentalReadings` |
| `SNS_TOPIC_ARN` | `arn:aws:sns:us-east-1:ACCOUNT:environmental-alerts` |
| `ALERT_THRESHOLD_PM25` | `35.4` |
| `ALERT_THRESHOLD_AQI` | `101` |
| `ALERT_THRESHOLD_PH_MIN` | `6.5` |
| `ALERT_THRESHOLD_PH_MAX` | `8.5` |

5. Click "Save"

### Option B: Using AWS CLI

```bash
# Navigate to project directory
cd /path/to/research-jumpstart/projects/environmental/ecosystem-monitoring/tier-2

# Create deployment package
cd scripts
zip lambda_package.zip lambda_function.py
cd ..

# Create Lambda function
aws lambda create-function \
  --function-name process-sensor-data \
  --runtime python3.11 \
  --role "$ROLE_ARN" \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://scripts/lambda_package.zip \
  --timeout 30 \
  --memory-size 256 \
  --environment "Variables={
    BUCKET_NAME=$BUCKET_NAME,
    DYNAMODB_TABLE=EnvironmentalReadings,
    SNS_TOPIC_ARN=$TOPIC_ARN,
    ALERT_THRESHOLD_PM25=35.4,
    ALERT_THRESHOLD_AQI=101,
    ALERT_THRESHOLD_PH_MIN=6.5,
    ALERT_THRESHOLD_PH_MAX=8.5
  }" \
  --region us-east-1

# Verify function created
aws lambda get-function --function-name process-sensor-data
```

---

## Step 6: Configure S3 Trigger

Set up S3 to automatically invoke Lambda when new sensor data is uploaded.

### Option A: Using AWS Console

#### Add S3 Permission to Lambda:

1. Go to [Lambda Console](https://console.aws.amazon.com/lambda/)
2. Open `process-sensor-data` function
3. Click "Configuration" tab
4. Click "Triggers" in left menu
5. Click "Add trigger"
6. Select trigger: **S3**
7. Bucket: Select `environmental-data-{your-bucket}`
8. Event type: **All object create events**
9. Prefix: `raw/`
10. Suffix: `.csv` or `.json` (optional - leave blank to process all files)
11. Acknowledge the notice
12. Click "Add"

#### Verify Trigger:

1. In function page, you should see S3 in the "Function overview" diagram
2. In Configuration > Triggers, verify S3 trigger is enabled

### Option B: Using AWS CLI

```bash
# Get Lambda function ARN
LAMBDA_ARN=$(aws lambda get-function --function-name process-sensor-data \
  --query 'Configuration.FunctionArn' --output text)

# Add permission for S3 to invoke Lambda
aws lambda add-permission \
  --function-name process-sensor-data \
  --statement-id s3-trigger-permission \
  --action lambda:InvokeFunction \
  --principal s3.amazonaws.com \
  --source-arn "arn:aws:s3:::$BUCKET_NAME" \
  --source-account $(aws sts get-caller-identity --query Account --output text)

# Create S3 notification configuration
NOTIFICATION_CONFIG=$(cat <<EOF
{
  "LambdaFunctionConfigurations": [
    {
      "Id": "environmental-data-processor",
      "LambdaFunctionArn": "$LAMBDA_ARN",
      "Events": ["s3:ObjectCreated:*"],
      "Filter": {
        "Key": {
          "FilterRules": [
            {
              "Name": "prefix",
              "Value": "raw/"
            }
          ]
        }
      }
    }
  ]
}
EOF
)

# Apply notification configuration to S3 bucket
aws s3api put-bucket-notification-configuration \
  --bucket "$BUCKET_NAME" \
  --notification-configuration "$NOTIFICATION_CONFIG"

# Verify configuration
aws s3api get-bucket-notification-configuration --bucket "$BUCKET_NAME"
```

---

## Step 7: (Optional) Set Up Athena

Athena allows SQL queries on S3 data for advanced analysis.

### Using AWS Console:

1. Go to [Athena Console](https://console.aws.amazon.com/athena/)
2. Click "Get Started" (first time only)
3. Click "Settings" tab
4. Set query result location: `s3://environmental-data-{your-bucket}/athena-results/`
5. Click "Save"

### Create Athena Table:

Run this query in Athena query editor:

```sql
CREATE EXTERNAL TABLE IF NOT EXISTS environmental_readings (
  location_id STRING,
  timestamp STRING,
  reading_id STRING,
  sensor_type STRING,
  pm25 DOUBLE,
  pm10 DOUBLE,
  co2 DOUBLE,
  temperature DOUBLE,
  humidity DOUBLE,
  ph DOUBLE,
  dissolved_oxygen DOUBLE,
  turbidity DOUBLE,
  aqi INT,
  wqi INT,
  alert_status STRING,
  latitude DOUBLE,
  longitude DOUBLE
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe'
WITH SERDEPROPERTIES (
  'field.delim' = ',',
  'skip.header.line.count' = '1'
)
LOCATION 's3://environmental-data-{your-bucket}/raw/'
TBLPROPERTIES ('has_encrypted_data'='false');
```

Replace `{your-bucket}` with your actual bucket name.

---

## Step 8: Test the Pipeline

Verify everything is working correctly.

### Test Lambda Function Directly:

1. Go to Lambda console
2. Open `process-sensor-data` function
3. Click "Test" tab
4. Create new test event:
   - Event name: `test-sensor-data`
   - Template: `s3-put`
   - Modify to use your bucket name and a test file

```json
{
  "Records": [
    {
      "s3": {
        "bucket": {
          "name": "environmental-data-{your-bucket}"
        },
        "object": {
          "key": "raw/test_data.csv"
        }
      }
    }
  ]
}
```

5. Click "Save"
6. Click "Test"
7. Check execution results in the response

### Test S3 Upload Trigger:

```bash
# Create test CSV file
cat > test_sensor_data.csv <<EOF
timestamp,location_id,sensor_type,pm25,pm10,co2,temperature,humidity,latitude,longitude
2025-01-14T10:00:00Z,station-01,air,25.3,45.2,420,22.5,65,40.7128,-74.0060
2025-01-14T10:15:00Z,station-01,air,28.7,48.1,425,22.8,63,40.7128,-74.0060
EOF

# Upload to S3 (this should trigger Lambda)
aws s3 cp test_sensor_data.csv "s3://$BUCKET_NAME/raw/"

# Wait 10 seconds for processing
sleep 10

# Check DynamoDB for results
aws dynamodb query \
  --table-name EnvironmentalReadings \
  --key-condition-expression "location_id = :loc" \
  --expression-attribute-values '{":loc":{"S":"station-01"}}' \
  --limit 5
```

### Check CloudWatch Logs:

```bash
# View Lambda logs
aws logs tail /aws/lambda/process-sensor-data --follow

# Or view in console:
# Lambda console > Monitor tab > View CloudWatch logs
```

---

## Step 9: Configure Cost Alerts

Set up billing alerts to avoid unexpected charges.

### Using AWS Console:

1. Go to [Billing Console](https://console.aws.amazon.com/billing/)
2. Click "Budgets" in left menu
3. Click "Create budget"
4. Budget type: **Cost budget**
5. Budget name: `environmental-monitoring-budget`
6. Period: **Monthly**
7. Budgeted amount: `$15.00`
8. Set alerts:
   - Alert at 50% ($7.50)
   - Alert at 80% ($12.00)
   - Alert at 100% ($15.00)
9. Email recipients: Your email
10. Click "Create budget"

### Using AWS CLI:

```bash
# Create budget configuration
BUDGET_CONFIG=$(cat <<EOF
{
  "BudgetName": "environmental-monitoring-budget",
  "BudgetType": "COST",
  "TimeUnit": "MONTHLY",
  "BudgetLimit": {
    "Amount": "15.0",
    "Unit": "USD"
  }
}
EOF
)

aws budgets create-budget \
  --account-id $(aws sts get-caller-identity --query Account --output text) \
  --budget "$BUDGET_CONFIG"
```

---

## Verification Checklist

Before proceeding, verify all resources are created:

```bash
# Check S3 bucket
aws s3 ls | grep environmental-data

# Check DynamoDB table
aws dynamodb describe-table --table-name EnvironmentalReadings \
  --query 'Table.TableStatus'

# Check SNS topic
aws sns list-topics | grep environmental-alerts

# Check IAM role
aws iam get-role --role-name lambda-environmental-processor

# Check Lambda function
aws lambda get-function --function-name process-sensor-data \
  --query 'Configuration.[FunctionName,Runtime,Handler,State]'

# Check S3 notification
aws s3api get-bucket-notification-configuration \
  --bucket "$BUCKET_NAME"
```

Expected results:
- S3 bucket: `environmental-data-*` exists
- DynamoDB: Status = `ACTIVE`
- SNS: Topic ARN returned
- IAM: Role exists
- Lambda: State = `Active`
- S3 notification: Lambda function configured

---

## Configuration Summary

Save these values for use in scripts and notebooks:

```bash
# Create .env file with all configuration
cat > .env <<EOF
# AWS Configuration
AWS_REGION=us-east-1
BUCKET_NAME=$BUCKET_NAME
DYNAMODB_TABLE=EnvironmentalReadings
SNS_TOPIC_ARN=$TOPIC_ARN
LAMBDA_FUNCTION_NAME=process-sensor-data

# Alert Thresholds
ALERT_THRESHOLD_PM25=35.4
ALERT_THRESHOLD_AQI=101
ALERT_THRESHOLD_PH_MIN=6.5
ALERT_THRESHOLD_PH_MAX=8.5
ALERT_THRESHOLD_DO_MIN=5.0
EOF

echo "Configuration saved to .env"
cat .env
```

---

## Troubleshooting Setup

### S3 Bucket Name Already Taken

```bash
# S3 bucket names must be globally unique
# Try adding more uniqueness:
BUCKET_NAME="environmental-data-$(whoami)-$(uuidgen | cut -d'-' -f1)"
aws s3 mb "s3://$BUCKET_NAME" --region us-east-1
```

### IAM Role Not Found in Lambda

- Wait 10-15 seconds after creating IAM role
- IAM changes can take time to propagate

### Lambda Permission Denied

```bash
# Verify Lambda has correct IAM role
aws lambda get-function-configuration \
  --function-name process-sensor-data \
  --query 'Role'

# Should return: arn:aws:iam::ACCOUNT:role/lambda-environmental-processor
```

### S3 Trigger Not Working

```bash
# Check Lambda permissions
aws lambda get-policy --function-name process-sensor-data

# Should include s3.amazonaws.com principal

# Check S3 notification configuration
aws s3api get-bucket-notification-configuration --bucket "$BUCKET_NAME"
```

### SNS Subscription Not Confirmed

- Check email (including spam folder)
- Look for "AWS Notification - Subscription Confirmation"
- Click "Confirm subscription" link

### DynamoDB Throttling

```bash
# Ensure on-demand billing is enabled
aws dynamodb describe-table --table-name EnvironmentalReadings \
  --query 'Table.BillingModeSummary.BillingMode'

# Should return: PAY_PER_REQUEST
```

---

## Next Steps

Once setup is complete:

1. **Generate Sample Data**: Run `python scripts/upload_to_s3.py --generate-sample`
2. **Monitor Processing**: Check CloudWatch logs
3. **Query Results**: Run `python scripts/query_results.py`
4. **Analyze**: Open `notebooks/environmental_analysis.ipynb`
5. **Cleanup**: When done, follow `cleanup_guide.md`

---

## Estimated Costs

Setup itself is free, but resources incur charges:

| Resource | Setup Cost | Running Cost |
|----------|------------|--------------|
| S3 Bucket | $0 | $0.023/GB-month |
| DynamoDB Table | $0 | $1.25 per million writes |
| SNS Topic | $0 | $0.50 per million emails |
| Lambda Function | $0 | $0.20 per million requests |
| IAM Role | $0 | $0 |

**Total setup cost: $0**
**Total running cost: $7-12 for full project**

---

## Getting Help

If you encounter issues:

1. Check CloudWatch logs: `aws logs tail /aws/lambda/process-sensor-data`
2. Review IAM permissions
3. Verify all ARNs and names are correct
4. Check AWS service quotas: https://console.aws.amazon.com/servicequotas/

---

**Setup complete!** Proceed to README.md for running instructions.

**Last updated:** 2025-11-14 | Research Jumpstart v1.0.0
