# AWS Setup Guide - Disease Surveillance Pipeline

This guide walks you through setting up all AWS services needed for the epidemiological surveillance pipeline.

**Estimated Time:** 30-45 minutes
**Prerequisites:** AWS account with billing enabled

---

## Table of Contents

1. [AWS Account Setup](#1-aws-account-setup)
2. [Configure AWS CLI](#2-configure-aws-cli)
3. [Create S3 Bucket](#3-create-s3-bucket)
4. [Create DynamoDB Table](#4-create-dynamodb-table)
5. [Create SNS Topic](#5-create-sns-topic)
6. [Create IAM Role](#6-create-iam-role)
7. [Deploy Lambda Function](#7-deploy-lambda-function)
8. [Configure S3 Event Trigger](#8-configure-s3-event-trigger)
9. [Optional: Setup Athena](#9-optional-setup-athena)
10. [Verify Setup](#10-verify-setup)
11. [Test the Pipeline](#11-test-the-pipeline)

---

## 1. AWS Account Setup

### 1.1 Create AWS Account (if needed)

1. Go to https://aws.amazon.com/
2. Click "Create an AWS Account"
3. Follow the registration process
4. Provide payment method (required even for free tier)
5. Verify identity via phone
6. Select "Basic Support - Free" plan

### 1.2 Sign in to AWS Console

1. Go to https://console.aws.amazon.com/
2. Sign in with your root account credentials
3. **Important:** Set up MFA (Multi-Factor Authentication) for security
   - Go to IAM → Dashboard → Security recommendations
   - Follow prompts to add MFA

### 1.3 Create IAM User (Best Practice)

Instead of using root account:

1. Go to IAM → Users → Add users
2. User name: `epidemiology-admin`
3. Check "AWS Management Console access"
4. Check "Programmatic access"
5. Click "Next: Permissions"
6. Select "Attach existing policies directly"
7. Add policies:
   - `AmazonS3FullAccess`
   - `AWSLambdaFullAccess`
   - `AmazonDynamoDBFullAccess`
   - `AmazonSNSFullAccess`
   - `IAMFullAccess` (for creating roles)
   - `AmazonAthenaFullAccess` (if using Athena)
8. Click through to "Create user"
9. **Save the Access Key ID and Secret Access Key** (download CSV)
10. Sign out and sign back in as the IAM user

---

## 2. Configure AWS CLI

### 2.1 Install AWS CLI

**macOS/Linux:**
```bash
# Using pip
pip install awscli

# Or using Homebrew (macOS)
brew install awscli
```

**Windows:**
Download from: https://aws.amazon.com/cli/

### 2.2 Configure Credentials

```bash
aws configure
```

Enter the following when prompted:
- **AWS Access Key ID:** [Your IAM user access key]
- **AWS Secret Access Key:** [Your IAM user secret key]
- **Default region name:** `us-east-1` (or your preferred region)
- **Default output format:** `json`

### 2.3 Verify Configuration

```bash
aws sts get-caller-identity
```

Expected output:
```json
{
    "UserId": "AIDAI...",
    "Account": "123456789012",
    "Arn": "arn:aws:iam::123456789012:user/epidemiology-admin"
}
```

---

## 3. Create S3 Bucket

### 3.1 Generate Unique Bucket Name

S3 bucket names must be globally unique. Create a unique identifier:

```bash
# Generate unique ID (timestamp-based)
UNIQUE_ID=$(date +%s)
BUCKET_NAME="epidemiology-data-${UNIQUE_ID}"
echo "Your bucket name: ${BUCKET_NAME}"

# Save for later use
echo "export EPIDEMIOLOGY_BUCKET=${BUCKET_NAME}" >> ~/.bashrc
source ~/.bashrc
```

### 3.2 Create Bucket via CLI

```bash
aws s3 mb s3://${BUCKET_NAME} --region us-east-1
```

### 3.3 Create Folder Structure

```bash
aws s3api put-object --bucket ${BUCKET_NAME} --key case-data/
aws s3api put-object --bucket ${BUCKET_NAME} --key processed-data/
aws s3api put-object --bucket ${BUCKET_NAME} --key logs/
```

### 3.4 Configure Bucket Settings (Optional)

Enable versioning for data safety:
```bash
aws s3api put-bucket-versioning \
    --bucket ${BUCKET_NAME} \
    --versioning-configuration Status=Enabled
```

Enable encryption at rest:
```bash
aws s3api put-bucket-encryption \
    --bucket ${BUCKET_NAME} \
    --server-side-encryption-configuration '{
        "Rules": [{
            "ApplyServerSideEncryptionByDefault": {
                "SSEAlgorithm": "AES256"
            }
        }]
    }'
```

### 3.5 Verify Bucket Creation

```bash
aws s3 ls s3://${BUCKET_NAME}/
```

Expected output:
```
                           PRE case-data/
                           PRE processed-data/
                           PRE logs/
```

---

## 4. Create DynamoDB Table

### 4.1 Create Table via CLI

```bash
aws dynamodb create-table \
    --table-name DiseaseReports \
    --attribute-definitions \
        AttributeName=case_id,AttributeType=S \
        AttributeName=report_date,AttributeType=S \
    --key-schema \
        AttributeName=case_id,KeyType=HASH \
        AttributeName=report_date,KeyType=RANGE \
    --billing-mode PAY_PER_REQUEST \
    --region us-east-1
```

**Explanation:**
- **Table name:** `DiseaseReports`
- **Partition key:** `case_id` (String) - unique case identifier
- **Sort key:** `report_date` (String) - allows time-based queries
- **Billing mode:** Pay-per-request (auto-scales, good for learning)

### 4.2 Create Global Secondary Index (GSI) for Disease Queries

```bash
aws dynamodb update-table \
    --table-name DiseaseReports \
    --attribute-definitions \
        AttributeName=disease,AttributeType=S \
        AttributeName=report_date,AttributeType=S \
    --global-secondary-index-updates '[
        {
            "Create": {
                "IndexName": "disease-date-index",
                "KeySchema": [
                    {"AttributeName": "disease", "KeyType": "HASH"},
                    {"AttributeName": "report_date", "KeyType": "RANGE"}
                ],
                "Projection": {"ProjectionType": "ALL"}
            }
        }
    ]'
```

### 4.3 Create GSI for Region Queries

```bash
aws dynamodb update-table \
    --table-name DiseaseReports \
    --attribute-definitions \
        AttributeName=region,AttributeType=S \
        AttributeName=report_date,AttributeType=S \
    --global-secondary-index-updates '[
        {
            "Create": {
                "IndexName": "region-date-index",
                "KeySchema": [
                    {"AttributeName": "region", "KeyType": "HASH"},
                    {"AttributeName": "report_date", "KeyType": "RANGE"}
                ],
                "Projection": {"ProjectionType": "ALL"}
            }
        }
    ]'
```

### 4.4 Wait for Table Creation

```bash
aws dynamodb wait table-exists --table-name DiseaseReports
echo "Table created successfully!"
```

### 4.5 Verify Table

```bash
aws dynamodb describe-table --table-name DiseaseReports \
    --query 'Table.[TableName,TableStatus,ItemCount]'
```

Expected output:
```json
[
    "DiseaseReports",
    "ACTIVE",
    0
]
```

---

## 5. Create SNS Topic

### 5.1 Create Topic

```bash
aws sns create-topic --name outbreak-alerts --region us-east-1
```

Output will include TopicArn:
```json
{
    "TopicArn": "arn:aws:sns:us-east-1:123456789012:outbreak-alerts"
}
```

**Save this ARN for later!**

```bash
TOPIC_ARN=$(aws sns create-topic --name outbreak-alerts --region us-east-1 --query 'TopicArn' --output text)
echo "export OUTBREAK_TOPIC_ARN=${TOPIC_ARN}" >> ~/.bashrc
source ~/.bashrc
echo "Topic ARN: ${TOPIC_ARN}"
```

### 5.2 Subscribe Email to Topic

```bash
aws sns subscribe \
    --topic-arn ${TOPIC_ARN} \
    --protocol email \
    --notification-endpoint your-email@example.com
```

Replace `your-email@example.com` with your actual email.

### 5.3 Confirm Subscription

1. Check your email for "AWS Notification - Subscription Confirmation"
2. Click the "Confirm subscription" link
3. You should see a confirmation page

### 5.4 Verify Subscription

```bash
aws sns list-subscriptions-by-topic --topic-arn ${TOPIC_ARN}
```

Expected output:
```json
{
    "Subscriptions": [
        {
            "SubscriptionArn": "arn:aws:sns:us-east-1:123456789012:outbreak-alerts:...",
            "Owner": "123456789012",
            "Protocol": "email",
            "Endpoint": "your-email@example.com",
            "TopicArn": "arn:aws:sns:us-east-1:123456789012:outbreak-alerts"
        }
    ]
}
```

---

## 6. Create IAM Role

### 6.1 Create Trust Policy

Create file `lambda-trust-policy.json`:

```json
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
```

### 6.2 Create IAM Role

```bash
aws iam create-role \
    --role-name lambda-epidemiology-role \
    --assume-role-policy-document file://lambda-trust-policy.json
```

### 6.3 Create Permission Policy

Create file `lambda-permissions-policy.json`:

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
      "Resource": "arn:aws:s3:::epidemiology-data-*/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:PutItem",
        "dynamodb:GetItem",
        "dynamodb:Query",
        "dynamodb:Scan"
      ],
      "Resource": [
        "arn:aws:dynamodb:us-east-1:*:table/DiseaseReports",
        "arn:aws:dynamodb:us-east-1:*:table/DiseaseReports/index/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "sns:Publish"
      ],
      "Resource": "arn:aws:sns:us-east-1:*:outbreak-alerts"
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

### 6.4 Attach Policy to Role

```bash
aws iam put-role-policy \
    --role-name lambda-epidemiology-role \
    --policy-name lambda-epidemiology-policy \
    --policy-document file://lambda-permissions-policy.json
```

### 6.5 Verify Role

```bash
aws iam get-role --role-name lambda-epidemiology-role
```

---

## 7. Deploy Lambda Function

### 7.1 Prepare Lambda Code

Navigate to the scripts directory:
```bash
cd scripts/
```

### 7.2 Install Dependencies

Lambda requires dependencies in a specific format:

```bash
mkdir lambda-package
cd lambda-package

# Copy lambda function
cp ../lambda_function.py .

# Install dependencies to this directory
pip install -t . numpy scipy pandas boto3
```

### 7.3 Create Deployment Package

```bash
zip -r ../lambda-deployment.zip .
cd ..
```

### 7.4 Deploy to Lambda

Get your account ID:
```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
```

Deploy the function:
```bash
aws lambda create-function \
    --function-name analyze-disease-surveillance \
    --runtime python3.9 \
    --role arn:aws:iam::${ACCOUNT_ID}:role/lambda-epidemiology-role \
    --handler lambda_function.lambda_handler \
    --zip-file fileb://lambda-deployment.zip \
    --timeout 60 \
    --memory-size 256 \
    --environment Variables="{
        S3_BUCKET_NAME=${BUCKET_NAME},
        DYNAMODB_TABLE=DiseaseReports,
        SNS_TOPIC_ARN=${TOPIC_ARN}
    }"
```

### 7.5 Verify Lambda Function

```bash
aws lambda get-function --function-name analyze-disease-surveillance
```

---

## 8. Configure S3 Event Trigger

### 8.1 Grant S3 Permission to Invoke Lambda

```bash
aws lambda add-permission \
    --function-name analyze-disease-surveillance \
    --statement-id s3-trigger-permission \
    --action lambda:InvokeFunction \
    --principal s3.amazonaws.com \
    --source-arn arn:aws:s3:::${BUCKET_NAME}
```

### 8.2 Create S3 Event Notification

Create file `s3-notification-config.json`:

```json
{
  "LambdaFunctionConfigurations": [
    {
      "Id": "case-data-upload-trigger",
      "LambdaFunctionArn": "arn:aws:lambda:us-east-1:ACCOUNT_ID:function:analyze-disease-surveillance",
      "Events": ["s3:ObjectCreated:*"],
      "Filter": {
        "Key": {
          "FilterRules": [
            {
              "Name": "prefix",
              "Value": "case-data/"
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
```

Replace `ACCOUNT_ID` with your account ID:
```bash
sed -i "s/ACCOUNT_ID/${ACCOUNT_ID}/g" s3-notification-config.json
```

### 8.3 Apply Notification Configuration

```bash
aws s3api put-bucket-notification-configuration \
    --bucket ${BUCKET_NAME} \
    --notification-configuration file://s3-notification-config.json
```

### 8.4 Verify Notification

```bash
aws s3api get-bucket-notification-configuration --bucket ${BUCKET_NAME}
```

---

## 9. Optional: Setup Athena

### 9.1 Create Athena Output Bucket

```bash
ATHENA_OUTPUT_BUCKET="athena-results-${UNIQUE_ID}"
aws s3 mb s3://${ATHENA_OUTPUT_BUCKET} --region us-east-1
```

### 9.2 Configure Athena Workgroup

```bash
aws athena create-work-group \
    --name epidemiology-workgroup \
    --configuration "ResultConfigurationUpdates={OutputLocation=s3://${ATHENA_OUTPUT_BUCKET}/}"
```

### 9.3 Create Athena Database

```bash
aws athena start-query-execution \
    --query-string "CREATE DATABASE IF NOT EXISTS epidemiology_db" \
    --work-group epidemiology-workgroup
```

### 9.4 Create External Table

Create a query file `create-athena-table.sql`:

```sql
CREATE EXTERNAL TABLE IF NOT EXISTS epidemiology_db.disease_surveillance (
    case_id STRING,
    disease STRING,
    report_date STRING,
    region STRING,
    zip_code STRING,
    age_group STRING,
    sex STRING,
    outcome STRING,
    symptom_onset_date STRING,
    incidence_rate DOUBLE,
    case_fatality_rate DOUBLE
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION 's3://BUCKET_NAME/case-data/'
TBLPROPERTIES ('skip.header.line.count'='1');
```

Replace `BUCKET_NAME` and execute:
```bash
sed "s/BUCKET_NAME/${BUCKET_NAME}/g" create-athena-table.sql > create-athena-table-final.sql

aws athena start-query-execution \
    --query-string "$(cat create-athena-table-final.sql)" \
    --work-group epidemiology-workgroup
```

---

## 10. Verify Setup

### 10.1 Check All Resources

Run this verification script:

```bash
echo "=== Verification Summary ==="
echo ""

# S3 Bucket
echo "✓ S3 Bucket:"
aws s3 ls s3://${BUCKET_NAME}/ && echo "  Status: OK" || echo "  Status: FAILED"
echo ""

# DynamoDB Table
echo "✓ DynamoDB Table:"
aws dynamodb describe-table --table-name DiseaseReports --query 'Table.TableStatus' && echo "  Status: OK" || echo "  Status: FAILED"
echo ""

# SNS Topic
echo "✓ SNS Topic:"
aws sns get-topic-attributes --topic-arn ${TOPIC_ARN} --query 'Attributes.TopicArn' && echo "  Status: OK" || echo "  Status: FAILED"
echo ""

# Lambda Function
echo "✓ Lambda Function:"
aws lambda get-function --function-name analyze-disease-surveillance --query 'Configuration.FunctionName' && echo "  Status: OK" || echo "  Status: FAILED"
echo ""

# IAM Role
echo "✓ IAM Role:"
aws iam get-role --role-name lambda-epidemiology-role --query 'Role.RoleName' && echo "  Status: OK" || echo "  Status: FAILED"
echo ""

echo "=== Setup Complete ==="
```

### 10.2 Expected Output

All checks should return "Status: OK"

---

## 11. Test the Pipeline

### 11.1 Create Test Case Data

Create file `test-cases.csv`:

```csv
case_id,disease,report_date,region,zip_code,age_group,sex,outcome,symptom_onset_date
case_001,influenza,2024-01-15,northeast,02101,45-54,M,recovered,2024-01-10
case_002,influenza,2024-01-16,northeast,02102,25-34,F,recovered,2024-01-12
case_003,influenza,2024-01-17,northeast,02103,55-64,M,hospitalized,2024-01-13
```

### 11.2 Upload Test Data

```bash
aws s3 cp test-cases.csv s3://${BUCKET_NAME}/case-data/test-cases.csv
```

### 11.3 Monitor Lambda Execution

```bash
# Wait a few seconds for processing
sleep 10

# Check CloudWatch logs
aws logs tail /aws/lambda/analyze-disease-surveillance --follow
```

### 11.4 Check DynamoDB

```bash
aws dynamodb scan --table-name DiseaseReports --limit 5
```

### 11.5 Check SNS Alert (if outbreak detected)

Check your email for outbreak alert notifications.

---

## Troubleshooting

### Issue: Lambda permission denied for S3

**Solution:**
```bash
# Re-add Lambda permission
aws lambda add-permission \
    --function-name analyze-disease-surveillance \
    --statement-id s3-trigger-permission-2 \
    --action lambda:InvokeFunction \
    --principal s3.amazonaws.com \
    --source-arn arn:aws:s3:::${BUCKET_NAME}
```

### Issue: Lambda timeout

**Solution:**
```bash
# Increase timeout to 5 minutes
aws lambda update-function-configuration \
    --function-name analyze-disease-surveillance \
    --timeout 300
```

### Issue: Lambda out of memory

**Solution:**
```bash
# Increase memory to 512 MB
aws lambda update-function-configuration \
    --function-name analyze-disease-surveillance \
    --memory-size 512
```

### Issue: SNS subscription not confirmed

**Solution:**
1. Check spam folder for confirmation email
2. Resend confirmation: Go to SNS console → Topics → outbreak-alerts → Subscriptions → Request confirmation
3. Click confirmation link in email

### Issue: DynamoDB table not found

**Solution:**
```bash
# Verify table exists
aws dynamodb list-tables

# If missing, recreate using Step 4
```

---

## Cost Monitoring

### Set up Billing Alert

1. Go to AWS Console → Billing → Budgets
2. Click "Create budget"
3. Select "Cost budget"
4. Set amount: $10
5. Set alert threshold: 80% ($8)
6. Enter email address
7. Click "Create budget"

### Check Current Costs

```bash
# Get current month costs
aws ce get-cost-and-usage \
    --time-period Start=$(date -u +%Y-%m-01),End=$(date -u +%Y-%m-%d) \
    --granularity MONTHLY \
    --metrics BlendedCost
```

---

## Summary

You've successfully set up:

- ✅ S3 bucket for case data storage
- ✅ DynamoDB table for surveillance records
- ✅ SNS topic for outbreak alerts
- ✅ IAM role with appropriate permissions
- ✅ Lambda function for epidemiological analysis
- ✅ S3 event trigger for automatic processing
- ✅ (Optional) Athena for SQL queries

**Next Steps:**
1. Run the upload script: `python scripts/upload_to_s3.py`
2. Monitor Lambda logs for processing status
3. Query results: `python scripts/query_results.py`
4. Analyze in Jupyter: `jupyter notebook notebooks/epidemiology_analysis.ipynb`

**Important:** Remember to clean up resources when done to avoid charges! See `cleanup_guide.md`.

---

**Setup Time:** ~30-45 minutes
**Ready for:** Data upload and analysis
**Cost:** ~$0.50 for setup (mostly free tier eligible)
