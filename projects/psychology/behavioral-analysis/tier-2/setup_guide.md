# AWS Setup Guide - Behavioral Data Analysis

**Time Required:** 20-30 minutes
**Cost:** Free (setup only, no charges until you upload data)

This guide walks you through setting up all AWS resources needed for the behavioral analysis pipeline.

---

## Prerequisites

- AWS account (free tier eligible)
- AWS CLI installed and configured
- Python 3.8+ installed
- Basic command-line knowledge

---

## Overview

You'll create the following AWS resources:
1. **S3 Bucket** - Store behavioral data (CSV files)
2. **IAM Role** - Grant Lambda permissions
3. **Lambda Function** - Process behavioral data
4. **DynamoDB Table** - Store analysis results
5. **(Optional) Athena Workspace** - SQL queries on results

---

## Step 1: Configure AWS CLI

### 1.1 Install AWS CLI

```bash
# macOS
brew install awscli

# Linux
sudo apt-get install awscli

# Windows or pip
pip install awscli
```

### 1.2 Configure Credentials

```bash
aws configure
```

Enter the following when prompted:
- **AWS Access Key ID**: Your IAM user access key
- **AWS Secret Access Key**: Your secret key
- **Default region name**: `us-east-1` (recommended)
- **Default output format**: `json`

### 1.3 Verify Configuration

```bash
aws sts get-caller-identity
```

You should see your AWS account ID and user ARN.

---

## Step 2: Create S3 Bucket

### 2.1 Choose Unique Bucket Name

S3 bucket names must be globally unique. Generate a unique name:

```bash
# Use timestamp for uniqueness
BUCKET_NAME="behavioral-data-$(date +%s)"
echo "Your bucket name: $BUCKET_NAME"

# Or use your username
BUCKET_NAME="behavioral-data-yourname-$(date +%s)"
```

**Important:** Save this bucket name! You'll need it throughout the project.

### 2.2 Create Bucket

```bash
# Create bucket in us-east-1 region
aws s3 mb s3://$BUCKET_NAME --region us-east-1

# Expected output:
# make_bucket: behavioral-data-1731600000
```

### 2.3 Verify Bucket Creation

```bash
aws s3 ls | grep behavioral-data
```

### 2.4 Create Folder Structure

```bash
# Create folders for organization
aws s3api put-object --bucket $BUCKET_NAME --key raw/
aws s3api put-object --bucket $BUCKET_NAME --key metadata/
aws s3api put-object --bucket $BUCKET_NAME --key logs/
```

### 2.5 (Optional) Enable Versioning

```bash
# Enable versioning for data safety
aws s3api put-bucket-versioning \
  --bucket $BUCKET_NAME \
  --versioning-configuration Status=Enabled
```

### 2.6 (Optional) Set Lifecycle Policy

To automatically delete old files and save costs:

```bash
# Create lifecycle policy file
cat > lifecycle.json <<EOF
{
  "Rules": [
    {
      "Id": "DeleteOldLogs",
      "Status": "Enabled",
      "Prefix": "logs/",
      "Expiration": {
        "Days": 7
      }
    },
    {
      "Id": "DeleteOldRaw",
      "Status": "Enabled",
      "Prefix": "raw/",
      "Expiration": {
        "Days": 30
      }
    }
  ]
}
EOF

# Apply lifecycle policy
aws s3api put-bucket-lifecycle-configuration \
  --bucket $BUCKET_NAME \
  --lifecycle-configuration file://lifecycle.json
```

---

## Step 3: Create IAM Role for Lambda

### 3.1 Create Trust Policy

Lambda needs permission to assume the role:

```bash
cat > trust-policy.json <<EOF
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
```

### 3.2 Create IAM Role

```bash
aws iam create-role \
  --role-name lambda-behavioral-processor \
  --assume-role-policy-document file://trust-policy.json

# Expected output:
# Role ARN: arn:aws:iam::123456789012:role/lambda-behavioral-processor
```

**Important:** Save the Role ARN! You'll need it when creating the Lambda function.

### 3.3 Create IAM Policy

This policy grants permissions to read S3, write DynamoDB, and log to CloudWatch:

```bash
cat > lambda-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::$BUCKET_NAME",
        "arn:aws:s3:::$BUCKET_NAME/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:PutItem",
        "dynamodb:UpdateItem",
        "dynamodb:GetItem"
      ],
      "Resource": "arn:aws:dynamodb:us-east-1:*:table/BehavioralAnalysis"
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:us-east-1:*:log-group:/aws/lambda/*"
    }
  ]
}
EOF
```

### 3.4 Attach Policy to Role

```bash
aws iam put-role-policy \
  --role-name lambda-behavioral-processor \
  --policy-name BehavioralAnalysisPolicy \
  --policy-document file://lambda-policy.json
```

### 3.5 Verify Role Creation

```bash
aws iam get-role --role-name lambda-behavioral-processor
```

---

## Step 4: Create DynamoDB Table

### 4.1 Define Table Schema

DynamoDB requires a primary key. We'll use `participant_id` as the partition key and `task_type` as the sort key:

```bash
aws dynamodb create-table \
  --table-name BehavioralAnalysis \
  --attribute-definitions \
    AttributeName=participant_id,AttributeType=S \
    AttributeName=task_type,AttributeType=S \
  --key-schema \
    AttributeName=participant_id,KeyType=HASH \
    AttributeName=task_type,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST \
  --region us-east-1

# Expected output:
# Table ARN: arn:aws:dynamodb:us-east-1:123456789012:table/BehavioralAnalysis
```

**Why PAY_PER_REQUEST?**
- No capacity planning needed
- Automatically scales
- Only pay for actual usage
- Perfect for variable workloads

### 4.2 Wait for Table to be Active

```bash
aws dynamodb wait table-exists --table-name BehavioralAnalysis
echo "Table is ready!"
```

### 4.3 Verify Table Creation

```bash
aws dynamodb describe-table --table-name BehavioralAnalysis
```

### 4.4 (Optional) Create Global Secondary Index

For querying by performance metrics:

```bash
aws dynamodb update-table \
  --table-name BehavioralAnalysis \
  --attribute-definitions \
    AttributeName=accuracy,AttributeType=N \
  --global-secondary-index-updates \
    "[{\"Create\":{\"IndexName\":\"accuracy-index\",\"KeySchema\":[{\"AttributeName\":\"accuracy\",\"KeyType\":\"HASH\"}],\"Projection\":{\"ProjectionType\":\"ALL\"}}}]"
```

---

## Step 5: Create Lambda Function

### 5.1 Prepare Lambda Deployment Package

Lambda functions need dependencies packaged as a ZIP file:

```bash
# Navigate to project directory
cd projects/psychology/behavioral-analysis/tier-2

# Create deployment directory
mkdir -p lambda_deployment
cd lambda_deployment

# Copy Lambda function
cp ../scripts/lambda_function.py .

# Install dependencies to this directory
pip install --target . \
  numpy \
  scipy \
  pandas \
  boto3

# Note: For scipy/numpy, you may need to use a Lambda layer
# See Step 5.3 for Lambda layer instructions
```

### 5.2 (Recommended) Use Lambda Layer for Scientific Libraries

Scientific libraries (numpy, scipy) are large. Use an AWS Lambda Layer:

#### Option A: Use AWS Data Science Layer (Easiest)

AWS provides a pre-built layer with scientific libraries:

```bash
# Use AWS provided layer ARN for us-east-1
# ARN: arn:aws:lambda:us-east-1:668099181075:layer:AWSLambda-Python38-SciPy1x:115
```

#### Option B: Create Custom Layer

```bash
# Create layer directory
mkdir -p python/lib/python3.8/site-packages

# Install libraries
pip install -t python/lib/python3.8/site-packages \
  numpy scipy pandas

# Create ZIP
zip -r scipy-layer.zip python/

# Create layer
aws lambda publish-layer-version \
  --layer-name scipy-numpy-pandas \
  --description "Scientific libraries for behavioral analysis" \
  --zip-file fileb://scipy-layer.zip \
  --compatible-runtimes python3.8 python3.9

# Note the Layer ARN from output
```

### 5.3 Create Lambda Deployment Package (Without Heavy Libraries)

```bash
cd lambda_deployment

# Only package Lambda function and boto3
zip -r lambda_function.zip lambda_function.py

# Boto3 is already included in Lambda runtime
```

### 5.4 Create Lambda Function

```bash
# Get the IAM role ARN from Step 3.2
ROLE_ARN="arn:aws:iam::YOUR_ACCOUNT_ID:role/lambda-behavioral-processor"

# Create Lambda function
aws lambda create-function \
  --function-name analyze-behavioral-data \
  --runtime python3.8 \
  --role $ROLE_ARN \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://lambda_function.zip \
  --timeout 300 \
  --memory-size 512 \
  --region us-east-1

# Expected output:
# Function ARN: arn:aws:lambda:us-east-1:123456789012:function:analyze-behavioral-data
```

### 5.5 Attach Lambda Layer

```bash
# Using AWS provided layer
aws lambda update-function-configuration \
  --function-name analyze-behavioral-data \
  --layers arn:aws:lambda:us-east-1:668099181075:layer:AWSLambda-Python38-SciPy1x:115

# Or using custom layer
# aws lambda update-function-configuration \
#   --function-name analyze-behavioral-data \
#   --layers arn:aws:lambda:us-east-1:YOUR_ACCOUNT_ID:layer:scipy-numpy-pandas:1
```

### 5.6 Configure Environment Variables

```bash
aws lambda update-function-configuration \
  --function-name analyze-behavioral-data \
  --environment "Variables={DYNAMODB_TABLE=BehavioralAnalysis,S3_BUCKET=$BUCKET_NAME}"
```

### 5.7 Test Lambda Function

Create test event:

```bash
cat > test-event.json <<EOF
{
  "Records": [
    {
      "s3": {
        "bucket": {
          "name": "$BUCKET_NAME"
        },
        "object": {
          "key": "raw/test_participant.csv"
        }
      }
    }
  ]
}
EOF

# Invoke Lambda with test event
aws lambda invoke \
  --function-name analyze-behavioral-data \
  --payload file://test-event.json \
  --cli-binary-format raw-in-base64-out \
  response.json

# Check response
cat response.json
```

---

## Step 6: Configure S3 Event Notifications (Optional)

Automatically trigger Lambda when new data is uploaded to S3:

### 6.1 Grant S3 Permission to Invoke Lambda

```bash
aws lambda add-permission \
  --function-name analyze-behavioral-data \
  --principal s3.amazonaws.com \
  --statement-id s3-trigger \
  --action "lambda:InvokeFunction" \
  --source-arn arn:aws:s3:::$BUCKET_NAME \
  --source-account YOUR_ACCOUNT_ID
```

### 6.2 Create S3 Notification Configuration

```bash
cat > s3-notification.json <<EOF
{
  "LambdaFunctionConfigurations": [
    {
      "Id": "BehavioralDataUpload",
      "LambdaFunctionArn": "arn:aws:lambda:us-east-1:YOUR_ACCOUNT_ID:function:analyze-behavioral-data",
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

# Apply notification configuration
aws s3api put-bucket-notification-configuration \
  --bucket $BUCKET_NAME \
  --notification-configuration file://s3-notification.json
```

---

## Step 7: Configure Athena (Optional)

For SQL queries on results:

### 7.1 Create Athena Workgroup

```bash
# Create S3 bucket for Athena results
ATHENA_BUCKET="athena-results-$(date +%s)"
aws s3 mb s3://$ATHENA_BUCKET --region us-east-1

# Create workgroup
aws athena create-work-group \
  --name behavioral-analysis \
  --configuration "ResultConfigurationUpdates={OutputLocation=s3://$ATHENA_BUCKET/}" \
  --region us-east-1
```

### 7.2 Create Athena Database

```bash
# Execute query to create database
aws athena start-query-execution \
  --query-string "CREATE DATABASE IF NOT EXISTS behavioral_db" \
  --work-group behavioral-analysis \
  --region us-east-1
```

### 7.3 Create External Table

Note: This requires exporting DynamoDB to S3 first, or using DynamoDB connector.

For simplicity, we'll skip this in Tier 2. Use `query_results.py` instead.

---

## Step 8: Verify Setup

### 8.1 Checklist

Run through this checklist to ensure everything is configured:

- [ ] S3 bucket created: `behavioral-data-xxxxx`
- [ ] S3 folders created: `raw/`, `metadata/`, `logs/`
- [ ] IAM role created: `lambda-behavioral-processor`
- [ ] IAM policy attached with S3, DynamoDB, CloudWatch permissions
- [ ] DynamoDB table created: `BehavioralAnalysis`
- [ ] Lambda function created: `analyze-behavioral-data`
- [ ] Lambda layer attached (scipy, numpy, pandas)
- [ ] Lambda environment variables set
- [ ] Lambda timeout set to 300 seconds
- [ ] Lambda memory set to 512 MB
- [ ] (Optional) S3 event notification configured
- [ ] (Optional) Athena workgroup created

### 8.2 Test End-to-End

```bash
# 1. Create sample CSV file
cat > sample_participant.csv <<EOF
participant_id,trial,task_type,stimulus,response,rt,accuracy
sub001,1,stroop,congruent,left,450,1
sub001,2,stroop,incongruent,right,620,1
sub001,3,stroop,congruent,left,430,1
sub001,4,stroop,incongruent,right,680,0
EOF

# 2. Upload to S3
aws s3 cp sample_participant.csv s3://$BUCKET_NAME/raw/sub001_stroop.csv

# 3. Wait a few seconds for Lambda to process

# 4. Check DynamoDB for results
aws dynamodb get-item \
  --table-name BehavioralAnalysis \
  --key '{"participant_id": {"S": "sub001"}, "task_type": {"S": "stroop"}}'

# 5. Check Lambda logs
aws logs tail /aws/lambda/analyze-behavioral-data --follow
```

If you see results in DynamoDB, congratulations! Your setup is complete.

---

## Step 9: Save Configuration

Create a configuration file for easy access:

```bash
cat > config.env <<EOF
# AWS Configuration
export AWS_REGION=us-east-1
export S3_BUCKET=$BUCKET_NAME
export LAMBDA_FUNCTION=analyze-behavioral-data
export DYNAMODB_TABLE=BehavioralAnalysis
export IAM_ROLE=lambda-behavioral-processor

# Optional
export ATHENA_WORKGROUP=behavioral-analysis
export ATHENA_BUCKET=$ATHENA_BUCKET
EOF

# Load configuration
source config.env
```

---

## Troubleshooting

### Issue: "Role not found" when creating Lambda

**Solution:** Wait 10-30 seconds after creating IAM role before creating Lambda function. AWS IAM has eventual consistency.

### Issue: "AccessDenied" when Lambda writes to DynamoDB

**Solution:** Verify IAM policy includes DynamoDB permissions. Run:
```bash
aws iam get-role-policy \
  --role-name lambda-behavioral-processor \
  --policy-name BehavioralAnalysisPolicy
```

### Issue: Lambda times out

**Solution:** Increase timeout:
```bash
aws lambda update-function-configuration \
  --function-name analyze-behavioral-data \
  --timeout 300
```

### Issue: "Unable to import module 'lambda_function'"

**Solution:**
1. Ensure ZIP file has `lambda_function.py` at root level (not in subdirectory)
2. Verify handler is set to `lambda_function.lambda_handler`

### Issue: "No module named 'numpy'" or "No module named 'scipy'"

**Solution:** Attach Lambda layer with scientific libraries (see Step 5.2)

### Issue: Lambda invocation fails with no error

**Solution:** Check CloudWatch logs:
```bash
aws logs tail /aws/lambda/analyze-behavioral-data --follow
```

---

## Cost Estimate for Setup

**Setup costs:** $0 (no charges for creating resources)

**Ongoing costs:**
- S3 storage: ~$0.023/GB/month
- Lambda invocations: ~$0.20 per 1M requests
- Lambda compute: ~$0.0000166667 per GB-second
- DynamoDB: ~$1.25/GB/month (on-demand)

**Typical project cost:** $7-13 (as specified in README.md)

---

## Next Steps

1. **Run the analysis:** Follow README.md to upload data and analyze results
2. **Explore the notebook:** `notebooks/behavioral_analysis.ipynb`
3. **Customize:** Modify Lambda function for your specific analysis needs
4. **Monitor costs:** Set up billing alerts in AWS Console

---

## Cleanup

When you're done, follow `cleanup_guide.md` to delete all resources and stop charges.

---

**Setup complete!** You're ready to analyze behavioral data in the cloud.

Return to [README.md](README.md) to continue with the project.
