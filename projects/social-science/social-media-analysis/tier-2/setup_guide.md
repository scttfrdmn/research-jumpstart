# AWS Setup Guide - Social Media Sentiment Analysis Tier 2

This guide walks you through setting up all AWS resources needed for the social media sentiment analysis project. Estimated time: 30-40 minutes.

---

## Prerequisites

Before starting, ensure you have:
- ✅ AWS account with billing enabled
- ✅ AWS CLI installed (`aws --version`)
- ✅ AWS credentials configured (`aws configure`)
- ✅ Python 3.8+ installed
- ✅ $15 budget available for testing

---

## Step 1: Create S3 Bucket

S3 stores your social media data and processing results.

### Option A: Using AWS Console (Recommended for beginners)

1. Go to [AWS S3 Console](https://console.aws.amazon.com/s3/)
2. Click "Create bucket"
3. Enter bucket name: `social-media-data-{your-name}-{date}`
   - Example: `social-media-data-alice-20250114`
   - Must be globally unique
   - Must be lowercase
   - No special characters except hyphens
4. Select region: **us-east-1** (recommended for Comprehend availability)
5. **Block Public Access**: Keep all settings enabled (recommended)
6. **Versioning**: Disabled (optional)
7. **Encryption**: Server-side encryption enabled (default)
8. Click "Create bucket"

### Option B: Using AWS CLI

```bash
# Create S3 bucket
BUCKET_NAME="social-media-data-$(whoami)-$(date +%s)"
aws s3 mb "s3://$BUCKET_NAME" --region us-east-1

# Verify bucket created
aws s3 ls | grep social-media-data

# Save bucket name for later
echo "BUCKET_NAME=$BUCKET_NAME" > .env
echo "Bucket created: $BUCKET_NAME"
```

### Create Folder Structure

```bash
# Create folders for organization
aws s3api put-object --bucket "$BUCKET_NAME" --key "raw/"
aws s3api put-object --bucket "$BUCKET_NAME" --key "processed/"
aws s3api put-object --bucket "$BUCKET_NAME" --key "exports/"

# Verify
aws s3 ls "s3://$BUCKET_NAME/" --recursive
```

**Save your bucket name!** You'll use it in later steps.

---

## Step 2: Create DynamoDB Table

DynamoDB stores sentiment analysis results for fast querying.

### Option A: Using AWS Console

1. Go to [DynamoDB Console](https://console.aws.amazon.com/dynamodb/)
2. Click "Create table"
3. Table name: `SocialMediaPosts`
4. Partition key: `post_id` (String)
5. Sort key: `timestamp` (Number)
6. Table settings: **On-demand** (recommended for variable workload)
   - Alternative: Provisioned with 5 RCU / 5 WCU for cost savings
7. Encryption: **AWS owned key** (default, free)
8. Click "Create table"
9. Wait 1-2 minutes for table to be created

### Option B: Using AWS CLI

```bash
# Create DynamoDB table with on-demand billing
aws dynamodb create-table \
    --table-name SocialMediaPosts \
    --attribute-definitions \
        AttributeName=post_id,AttributeType=S \
        AttributeName=timestamp,AttributeType=N \
    --key-schema \
        AttributeName=post_id,KeyType=HASH \
        AttributeName=timestamp,KeyType=RANGE \
    --billing-mode PAY_PER_REQUEST \
    --region us-east-1

# Wait for table to be active
aws dynamodb wait table-exists --table-name SocialMediaPosts

# Verify table created
aws dynamodb describe-table --table-name SocialMediaPosts \
    --query 'Table.TableStatus'
```

### Create Global Secondary Index (Optional - for querying by sentiment)

```bash
# Add GSI for querying by sentiment
aws dynamodb update-table \
    --table-name SocialMediaPosts \
    --attribute-definitions \
        AttributeName=sentiment,AttributeType=S \
        AttributeName=timestamp,AttributeType=N \
    --global-secondary-index-updates \
        "[{
            \"Create\": {
                \"IndexName\": \"sentiment-timestamp-index\",
                \"KeySchema\": [
                    {\"AttributeName\": \"sentiment\", \"KeyType\": \"HASH\"},
                    {\"AttributeName\": \"timestamp\", \"KeyType\": \"RANGE\"}
                ],
                \"Projection\": {\"ProjectionType\": \"ALL\"},
                \"ProvisionedThroughput\": {
                    \"ReadCapacityUnits\": 5,
                    \"WriteCapacityUnits\": 5
                }
            }
        }]"
```

---

## Step 3: Create IAM Role for Lambda

Lambda needs permissions to read/write S3, DynamoDB, call Comprehend, and write logs.

### Option A: Using AWS Console

1. Go to [IAM Console](https://console.aws.amazon.com/iam/home)
2. Click "Roles" in left menu
3. Click "Create role"
4. **Trusted entity**: AWS service → Lambda
5. Click "Next"
6. **Add permissions** - Search and select:
   - `AWSLambdaBasicExecutionRole` (CloudWatch logs)
   - `AmazonS3FullAccess` (S3 read/write)
   - `AmazonDynamoDBFullAccess` (DynamoDB read/write)
   - `ComprehendReadOnly` (Sentiment analysis)
7. Click "Next"
8. Role name: `lambda-social-analysis`
9. Description: `Lambda role for social media sentiment analysis`
10. Click "Create role"

### Option B: Using AWS CLI

```bash
# Create trust policy for Lambda
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

# Create IAM role
aws iam create-role \
  --role-name lambda-social-analysis \
  --assume-role-policy-document file://trust-policy.json \
  --description "Lambda role for social media sentiment analysis"

# Attach managed policies
aws iam attach-role-policy \
  --role-name lambda-social-analysis \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam attach-role-policy \
  --role-name lambda-social-analysis \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam attach-role-policy \
  --role-name lambda-social-analysis \
  --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess

aws iam attach-role-policy \
  --role-name lambda-social-analysis \
  --policy-arn arn:aws:iam::aws:policy/ComprehendReadOnly

# Get role ARN (you'll need this)
ROLE_ARN=$(aws iam get-role --role-name lambda-social-analysis \
  --query 'Role.Arn' --output text)
echo "ROLE_ARN=$ROLE_ARN" >> .env
echo "IAM Role ARN: $ROLE_ARN"
```

### Verify Role Created

```bash
aws iam get-role --role-name lambda-social-analysis
```

**Save your role ARN!** Format: `arn:aws:iam::ACCOUNT_ID:role/lambda-social-analysis`

---

## Step 4: Create Lambda Function

Lambda processes your social media posts with sentiment analysis.

### Option A: Using AWS Console

1. Go to [Lambda Console](https://console.aws.amazon.com/lambda/)
2. Click "Create function"
3. Function name: `analyze-sentiment`
4. Runtime: **Python 3.11**
5. Architecture: **x86_64**
6. Permissions: Choose existing role
   - Select `lambda-social-analysis`
7. Click "Create function"

#### Upload Function Code:

1. In function page, scroll to "Code source"
2. Delete default code
3. Copy entire contents of `scripts/lambda_function.py`
4. Paste into editor (lambda_function.py file)
5. Click "Deploy"

#### Configure Function Settings:

1. Click "Configuration" tab
2. Click "General configuration" → "Edit"
3. **Timeout**: Change to **30 seconds**
4. **Memory**: Set to **256 MB**
5. Click "Save"

#### Set Environment Variables:

1. In Configuration tab
2. Click "Environment variables" → "Edit"
3. Add variables:
   - `BUCKET_NAME` = `social-media-data-xxxx` (your bucket)
   - `DYNAMODB_TABLE` = `SocialMediaPosts`
   - `AWS_REGION` = `us-east-1`
4. Click "Save"

### Option B: Using AWS CLI

```bash
# Navigate to scripts directory
cd scripts/

# Create deployment package
zip lambda_function.zip lambda_function.py

# Create Lambda function
aws lambda create-function \
  --function-name analyze-sentiment \
  --runtime python3.11 \
  --role "$ROLE_ARN" \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://lambda_function.zip \
  --timeout 30 \
  --memory-size 256 \
  --environment "Variables={
    BUCKET_NAME=$BUCKET_NAME,
    DYNAMODB_TABLE=SocialMediaPosts,
    AWS_REGION=us-east-1
  }" \
  --region us-east-1

# Verify function created
aws lambda get-function --function-name analyze-sentiment

cd ..
```

### Test Lambda Function

```bash
# Create test event
cat > test-event.json << 'EOF'
{
  "Records": [
    {
      "s3": {
        "bucket": {
          "name": "social-media-data-test"
        },
        "object": {
          "key": "raw/sample_post.json"
        }
      }
    }
  ]
}
EOF

# Invoke Lambda with test event (update bucket name)
aws lambda invoke \
  --function-name analyze-sentiment \
  --payload file://test-event.json \
  --cli-binary-format raw-in-base64-out \
  response.json

# Check response
cat response.json
```

---

## Step 5: Configure S3 Event Trigger

Set up S3 to automatically trigger Lambda when new data is uploaded.

### Option A: Using AWS Console

1. Go to [S3 Console](https://console.aws.amazon.com/s3/)
2. Open your bucket: `social-media-data-xxxx`
3. Click "Properties" tab
4. Scroll to "Event notifications"
5. Click "Create event notification"
6. Event name: `trigger-sentiment-analysis`
7. Prefix: `raw/`
8. Suffix: `.json`
9. Event types: Check "All object create events" (`s3:ObjectCreated:*`)
10. Destination: **Lambda function**
11. Lambda function: `analyze-sentiment`
12. Click "Save changes"

### Option B: Using AWS CLI

```bash
# Add permission for S3 to invoke Lambda
aws lambda add-permission \
  --function-name analyze-sentiment \
  --statement-id AllowS3Invoke \
  --action lambda:InvokeFunction \
  --principal s3.amazonaws.com \
  --source-arn "arn:aws:s3:::$BUCKET_NAME"

# Create S3 event notification configuration
cat > notification.json << EOF
{
  "LambdaFunctionConfigurations": [
    {
      "Id": "trigger-sentiment-analysis",
      "LambdaFunctionArn": "arn:aws:lambda:us-east-1:$(aws sts get-caller-identity --query Account --output text):function:analyze-sentiment",
      "Events": ["s3:ObjectCreated:*"],
      "Filter": {
        "Key": {
          "FilterRules": [
            {"Name": "prefix", "Value": "raw/"},
            {"Name": "suffix", "Value": ".json"}
          ]
        }
      }
    }
  ]
}
EOF

# Apply notification configuration
aws s3api put-bucket-notification-configuration \
  --bucket "$BUCKET_NAME" \
  --notification-configuration file://notification.json

# Verify notification configured
aws s3api get-bucket-notification-configuration \
  --bucket "$BUCKET_NAME"
```

---

## Step 6: Setup Athena (Optional - for SQL queries)

Athena allows you to query S3 data using SQL.

### Using AWS Console

1. Go to [Athena Console](https://console.aws.amazon.com/athena/)
2. Click "Get Started"
3. Click "Settings" tab
4. Query result location: `s3://social-media-data-xxxx/athena-results/`
5. Click "Save"

### Create Athena Database and Table

```sql
-- Create database
CREATE DATABASE social_media_db;

-- Create external table pointing to S3 exports
CREATE EXTERNAL TABLE IF NOT EXISTS social_media_db.posts (
  post_id STRING,
  timestamp BIGINT,
  text STRING,
  user_id STRING,
  username STRING,
  sentiment STRING,
  positive_score DOUBLE,
  negative_score DOUBLE,
  neutral_score DOUBLE,
  mixed_score DOUBLE,
  hashtags ARRAY<STRING>,
  mentions ARRAY<STRING>
)
ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
LOCATION 's3://social-media-data-xxxx/exports/';

-- Test query
SELECT sentiment, COUNT(*) as count
FROM social_media_db.posts
GROUP BY sentiment;
```

---

## Step 7: Verify Setup

Test that all components are working together.

### Test 1: Upload Sample Data

```bash
# Create sample post
cat > sample_post.json << 'EOF'
[
  {
    "post_id": "test001",
    "text": "I love this new product! It's amazing and works perfectly.",
    "timestamp": 1705246800,
    "user_id": "user123",
    "username": "happy_customer"
  }
]
EOF

# Upload to S3 (this should trigger Lambda)
aws s3 cp sample_post.json "s3://$BUCKET_NAME/raw/"

# Wait a few seconds for processing
sleep 10
```

### Test 2: Check Lambda Logs

```bash
# View CloudWatch logs
aws logs tail /aws/lambda/analyze-sentiment --follow
```

### Test 3: Query DynamoDB

```bash
# Scan DynamoDB table
aws dynamodb scan --table-name SocialMediaPosts --limit 5

# Or query specific post
aws dynamodb get-item \
  --table-name SocialMediaPosts \
  --key '{"post_id": {"S": "test001"}, "timestamp": {"N": "1705246800"}}'
```

### Test 4: Check Results

```python
# Run this in Python
import boto3

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table('SocialMediaPosts')

response = table.scan(Limit=10)
for item in response['Items']:
    print(f"Post: {item['post_id']}")
    print(f"Sentiment: {item.get('sentiment', 'N/A')}")
    print(f"Positive score: {item.get('positive_score', 'N/A')}")
    print("---")
```

---

## Cost Estimation

Based on this setup:

| Component | Free Tier | Beyond Free Tier |
|-----------|-----------|------------------|
| S3 Storage (1GB) | ✅ Free for 12 months | $0.023/GB/month |
| S3 Requests (1000) | ✅ Free for 12 months | $0.005/1000 requests |
| Lambda (1000 invocations) | ✅ Free always | $0.20/1M requests |
| DynamoDB (25GB storage) | ✅ Free always | $1.25/GB/month |
| Comprehend (1000 units) | ✅ 50k free for 12 months | $0.0001/unit |
| CloudWatch Logs | ✅ 5GB free always | $0.50/GB |

**Expected monthly cost for this project: $3-7** (after free tier)

---

## Security Best Practices

### 1. Least Privilege IAM Policies

Instead of `AmazonS3FullAccess`, use a custom policy:

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
      "Resource": "arn:aws:s3:::social-media-data-*/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:PutItem",
        "dynamodb:GetItem",
        "dynamodb:Query"
      ],
      "Resource": "arn:aws:dynamodb:us-east-1:*:table/SocialMediaPosts"
    },
    {
      "Effect": "Allow",
      "Action": "comprehend:DetectSentiment",
      "Resource": "*"
    }
  ]
}
```

### 2. Enable S3 Bucket Encryption

```bash
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

### 3. Enable DynamoDB Point-in-Time Recovery

```bash
aws dynamodb update-continuous-backups \
  --table-name SocialMediaPosts \
  --point-in-time-recovery-specification PointInTimeRecoveryEnabled=true
```

---

## Troubleshooting

### Issue: Lambda cannot write to DynamoDB

**Error:** `AccessDeniedException: User: arn:aws:sts::xxx:assumed-role/lambda-social-analysis is not authorized to perform: dynamodb:PutItem`

**Solution:**
```bash
# Verify IAM role has DynamoDB permissions
aws iam list-attached-role-policies --role-name lambda-social-analysis

# If missing, attach policy
aws iam attach-role-policy \
  --role-name lambda-social-analysis \
  --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess
```

### Issue: Lambda timeout

**Error:** `Task timed out after 3.00 seconds`

**Solution:**
```bash
# Increase timeout to 30 seconds
aws lambda update-function-configuration \
  --function-name analyze-sentiment \
  --timeout 30
```

### Issue: Comprehend API throttling

**Error:** `ThrottlingException: Rate exceeded`

**Solution:**
- Implement exponential backoff in Lambda code
- Request service limit increase via AWS Support
- Process posts in smaller batches

### Issue: S3 trigger not working

**Error:** Lambda not invoked when uploading to S3

**Solution:**
```bash
# Verify Lambda has permission
aws lambda get-policy --function-name analyze-sentiment

# Check S3 notification configuration
aws s3api get-bucket-notification-configuration --bucket "$BUCKET_NAME"

# Re-add permission if needed
aws lambda add-permission \
  --function-name analyze-sentiment \
  --statement-id AllowS3Invoke \
  --action lambda:InvokeFunction \
  --principal s3.amazonaws.com \
  --source-arn "arn:aws:s3:::$BUCKET_NAME"
```

---

## Next Steps

1. ✅ All AWS resources created
2. ✅ Lambda function deployed
3. ✅ S3 trigger configured
4. ✅ DynamoDB table ready

**Now you can:**
- Run `scripts/upload_to_s3.py` to upload social media data
- Open `notebooks/social_analysis.ipynb` to analyze results
- Use `scripts/query_results.py` to query DynamoDB

---

## Environment Variables Summary

Save these to `.env` file:

```bash
# AWS Configuration
AWS_REGION=us-east-1
BUCKET_NAME=social-media-data-xxxx
DYNAMODB_TABLE=SocialMediaPosts
LAMBDA_FUNCTION=analyze-sentiment

# IAM
ROLE_ARN=arn:aws:iam::ACCOUNT_ID:role/lambda-social-analysis
```

---

## Quick Reference Commands

```bash
# Check Lambda logs
aws logs tail /aws/lambda/analyze-sentiment --follow

# Query DynamoDB
aws dynamodb scan --table-name SocialMediaPosts --limit 10

# List S3 objects
aws s3 ls "s3://$BUCKET_NAME/" --recursive

# Test Lambda manually
aws lambda invoke \
  --function-name analyze-sentiment \
  --payload file://test-event.json \
  response.json

# Check costs
aws ce get-cost-and-usage \
  --time-period Start=2025-01-01,End=2025-01-31 \
  --granularity MONTHLY \
  --metrics "BlendedCost"
```

---

**Setup complete!** Proceed to the main [README.md](README.md) to start analyzing social media data.

**Last updated:** 2025-11-14 | Research Jumpstart v1.0.0
