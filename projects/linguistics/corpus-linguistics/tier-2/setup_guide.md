# AWS Setup Guide - Corpus Linguistics Tier 2

This guide walks you through setting up all AWS resources needed for the Corpus Linguistics project. Follow each step carefully.

**Total setup time:** 30-40 minutes (first time)

---

## Prerequisites

Before starting, ensure you have:

1. **AWS Account** - https://aws.amazon.com/
2. **AWS CLI installed and configured**
   ```bash
   aws --version
   aws configure  # Enter credentials
   ```
3. **Python 3.8+** installed
4. **Project dependencies** installed
   ```bash
   pip install -r requirements.txt
   ```

---

## Overview

You'll create these AWS resources:

1. S3 bucket for corpus storage
2. DynamoDB table for linguistic metadata
3. IAM role for Lambda with necessary permissions
4. Lambda function for NLP processing
5. (Optional) Athena workgroup for SQL queries

**Estimated cost:** $6-12 for the entire project

---

## Step 1: Create S3 Bucket (5 minutes)

### 1.1 Generate Unique Bucket Name

```bash
# Generate unique bucket name (save this!)
export BUCKET_NAME="linguistic-corpus-$(date +%s)-$(whoami)"
echo "Your bucket name: $BUCKET_NAME"
```

**Example:** `linguistic-corpus-1731614400-username`

### 1.2 Create Bucket via AWS Console

1. Open AWS Console: https://console.aws.amazon.com/s3/
2. Click **Create bucket**
3. Enter bucket name: `linguistic-corpus-{your-unique-id}`
4. Choose region: **us-east-1** (recommended for lower costs)
5. **Block Public Access settings**: Keep all boxes CHECKED (block all public access)
6. **Bucket Versioning**: Disabled (save costs)
7. **Tags** (optional):
   - Key: `Project` Value: `corpus-linguistics`
   - Key: `Tier` Value: `2`
8. Click **Create bucket**

### 1.3 Create Bucket via AWS CLI (Alternative)

```bash
# Create bucket
aws s3 mb s3://$BUCKET_NAME --region us-east-1

# Verify bucket exists
aws s3 ls | grep linguistic-corpus

# Expected output: linguistic-corpus-{your-id}
```

### 1.4 Create Folder Structure

```bash
# Create folder structure
aws s3api put-object --bucket $BUCKET_NAME --key raw/
aws s3api put-object --bucket $BUCKET_NAME --key processed/
aws s3api put-object --bucket $BUCKET_NAME --key logs/

# Verify structure
aws s3 ls s3://$BUCKET_NAME/
```

**Expected output:**
```
                           PRE processed/
                           PRE raw/
                           PRE logs/
```

---

## Step 2: Create DynamoDB Table (5 minutes)

### 2.1 Define Table Schema

We'll store linguistic analysis results with this schema:

- **Primary Key**: `text_id` (String) - Unique identifier for each text
- **Sort Key**: None (simple key schema)
- **Attributes**: language, genre, register, word_count, lexical_diversity, etc.

### 2.2 Create Table via AWS Console

1. Open AWS Console: https://console.aws.amazon.com/dynamodb/
2. Click **Create table**
3. **Table name**: `LinguisticAnalysis`
4. **Partition key**: `text_id` (String)
5. **Sort key**: Leave empty
6. **Table settings**: Default settings
7. **Table class**: DynamoDB Standard
8. **Capacity mode**: On-demand (pay per request)
9. **Encryption**: AWS owned key (free)
10. Click **Create table**

### 2.3 Create Table via AWS CLI (Alternative)

```bash
# Create DynamoDB table
aws dynamodb create-table \
    --table-name LinguisticAnalysis \
    --attribute-definitions \
        AttributeName=text_id,AttributeType=S \
    --key-schema \
        AttributeName=text_id,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --region us-east-1

# Wait for table to be active
aws dynamodb wait table-exists --table-name LinguisticAnalysis

# Verify table status
aws dynamodb describe-table --table-name LinguisticAnalysis --query 'Table.TableStatus'
```

**Expected output:** `"ACTIVE"`

### 2.4 Create Global Secondary Index (Optional, for advanced queries)

```bash
# Add GSI for querying by language
aws dynamodb update-table \
    --table-name LinguisticAnalysis \
    --attribute-definitions \
        AttributeName=language,AttributeType=S \
        AttributeName=genre,AttributeType=S \
    --global-secondary-index-updates \
        "[{\"Create\":{\"IndexName\":\"language-genre-index\",\"KeySchema\":[{\"AttributeName\":\"language\",\"KeyType\":\"HASH\"},{\"AttributeName\":\"genre\",\"KeyType\":\"RANGE\"}],\"Projection\":{\"ProjectionType\":\"ALL\"}}}]"
```

---

## Step 3: Create IAM Role for Lambda (10 minutes)

### 3.1 Create Trust Policy

Lambda needs permission to assume this role.

```bash
# Create trust policy file
cat > /tmp/lambda-trust-policy.json <<EOF
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
# Create IAM role
aws iam create-role \
    --role-name lambda-linguistic-processor \
    --assume-role-policy-document file:///tmp/lambda-trust-policy.json \
    --description "Lambda role for linguistic corpus processing"

# Save role ARN (you'll need this later)
export ROLE_ARN=$(aws iam get-role --role-name lambda-linguistic-processor --query 'Role.Arn' --output text)
echo "Role ARN: $ROLE_ARN"
```

### 3.3 Create and Attach Permissions Policy

```bash
# Create permissions policy
cat > /tmp/lambda-permissions-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
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
        "dynamodb:GetItem",
        "dynamodb:Query",
        "dynamodb:Scan"
      ],
      "Resource": [
        "arn:aws:dynamodb:us-east-1:*:table/LinguisticAnalysis",
        "arn:aws:dynamodb:us-east-1:*:table/LinguisticAnalysis/index/*"
      ]
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
EOF

# Attach policy to role
aws iam put-role-policy \
    --role-name lambda-linguistic-processor \
    --policy-name linguistic-processor-policy \
    --policy-document file:///tmp/lambda-permissions-policy.json

# Attach AWS managed policy for basic Lambda execution
aws iam attach-role-policy \
    --role-name lambda-linguistic-processor \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
```

### 3.4 Wait for IAM Propagation

```bash
# IAM changes take time to propagate (30-60 seconds)
echo "Waiting for IAM role to propagate..."
sleep 60
```

---

## Step 4: Deploy Lambda Function (15 minutes)

### 4.1 Prepare Lambda Deployment Package

Lambda needs NLTK and its data. We'll create a deployment package.

```bash
# Create deployment directory
mkdir -p /tmp/lambda-package
cd /tmp/lambda-package

# Copy lambda function code
cp /path/to/your/scripts/lambda_function.py .

# Install dependencies
pip install -t . nltk boto3

# Download NLTK data
python -c "import nltk; nltk.download('punkt', download_dir='./nltk_data'); nltk.download('averaged_perceptron_tagger', download_dir='./nltk_data'); nltk.download('wordnet', download_dir='./nltk_data')"

# Create deployment package
zip -r lambda-package.zip .

# Move to project directory
mv lambda-package.zip ~/corpus-linguistics-lambda.zip
cd ~
```

**Package size:** ~10-15MB (with NLTK data)

### 4.2 Create Lambda Function via AWS Console

1. Open AWS Console: https://console.aws.amazon.com/lambda/
2. Click **Create function**
3. Choose **Author from scratch**
4. **Function name**: `analyze-linguistic-corpus`
5. **Runtime**: Python 3.11
6. **Architecture**: x86_64
7. **Permissions**: Choose **Use an existing role**
8. **Existing role**: `lambda-linguistic-processor`
9. Click **Create function**

### 4.3 Upload Code

1. In function page, scroll to **Code source**
2. Click **Upload from** → **.zip file**
3. Choose your `corpus-linguistics-lambda.zip` file
4. Click **Save**

### 4.4 Configure Lambda Settings

1. **General configuration**:
   - **Memory**: 512 MB (NLTK needs more memory)
   - **Timeout**: 5 minutes (300 seconds)
   - **Ephemeral storage**: 512 MB

2. **Environment variables**:
   - Key: `DYNAMODB_TABLE`, Value: `LinguisticAnalysis`
   - Key: `S3_BUCKET`, Value: `{your-bucket-name}`
   - Key: `NLTK_DATA`, Value: `/var/task/nltk_data`

3. **Save changes**

### 4.5 Create Lambda via AWS CLI (Alternative)

```bash
# Create Lambda function
aws lambda create-function \
    --function-name analyze-linguistic-corpus \
    --runtime python3.11 \
    --role $ROLE_ARN \
    --handler lambda_function.lambda_handler \
    --zip-file fileb://corpus-linguistics-lambda.zip \
    --timeout 300 \
    --memory-size 512 \
    --environment Variables="{DYNAMODB_TABLE=LinguisticAnalysis,S3_BUCKET=$BUCKET_NAME,NLTK_DATA=/var/task/nltk_data}" \
    --region us-east-1

# Verify function exists
aws lambda get-function --function-name analyze-linguistic-corpus
```

### 4.6 Test Lambda Function

Create test event:

```json
{
  "Records": [
    {
      "s3": {
        "bucket": {
          "name": "your-bucket-name"
        },
        "object": {
          "key": "raw/english/academic/sample.txt"
        }
      }
    }
  ]
}
```

1. In Lambda console, click **Test** tab
2. Create new test event with name `test-event`
3. Paste JSON above (update bucket name)
4. Click **Test**
5. Check execution results (should succeed or fail with clear error)

---

## Step 5: Configure S3 Event Trigger (Optional) (5 minutes)

To automatically process files when uploaded to S3:

### 5.1 Add S3 Trigger via Console

1. In Lambda function page, click **Add trigger**
2. Select **S3**
3. **Bucket**: Select your bucket
4. **Event type**: All object create events
5. **Prefix**: `raw/` (only trigger for files in raw folder)
6. **Suffix**: `.txt` (only trigger for text files)
7. Acknowledge recursive invocation warning
8. Click **Add**

### 5.2 Add S3 Trigger via CLI (Alternative)

```bash
# Create S3 notification configuration
cat > /tmp/s3-notification.json <<EOF
{
  "LambdaFunctionConfigurations": [
    {
      "Id": "corpus-processing-trigger",
      "LambdaFunctionArn": "$(aws lambda get-function --function-name analyze-linguistic-corpus --query 'Configuration.FunctionArn' --output text)",
      "Events": ["s3:ObjectCreated:*"],
      "Filter": {
        "Key": {
          "FilterRules": [
            {"Name": "prefix", "Value": "raw/"},
            {"Name": "suffix", "Value": ".txt"}
          ]
        }
      }
    }
  ]
}
EOF

# Grant S3 permission to invoke Lambda
aws lambda add-permission \
    --function-name analyze-linguistic-corpus \
    --statement-id s3-invoke-lambda \
    --action lambda:InvokeFunction \
    --principal s3.amazonaws.com \
    --source-arn arn:aws:s3:::$BUCKET_NAME

# Add notification configuration to S3 bucket
aws s3api put-bucket-notification-configuration \
    --bucket $BUCKET_NAME \
    --notification-configuration file:///tmp/s3-notification.json
```

---

## Step 6: Set Up Athena (Optional) (5 minutes)

For SQL queries on linguistic data.

### 6.1 Create Athena Workgroup

1. Open AWS Console: https://console.aws.amazon.com/athena/
2. First-time setup:
   - Create S3 bucket for query results: `s3://aws-athena-query-results-{account-id}-us-east-1`
   - Or use existing bucket
3. Click **Workgroups** → **Create workgroup**
4. **Name**: `corpus-linguistics`
5. **Query result location**: `s3://your-results-bucket/athena/`
6. Click **Create workgroup**

### 6.2 Create Athena Table (DynamoDB Connector)

Athena can query DynamoDB directly.

```sql
-- In Athena Query Editor
CREATE EXTERNAL TABLE linguistic_analysis (
    text_id string,
    language string,
    genre string,
    register string,
    word_count int,
    unique_words int,
    type_token_ratio double,
    avg_sentence_length double
)
STORED BY 'org.apache.hadoop.hive.dynamodb.DynamoDBStorageHandler'
TBLPROPERTIES (
    "dynamodb.table.name" = "LinguisticAnalysis",
    "dynamodb.column.mapping" = "text_id:text_id,language:language,genre:genre,register:register,word_count:word_count,unique_words:unique_words,type_token_ratio:type_token_ratio,avg_sentence_length:avg_sentence_length"
);
```

**Note:** Athena queries on DynamoDB consume read capacity. Use sparingly.

---

## Step 7: Verify Setup (5 minutes)

### 7.1 Check S3 Bucket

```bash
# List bucket contents
aws s3 ls s3://$BUCKET_NAME/

# Expected output:
#                            PRE processed/
#                            PRE raw/
#                            PRE logs/
```

### 7.2 Check DynamoDB Table

```bash
# Describe table
aws dynamodb describe-table --table-name LinguisticAnalysis --query 'Table.[TableName,TableStatus,ItemCount]'

# Expected output:
# [
#     "LinguisticAnalysis",
#     "ACTIVE",
#     0
# ]
```

### 7.3 Check Lambda Function

```bash
# Get Lambda configuration
aws lambda get-function-configuration --function-name analyze-linguistic-corpus --query '[FunctionName,Runtime,Timeout,MemorySize]'

# Expected output:
# [
#     "analyze-linguistic-corpus",
#     "python3.11",
#     300,
#     512
# ]
```

### 7.4 Check IAM Role

```bash
# List attached policies
aws iam list-attached-role-policies --role-name lambda-linguistic-processor

# Expected output includes:
# - AWSLambdaBasicExecutionRole
```

---

## Step 8: Test End-to-End (5 minutes)

### 8.1 Create Sample Text File

```bash
# Create sample text
cat > /tmp/sample.txt <<EOF
This is a sample text for linguistic analysis. The text contains multiple sentences with various parts of speech. We will analyze word frequencies, collocations, and lexical diversity. Natural language processing enables automated corpus linguistics at scale.
EOF
```

### 8.2 Upload to S3

```bash
# Upload sample text
aws s3 cp /tmp/sample.txt s3://$BUCKET_NAME/raw/english/sample/sample.txt

# Verify upload
aws s3 ls s3://$BUCKET_NAME/raw/english/sample/
```

### 8.3 Check Lambda Execution

```bash
# Wait 10 seconds for Lambda to process
sleep 10

# Check CloudWatch logs
aws logs tail /aws/lambda/analyze-linguistic-corpus --follow

# Look for successful execution
```

### 8.4 Verify DynamoDB Entry

```bash
# Query DynamoDB for results
aws dynamodb get-item \
    --table-name LinguisticAnalysis \
    --key '{"text_id": {"S": "english_sample_sample"}}' \
    --query 'Item'

# Should return linguistic analysis results
```

---

## Troubleshooting

### Issue: "Access Denied" errors

**Cause:** IAM permissions not propagated or incorrect

**Solution:**
```bash
# Wait for IAM propagation
sleep 60

# Verify role has correct policies
aws iam list-role-policies --role-name lambda-linguistic-processor
aws iam list-attached-role-policies --role-name lambda-linguistic-processor
```

### Issue: Lambda timeout

**Cause:** Processing takes longer than timeout setting

**Solution:**
```bash
# Increase timeout to 5 minutes
aws lambda update-function-configuration \
    --function-name analyze-linguistic-corpus \
    --timeout 300
```

### Issue: Lambda out of memory

**Cause:** NLTK/text processing requires more memory

**Solution:**
```bash
# Increase memory to 1024 MB
aws lambda update-function-configuration \
    --function-name analyze-linguistic-corpus \
    --memory-size 1024
```

### Issue: NLTK data not found

**Cause:** NLTK data not included in deployment package

**Solution:**
```bash
# Verify NLTK data in package
unzip -l corpus-linguistics-lambda.zip | grep nltk_data

# Re-download if missing
python -c "import nltk; nltk.download('punkt', download_dir='./nltk_data')"
```

### Issue: S3 trigger not firing

**Cause:** Notification configuration incorrect

**Solution:**
```bash
# Check notification configuration
aws s3api get-bucket-notification-configuration --bucket $BUCKET_NAME

# Re-add trigger (see Step 5)
```

---

## Cost Estimates

After setup, here's what you'll pay:

| Resource | Ongoing Cost | Notes |
|----------|--------------|-------|
| S3 Storage | ~$0.023/GB/month | Only pay for stored data |
| DynamoDB | ~$0 (free tier) | Under 25GB + 25 read/write units |
| Lambda | ~$0 (free tier) | 1M invocations free |
| Athena | ~$5/TB scanned | Optional, pay per query |
| CloudWatch | ~$0 (free tier) | 5GB logs free |

**Total ongoing cost:** ~$0-2/month if you keep data in S3

---

## Next Steps

Setup complete! Now you can:

1. **Upload corpus**: Run `python scripts/upload_to_s3.py`
2. **Process texts**: Files automatically processed by Lambda
3. **Query results**: Run `python scripts/query_results.py`
4. **Analyze**: Open `notebooks/corpus_analysis.ipynb`

---

## Reference: All Resources Created

Save this information for cleanup:

```bash
# S3 Bucket
BUCKET_NAME=linguistic-corpus-{your-id}

# DynamoDB Table
TABLE_NAME=LinguisticAnalysis

# Lambda Function
FUNCTION_NAME=analyze-linguistic-corpus

# IAM Role
ROLE_NAME=lambda-linguistic-processor

# Region
REGION=us-east-1
```

---

## Quick Start Commands

For future reference, here are all setup commands in one place:

```bash
# Set variables
export BUCKET_NAME="linguistic-corpus-$(date +%s)-$(whoami)"
export REGION="us-east-1"

# Create S3 bucket
aws s3 mb s3://$BUCKET_NAME --region $REGION

# Create DynamoDB table
aws dynamodb create-table \
    --table-name LinguisticAnalysis \
    --attribute-definitions AttributeName=text_id,AttributeType=S \
    --key-schema AttributeName=text_id,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --region $REGION

# Create IAM role (requires trust policy file)
aws iam create-role --role-name lambda-linguistic-processor \
    --assume-role-policy-document file:///tmp/lambda-trust-policy.json

# Attach policies (requires permissions policy file)
aws iam put-role-policy --role-name lambda-linguistic-processor \
    --policy-name linguistic-processor-policy \
    --policy-document file:///tmp/lambda-permissions-policy.json

# Create Lambda function (requires deployment package)
aws lambda create-function \
    --function-name analyze-linguistic-corpus \
    --runtime python3.11 \
    --role $(aws iam get-role --role-name lambda-linguistic-processor --query 'Role.Arn' --output text) \
    --handler lambda_function.lambda_handler \
    --zip-file fileb://corpus-linguistics-lambda.zip \
    --timeout 300 \
    --memory-size 512 \
    --environment Variables="{DYNAMODB_TABLE=LinguisticAnalysis,S3_BUCKET=$BUCKET_NAME}" \
    --region $REGION
```

---

**Setup complete!** Return to [README.md](README.md) to start analyzing your corpus.

**Last updated:** 2025-11-14
