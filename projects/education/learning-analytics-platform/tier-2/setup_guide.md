# AWS Setup Guide for Learning Analytics Platform

This guide provides step-by-step instructions to set up the AWS environment for the Learning Analytics Platform Tier 2 project.

**Total Setup Time:** 45-60 minutes

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

The S3 bucket will store raw student data, processed results, and Athena query results.

### Using AWS Management Console

1. Go to **S3** > **Buckets** > **Create Bucket**
2. Enter bucket name: `learning-analytics-{your-user-id}`
   - **Important:** Bucket names must be globally unique
   - Use lowercase letters and hyphens only
   - Example: `learning-analytics-edu-2025`
3. Region: Select your region (e.g., `us-east-1`)
4. **Block Public Access:** Keep all settings enabled (default)
5. **Versioning:** Optional (recommended for data recovery)
6. Click **Create Bucket**

### Using AWS CLI

```bash
# Replace {your-user-id} with your name/ID
export BUCKET_NAME="learning-analytics-$(whoami)-$(date +%s)"
aws s3 mb s3://${BUCKET_NAME} --region us-east-1

# Verify bucket creation
aws s3 ls | grep learning-analytics
```

### Create Bucket Folders

Create folder structure for organizing data:

```bash
# Replace with your bucket name
export BUCKET_NAME="learning-analytics-your-user-id"

# Raw student data
aws s3api put-object --bucket ${BUCKET_NAME} --key raw-data/

# Processed analytics results
aws s3api put-object --bucket ${BUCKET_NAME} --key processed-data/

# Athena query results
aws s3api put-object --bucket ${BUCKET_NAME} --key athena-results/

# Logs
aws s3api put-object --bucket ${BUCKET_NAME} --key logs/
```

### Enable S3 Event Notifications (Optional)

To automatically trigger Lambda on new uploads:

1. Go to S3 bucket > **Properties** > **Event notifications**
2. Click **Create event notification**
3. Name: `lambda-trigger-analytics`
4. Event types: Select **All object create events**
5. Prefix: `raw-data/`
6. Destination: **Lambda function** (select after creating Lambda)

---

## Step 2: Create DynamoDB Table

The DynamoDB table stores student performance metrics for fast queries.

### Using AWS Management Console

1. Go to **DynamoDB** > **Tables** > **Create Table**
2. Table name: `StudentAnalytics`
3. Partition Key: `student_id` (String)
4. Sort Key: `course_id` (String)
5. Billing Mode: **On-demand** (auto-scales, easier for learning)
6. Click **Create Table**
7. Wait for table to become ACTIVE (1-2 minutes)

### Using AWS CLI

```bash
aws dynamodb create-table \
    --table-name StudentAnalytics \
    --attribute-definitions \
        AttributeName=student_id,AttributeType=S \
        AttributeName=course_id,AttributeType=S \
    --key-schema \
        AttributeName=student_id,KeyType=HASH \
        AttributeName=course_id,KeyType=RANGE \
    --billing-mode PAY_PER_REQUEST \
    --region us-east-1

# Verify table creation
aws dynamodb list-tables --region us-east-1
```

### Add Global Secondary Index (Optional)

For querying by risk level:

```bash
aws dynamodb update-table \
    --table-name StudentAnalytics \
    --attribute-definitions \
        AttributeName=risk_level,AttributeType=S \
        AttributeName=avg_grade,AttributeType=N \
    --global-secondary-index-updates \
        "[{\"Create\":{\"IndexName\":\"risk-level-index\",\"KeySchema\":[{\"AttributeName\":\"risk_level\",\"KeyType\":\"HASH\"},{\"AttributeName\":\"avg_grade\",\"KeyType\":\"RANGE\"}],\"Projection\":{\"ProjectionType\":\"ALL\"},\"ProvisionedThroughput\":{\"ReadCapacityUnits\":5,\"WriteCapacityUnits\":5}}}]"
```

### Verify Table

```bash
aws dynamodb describe-table --table-name StudentAnalytics --region us-east-1
```

Check that status is `ACTIVE`.

---

## Step 3: Create IAM Role for Lambda

Lambda needs permissions to access S3, DynamoDB, and CloudWatch.

### Using AWS Management Console

1. Go to **IAM** > **Roles** > **Create Role**
2. Trusted entity type: **AWS service**
3. Use case: **Lambda**
4. Click **Next**
5. Add permissions:
   - Search for and select: `AmazonS3FullAccess`
   - Search for and select: `AmazonDynamoDBFullAccess`
   - Search for and select: `CloudWatchLogsFullAccess`
   - Search for and select: `AmazonAthenaFullAccess`
   - Click **Next**
6. Role name: `lambda-learning-analytics`
7. Click **Create Role**

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
    --role-name lambda-learning-analytics \
    --assume-role-policy-document file://trust-policy.json

# Attach policies
aws iam attach-role-policy \
    --role-name lambda-learning-analytics \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam attach-role-policy \
    --role-name lambda-learning-analytics \
    --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess

aws iam attach-role-policy \
    --role-name lambda-learning-analytics \
    --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess

aws iam attach-role-policy \
    --role-name lambda-learning-analytics \
    --policy-arn arn:aws:iam::aws:policy/AmazonAthenaFullAccess

# Get role ARN (you'll need this for Lambda)
aws iam get-role --role-name lambda-learning-analytics --query 'Role.Arn'
```

**Save the role ARN** - you'll need it for Lambda deployment.

Example ARN: `arn:aws:iam::123456789012:role/lambda-learning-analytics`

---

## Step 4: Create Lambda Function

Deploy the analytics processing function.

### Prepare Lambda Package

Lambda functions need dependencies packaged in a ZIP file.

```bash
# Navigate to project directory
cd projects/education/learning-analytics-platform/tier-2

# Create package directory
mkdir -p lambda-package
cd lambda-package

# Install dependencies
pip install pandas numpy scipy -t .

# Copy Lambda function code
cp ../scripts/lambda_function.py .

# Create deployment package
zip -r ../lambda-analytics.zip .

# Return to project directory
cd ..
```

### Deploy Lambda Function (Console)

1. Go to **Lambda** > **Functions** > **Create Function**
2. Choose **Author from scratch**
3. Function name: `analyze-student-performance`
4. Runtime: **Python 3.11**
5. Architecture: **x86_64**
6. Execution role: **Use an existing role**
   - Select: `lambda-learning-analytics`
7. Click **Create Function**
8. In the **Code** tab:
   - Click **Upload from** > **.zip file**
   - Upload `lambda-analytics.zip`
   - Click **Save**
9. In **Configuration** tab:
   - **General configuration**: Set timeout to **60 seconds**, memory to **512 MB**
   - **Environment variables**: Add:
     - `DYNAMODB_TABLE`: `StudentAnalytics`
     - `S3_BUCKET`: `learning-analytics-your-user-id`

### Deploy Lambda Function (CLI)

```bash
# Get the role ARN from Step 3
export ROLE_ARN="arn:aws:iam::123456789012:role/lambda-learning-analytics"
export BUCKET_NAME="learning-analytics-your-user-id"

# Create Lambda function
aws lambda create-function \
    --function-name analyze-student-performance \
    --runtime python3.11 \
    --role ${ROLE_ARN} \
    --handler lambda_function.lambda_handler \
    --zip-file fileb://lambda-analytics.zip \
    --timeout 60 \
    --memory-size 512 \
    --environment "Variables={DYNAMODB_TABLE=StudentAnalytics,S3_BUCKET=${BUCKET_NAME}}" \
    --region us-east-1
```

### Configure S3 Trigger

Add permission for S3 to invoke Lambda:

```bash
aws lambda add-permission \
    --function-name analyze-student-performance \
    --statement-id s3-trigger-permission \
    --action lambda:InvokeFunction \
    --principal s3.amazonaws.com \
    --source-arn arn:aws:s3:::${BUCKET_NAME}

# Add S3 notification configuration
cat > notification.json << EOF
{
  "LambdaFunctionConfigurations": [
    {
      "LambdaFunctionArn": "arn:aws:lambda:us-east-1:123456789012:function:analyze-student-performance",
      "Events": ["s3:ObjectCreated:*"],
      "Filter": {
        "Key": {
          "FilterRules": [
            {
              "Name": "prefix",
              "Value": "raw-data/"
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
    --bucket ${BUCKET_NAME} \
    --notification-configuration file://notification.json
```

### Test Lambda Function

Create a test event:

```bash
cat > test-event.json << 'EOF'
{
  "Records": [
    {
      "s3": {
        "bucket": {
          "name": "learning-analytics-your-user-id"
        },
        "object": {
          "key": "raw-data/sample-data.csv"
        }
      }
    }
  ]
}
EOF

# Invoke Lambda
aws lambda invoke \
    --function-name analyze-student-performance \
    --payload file://test-event.json \
    --cli-binary-format raw-in-base64-out \
    response.json

# Check response
cat response.json
```

---

## Step 5: Set Up AWS Athena

Athena enables SQL queries on S3 data without loading it into a database.

### Create Athena Workgroup (Console)

1. Go to **Athena** > **Workgroups** > **Create workgroup**
2. Workgroup name: `learning-analytics-queries`
3. Query result location: `s3://learning-analytics-your-user-id/athena-results/`
4. Click **Create workgroup**

### Create Athena Workgroup (CLI)

```bash
export BUCKET_NAME="learning-analytics-your-user-id"

aws athena create-work-group \
    --name learning-analytics-queries \
    --configuration "ResultConfigurationUpdates={OutputLocation=s3://${BUCKET_NAME}/athena-results/}" \
    --region us-east-1
```

### Create Athena Database

```bash
# Create database
aws athena start-query-execution \
    --query-string "CREATE DATABASE IF NOT EXISTS learning_analytics" \
    --result-configuration "OutputLocation=s3://${BUCKET_NAME}/athena-results/" \
    --region us-east-1
```

### Create Athena Table

Create a table pointing to processed data in S3:

```sql
-- Create external table for student analytics
CREATE EXTERNAL TABLE IF NOT EXISTS learning_analytics.student_metrics (
  student_id STRING,
  course_id STRING,
  avg_grade DOUBLE,
  median_grade DOUBLE,
  grade_trend DOUBLE,
  completion_rate DOUBLE,
  engagement_score DOUBLE,
  risk_level STRING,
  total_assessments INT,
  assignments_completed INT,
  last_updated TIMESTAMP
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION 's3://learning-analytics-your-user-id/processed-data/'
TBLPROPERTIES ('skip.header.line.count'='1');
```

Save this SQL to a file and execute:

```bash
cat > create-table.sql << 'EOF'
CREATE EXTERNAL TABLE IF NOT EXISTS learning_analytics.student_metrics (
  student_id STRING,
  course_id STRING,
  avg_grade DOUBLE,
  median_grade DOUBLE,
  grade_trend DOUBLE,
  completion_rate DOUBLE,
  engagement_score DOUBLE,
  risk_level STRING,
  total_assessments INT,
  assignments_completed INT,
  last_updated TIMESTAMP
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION 's3://${BUCKET_NAME}/processed-data/'
TBLPROPERTIES ('skip.header.line.count'='1');
EOF

# Execute query
aws athena start-query-execution \
    --query-string file://create-table.sql \
    --query-execution-context Database=learning_analytics \
    --result-configuration "OutputLocation=s3://${BUCKET_NAME}/athena-results/" \
    --region us-east-1
```

---

## Step 6: Install Python Dependencies

Install required Python packages for local scripts:

```bash
# Navigate to tier-2 directory
cd projects/education/learning-analytics-platform/tier-2

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Step 7: Configure Environment Variables

Create a `.env` file for configuration:

```bash
cat > .env << 'EOF'
# AWS Configuration
AWS_REGION=us-east-1
S3_BUCKET_NAME=learning-analytics-your-user-id
DYNAMODB_TABLE=StudentAnalytics
LAMBDA_FUNCTION_NAME=analyze-student-performance

# S3 Prefixes
S3_RAW_PREFIX=raw-data/
S3_PROCESSED_PREFIX=processed-data/
S3_ATHENA_RESULTS=athena-results/

# Athena Configuration
ATHENA_DATABASE=learning_analytics
ATHENA_TABLE=student_metrics
ATHENA_WORKGROUP=learning-analytics-queries
EOF
```

**Important:** Replace `learning-analytics-your-user-id` with your actual bucket name.

---

## Step 8: Verify Setup

Run verification checks:

```bash
# Check S3 bucket
aws s3 ls s3://${BUCKET_NAME}/

# Check DynamoDB table
aws dynamodb describe-table --table-name StudentAnalytics --region us-east-1

# Check Lambda function
aws lambda get-function --function-name analyze-student-performance --region us-east-1

# Check IAM role
aws iam get-role --role-name lambda-learning-analytics

# Check Athena workgroup
aws athena get-work-group --work-group learning-analytics-queries --region us-east-1
```

All commands should succeed without errors.

---

## Step 9: Test the Pipeline

### Upload Sample Data

```bash
# Generate sample data
python scripts/upload_to_s3.py --generate-sample --num-students 100

# This will:
# 1. Generate synthetic student data
# 2. Anonymize student IDs
# 3. Upload to S3
# 4. Trigger Lambda automatically (if S3 trigger configured)
```

### Check Lambda Execution

```bash
# View CloudWatch logs
aws logs tail /aws/lambda/analyze-student-performance --follow

# Check DynamoDB for results
aws dynamodb scan --table-name StudentAnalytics --limit 5
```

### Query with Athena

```bash
# Run a simple query
python scripts/query_results.py --query "SELECT risk_level, COUNT(*) as count FROM student_metrics GROUP BY risk_level"
```

---

## Common Setup Issues

### Issue: "Bucket already exists"
**Solution:** Bucket names must be globally unique. Add a unique suffix:
```bash
export BUCKET_NAME="learning-analytics-$(whoami)-$(date +%s)"
```

### Issue: "Access Denied" to S3 or DynamoDB
**Solution:** Verify IAM permissions:
```bash
# Check your user's policies
aws iam list-attached-user-policies --user-name your-username

# You need at least: AmazonS3FullAccess, AmazonDynamoDBFullAccess
```

### Issue: Lambda timeout
**Solution:** Increase Lambda timeout and memory:
```bash
aws lambda update-function-configuration \
    --function-name analyze-student-performance \
    --timeout 120 \
    --memory-size 1024
```

### Issue: Lambda package too large
**Solution:** Use Lambda Layers for dependencies:
```bash
# Create layer with pandas/numpy
mkdir python
pip install pandas numpy scipy -t python/
zip -r analytics-layer.zip python/

# Create Lambda layer
aws lambda publish-layer-version \
    --layer-name analytics-dependencies \
    --zip-file fileb://analytics-layer.zip \
    --compatible-runtimes python3.11
```

### Issue: Athena "HIVE_PARTITION_SCHEMA_MISMATCH"
**Solution:** Ensure CSV data matches table schema. Check:
- Column count matches
- Data types are compatible
- No extra headers in files

---

## Cost Monitoring

Set up billing alerts:

1. Go to **Billing** > **Budgets** > **Create Budget**
2. Budget type: **Cost budget**
3. Set amount: **$15** (covers this project)
4. Configure alerts at 80% and 100%
5. Enter your email for notifications

---

## Security Best Practices

### Least Privilege IAM

Replace FullAccess policies with specific permissions:

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
      "Resource": "arn:aws:s3:::learning-analytics-*/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:PutItem",
        "dynamodb:GetItem",
        "dynamodb:Query"
      ],
      "Resource": "arn:aws:dynamodb:us-east-1:*:table/StudentAnalytics"
    }
  ]
}
```

### Enable S3 Encryption

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

### Enable CloudTrail Logging

```bash
aws cloudtrail create-trail \
    --name learning-analytics-audit \
    --s3-bucket-name ${BUCKET_NAME} \
    --is-multi-region-trail
```

---

## Next Steps

After completing setup:

1. **Run the pipeline:** `python scripts/upload_to_s3.py`
2. **Query results:** `python scripts/query_results.py`
3. **Open notebook:** `jupyter notebook notebooks/learning_analysis.ipynb`
4. **Monitor costs:** Check AWS Cost Explorer daily
5. **Clean up:** Follow `cleanup_guide.md` when done

---

## Setup Summary

✅ S3 bucket created: `learning-analytics-{user-id}`
✅ DynamoDB table created: `StudentAnalytics`
✅ IAM role created: `lambda-learning-analytics`
✅ Lambda function deployed: `analyze-student-performance`
✅ Athena workspace configured: `learning-analytics-queries`
✅ Python environment ready

**Total time:** 45-60 minutes
**Ready to analyze learning data!**

For troubleshooting, see README.md or cleanup_guide.md.
