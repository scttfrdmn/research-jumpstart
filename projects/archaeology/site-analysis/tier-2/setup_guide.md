# AWS Setup Guide - Archaeological Site Analysis Tier 2

This guide walks you through setting up all AWS resources needed for the archaeological site analysis project. Estimated time: 30-45 minutes.

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

S3 stores your artifact data, images, processed results, and Lambda function logs.

### Option A: Using AWS Console (Recommended for beginners)

1. Go to [AWS S3 Console](https://console.aws.amazon.com/s3/)
2. Click "Create bucket"
3. Enter bucket name: `archaeology-data-{your-name}-{date}`
   - Example: `archaeology-data-alice-20250114`
   - Must be globally unique
   - Must be lowercase
   - No special characters
4. Select region: **us-east-1** (recommended)
5. Keep all default settings
6. Click "Create bucket"

### Option B: Using AWS CLI

```bash
# Create S3 bucket
BUCKET_NAME="archaeology-data-$(whoami)-$(date +%s)"
aws s3 mb "s3://$BUCKET_NAME" --region us-east-1

# Verify bucket created
aws s3 ls | grep archaeology-data

# Save bucket name for later
echo "BUCKET_NAME=$BUCKET_NAME" > .env
```

### Verify Bucket Created

```bash
# List your buckets
aws s3 ls

# Should see your new archaeology-data-xxxx bucket
```

**Save your bucket name!** You'll use it in later steps.

---

## Step 2: Create S3 Folder Structure

Lambda will expect data organized in specific folders.

### Using AWS Console

1. Open your archaeology-data bucket
2. Create folders:
   - Click "Create folder" → name it `raw` → Create
   - Click "Create folder" → name it `processed` → Create
   - Click "Create folder" → name it `analysis` → Create
   - Click "Create folder" → name it `images` → Create
   - Click "Create folder" → name it `logs` → Create

### Using AWS CLI

```bash
# Create folder structure
aws s3api put-object --bucket "$BUCKET_NAME" --key "raw/"
aws s3api put-object --bucket "$BUCKET_NAME" --key "processed/"
aws s3api put-object --bucket "$BUCKET_NAME" --key "analysis/"
aws s3api put-object --bucket "$BUCKET_NAME" --key "images/"
aws s3api put-object --bucket "$BUCKET_NAME" --key "logs/"

# Verify
aws s3 ls "s3://$BUCKET_NAME/" --recursive
```

---

## Step 3: Create DynamoDB Table

DynamoDB stores artifact catalog with metadata for fast queries.

### Option A: Using AWS Console

1. Go to [DynamoDB Console](https://console.aws.amazon.com/dynamodb/)
2. Click "Create table"
3. Table name: `ArtifactCatalog`
4. Partition key: `artifact_id` (String)
5. Table settings: **On-demand** (no need to provision capacity)
6. Scroll down and click "Create table"

#### Create Global Secondary Indexes (Optional but recommended):

After table is created:
1. Click on your table "ArtifactCatalog"
2. Go to "Indexes" tab
3. Click "Create index"
4. Index name: `site-period-index`
5. Partition key: `site_id` (String)
6. Sort key: `period` (String)
7. Click "Create index"

Repeat for additional indexes:
- `type-index`: Partition key = `artifact_type`
- `period-index`: Partition key = `period`

### Option B: Using AWS CLI

```bash
# Create DynamoDB table
aws dynamodb create-table \
  --table-name ArtifactCatalog \
  --attribute-definitions \
    AttributeName=artifact_id,AttributeType=S \
    AttributeName=site_id,AttributeType=S \
    AttributeName=period,AttributeType=S \
    AttributeName=artifact_type,AttributeType=S \
  --key-schema AttributeName=artifact_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region us-east-1

# Wait for table to be created (30-60 seconds)
aws dynamodb wait table-exists --table-name ArtifactCatalog

# Create Global Secondary Index for site queries
aws dynamodb update-table \
  --table-name ArtifactCatalog \
  --attribute-definitions \
    AttributeName=site_id,AttributeType=S \
    AttributeName=period,AttributeType=S \
  --global-secondary-index-updates '[{
    "Create": {
      "IndexName": "site-period-index",
      "KeySchema": [
        {"AttributeName":"site_id","KeyType":"HASH"},
        {"AttributeName":"period","KeyType":"RANGE"}
      ],
      "Projection": {"ProjectionType":"ALL"}
    }
  }]'

# Create index for artifact type queries
aws dynamodb update-table \
  --table-name ArtifactCatalog \
  --attribute-definitions AttributeName=artifact_type,AttributeType=S \
  --global-secondary-index-updates '[{
    "Create": {
      "IndexName": "type-index",
      "KeySchema": [{"AttributeName":"artifact_type","KeyType":"HASH"}],
      "Projection": {"ProjectionType":"ALL"}
    }
  }]'

# Save table name
echo "TABLE_NAME=ArtifactCatalog" >> .env
```

### Verify Table Created

```bash
aws dynamodb describe-table --table-name ArtifactCatalog
```

---

## Step 4: Create IAM Role for Lambda

Lambda needs permissions to read/write S3, write to DynamoDB, and write logs.

### Option A: Using AWS Console

1. Go to [IAM Console](https://console.aws.amazon.com/iam/home)
2. Click "Roles" in left menu
3. Click "Create role"
4. Choose "Lambda" as trusted entity
5. Click "Next: Permissions"
6. Search for and select these policies:
   - `AWSLambdaBasicExecutionRole` (CloudWatch logs)
   - `AmazonS3FullAccess` (S3 read/write)
   - `AmazonDynamoDBFullAccess` (DynamoDB read/write)
7. Click "Next: Tags"
8. Add tag: Key=`Environment` Value=`tier-2`
9. Role name: `lambda-archaeology-processor`
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
  --role-name lambda-archaeology-processor \
  --assume-role-policy-document "$ROLE_JSON"

# Attach policies
aws iam attach-role-policy \
  --role-name lambda-archaeology-processor \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam attach-role-policy \
  --role-name lambda-archaeology-processor \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam attach-role-policy \
  --role-name lambda-archaeology-processor \
  --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess

# Get role ARN (you'll need this)
ROLE_ARN=$(aws iam get-role --role-name lambda-archaeology-processor \
  --query 'Role.Arn' --output text)
echo "ROLE_ARN=$ROLE_ARN" >> .env
```

### Verify Role Created

```bash
aws iam get-role --role-name lambda-archaeology-processor
```

**Save your role ARN!** Format: `arn:aws:iam::ACCOUNT_ID:role/lambda-archaeology-processor`

---

## Step 5: Create Lambda Function

Lambda processes artifact data and classifies artifacts.

### Option A: Using AWS Console

1. Go to [Lambda Console](https://console.aws.amazon.com/lambda/)
2. Click "Create function"
3. Function name: `classify-artifacts`
4. Runtime: **Python 3.11**
5. Architecture: **x86_64**
6. Permissions: Choose existing role
   - Select `lambda-archaeology-processor`
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
4. Memory: Set to **256 MB** (sufficient for most datasets)
5. Click "Save"

#### Set Environment Variables:

1. In Configuration tab
2. Scroll to "Environment variables"
3. Click "Edit"
4. Add variable: `BUCKET_NAME` = `archaeology-data-xxxx`
5. Add variable: `TABLE_NAME` = `ArtifactCatalog`
6. Add variable: `AWS_REGION` = `us-east-1`
7. Click "Save"

### Option B: Using AWS CLI

```bash
# Create Lambda deployment package
cd scripts
zip lambda_function.zip lambda_function.py
cd ..

# Wait for role to be available (usually 10-30 seconds)
sleep 15

# Create function
LAMBDA_ARN=$(aws lambda create-function \
  --function-name classify-artifacts \
  --runtime python3.11 \
  --role "$ROLE_ARN" \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://scripts/lambda_function.zip \
  --timeout 300 \
  --memory-size 256 \
  --environment Variables="{BUCKET_NAME=$BUCKET_NAME,TABLE_NAME=ArtifactCatalog,AWS_REGION=us-east-1}" \
  --query 'FunctionArn' \
  --output text)

echo "LAMBDA_ARN=$LAMBDA_ARN" >> .env

# Verify function created
aws lambda get-function --function-name classify-artifacts
```

### Test Lambda Function

```bash
# Create test event
TEST_EVENT=$(cat <<'EOF'
{
  "bucket": "archaeology-data-xxxx",
  "key": "raw/sample_artifacts.csv"
}
EOF
)

# Note: Replace archaeology-data-xxxx with your actual bucket name

# Invoke function with test event
aws lambda invoke \
  --function-name classify-artifacts \
  --payload "$TEST_EVENT" \
  --cli-binary-format raw-in-base64-out \
  response.json

# View response
cat response.json
```

---

## Step 6: Configure S3 Event Trigger (Optional)

Auto-trigger Lambda when new artifact files are uploaded to S3.

### Option A: Using AWS Console

1. Open S3 bucket
2. Go to "Properties" tab
3. Scroll to "Event notifications"
4. Click "Create event notification"
5. Name: `artifact-upload-trigger`
6. Prefix: `raw/`
7. Suffix: `.csv`
8. Events: Select "All object create events"
9. Destination: Lambda function
10. Function: `classify-artifacts`
11. Click "Save changes"

### Option B: Using AWS CLI

```bash
# Create Lambda permission for S3
aws lambda add-permission \
  --function-name classify-artifacts \
  --principal s3.amazonaws.com \
  --action lambda:InvokeFunction \
  --source-arn "arn:aws:s3:::$BUCKET_NAME" \
  --statement-id AllowS3Invoke

# Create S3 event notification
NOTIFICATION_JSON=$(cat <<EOF
{
  "LambdaFunctionConfigurations": [
    {
      "LambdaFunctionArn": "$LAMBDA_ARN",
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
)

aws s3api put-bucket-notification-configuration \
  --bucket "$BUCKET_NAME" \
  --notification-configuration "$NOTIFICATION_JSON"
```

**Note:** S3 triggers can take 2-3 minutes to activate.

---

## Step 7: Upload Sample Data

Upload artifact data to test the pipeline.

### Option A: Using Python Script (Recommended)

```bash
# Configure bucket name in .env file
export BUCKET_NAME="archaeology-data-xxxx"

# Run upload script (generates sample data automatically)
python scripts/upload_to_s3.py

# Verify upload
aws s3 ls "s3://$BUCKET_NAME/raw/" --recursive
```

### Option B: Using AWS Console

1. Go to your S3 bucket
2. Open `raw/` folder
3. Click "Upload"
4. Select CSV files with artifact data
5. Click "Upload"

### Option C: Using AWS CLI

```bash
# Upload sample data (if you have local CSV files)
aws s3 cp sample_data/ \
  "s3://$BUCKET_NAME/raw/" \
  --recursive

# Verify upload
aws s3 ls "s3://$BUCKET_NAME/raw/" --recursive
```

---

## Step 8: Test the Pipeline

Test that data flows through successfully.

### Test 1: Manual Lambda Invocation

```bash
# Create test CSV file
cat > test_artifacts.csv << 'EOF'
artifact_id,site_id,artifact_type,material,length,width,thickness,weight,gps_lat,gps_lon,stratigraphic_unit,period
ART001,SITE_A,pottery,ceramic,120,85,8,250,40.7128,-74.0060,Layer_3,Bronze Age
ART002,SITE_A,lithic,flint,45,32,12,35,40.7130,-74.0062,Layer_3,Bronze Age
EOF

# Upload test file
aws s3 cp test_artifacts.csv "s3://$BUCKET_NAME/raw/test_artifacts.csv"

# If S3 trigger is configured, Lambda will run automatically
# Or invoke Lambda manually:
aws lambda invoke \
  --function-name classify-artifacts \
  --payload "{\"bucket\":\"$BUCKET_NAME\",\"key\":\"raw/test_artifacts.csv\"}" \
  --cli-binary-format raw-in-base64-out \
  response.json

# Check response
cat response.json
```

### Test 2: Check CloudWatch Logs

```bash
# Find log group
LOG_GROUP="/aws/lambda/classify-artifacts"

# Get latest log streams
aws logs describe-log-streams \
  --log-group-name "$LOG_GROUP" \
  --order-by LastEventTime \
  --descending \
  --max-items 5

# View logs (replace STREAM_NAME with actual stream)
aws logs get-log-events \
  --log-group-name "$LOG_GROUP" \
  --log-stream-name "STREAM_NAME"

# Or tail logs in real-time
aws logs tail "$LOG_GROUP" --follow
```

### Test 3: Verify DynamoDB Results

```bash
# Query DynamoDB for artifacts
aws dynamodb scan \
  --table-name ArtifactCatalog \
  --max-items 5

# Query specific artifact
aws dynamodb get-item \
  --table-name ArtifactCatalog \
  --key '{"artifact_id":{"S":"ART001"}}'

# Query by type using GSI
aws dynamodb query \
  --table-name ArtifactCatalog \
  --index-name type-index \
  --key-condition-expression "artifact_type = :type" \
  --expression-attribute-values '{":type":{"S":"pottery"}}'
```

### Test 4: Verify S3 Results

```bash
# Download processed result
aws s3 cp "s3://$BUCKET_NAME/processed/" ./results/ --recursive

# View results
ls -lah results/

# View JSON summary
cat results/test_artifacts_summary.json | python -m json.tool
```

---

## Step 9: Set Up Local Environment

Configure your local machine for analysis.

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
cat > .env << EOF
AWS_REGION=us-east-1
BUCKET_NAME=archaeology-data-xxxx
TABLE_NAME=ArtifactCatalog
LAMBDA_FUNCTION=classify-artifacts
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

## Step 10: Run Full Pipeline

Now run the complete analysis workflow.

### Script 1: Upload Data

```bash
# Upload artifact data to S3
python scripts/upload_to_s3.py

# Monitor upload
watch -n 2 'aws s3 ls s3://$BUCKET_NAME/raw/ --recursive'
```

### Script 2: Process Data

```bash
# Lambda processes automatically via S3 trigger
# Or manually trigger for all files:
for file in $(aws s3 ls s3://$BUCKET_NAME/raw/ | grep '.csv' | awk '{print $NF}'); do
  aws lambda invoke \
    --function-name classify-artifacts \
    --payload "{\"bucket\":\"$BUCKET_NAME\",\"key\":\"raw/$file\"}" \
    --cli-binary-format raw-in-base64-out \
    response_$file.json
done

# Monitor Lambda logs
aws logs tail /aws/lambda/classify-artifacts --follow
```

### Script 3: Query Results

```bash
# Query and analyze results from DynamoDB
python scripts/query_results.py

# Results displayed in terminal
```

### Script 4: Jupyter Analysis

```bash
# Launch Jupyter notebook
jupyter notebook notebooks/archaeology_analysis.ipynb

# Run all cells for full analysis
```

---

## Step 11: Set Up Athena (Optional)

For SQL queries on artifact data.

### Create Athena Database

```bash
# Create Athena database
aws athena start-query-execution \
  --query-string "CREATE DATABASE IF NOT EXISTS archaeology" \
  --result-configuration "OutputLocation=s3://$BUCKET_NAME/athena-results/"

# Wait for query to complete (5-10 seconds)
sleep 10
```

### Create Athena Table

```bash
# Create external table pointing to S3 data
ATHENA_QUERY=$(cat <<'EOF'
CREATE EXTERNAL TABLE IF NOT EXISTS archaeology.artifacts (
  artifact_id STRING,
  site_id STRING,
  artifact_type STRING,
  material STRING,
  length DOUBLE,
  width DOUBLE,
  thickness DOUBLE,
  weight DOUBLE,
  gps_lat DOUBLE,
  gps_lon DOUBLE,
  stratigraphic_unit STRING,
  period STRING,
  classification_confidence DOUBLE
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES ('separatorChar' = ',', 'quoteChar' = '"')
LOCATION 's3://archaeology-data-xxxx/processed/'
TBLPROPERTIES ('skip.header.line.count'='1');
EOF
)

# Replace bucket name
ATHENA_QUERY=$(echo "$ATHENA_QUERY" | sed "s/archaeology-data-xxxx/$BUCKET_NAME/g")

# Execute query
aws athena start-query-execution \
  --query-string "$ATHENA_QUERY" \
  --result-configuration "OutputLocation=s3://$BUCKET_NAME/athena-results/"
```

### Query with Athena

```bash
# Example: Count artifacts by type
aws athena start-query-execution \
  --query-string "SELECT artifact_type, COUNT(*) as count FROM archaeology.artifacts GROUP BY artifact_type" \
  --query-execution-context Database=archaeology \
  --result-configuration "OutputLocation=s3://$BUCKET_NAME/athena-results/"
```

---

## Step 12: Monitor Costs

Track spending to avoid surprises.

### Set Billing Alerts

1. Go to [AWS Billing Console](https://console.aws.amazon.com/billing/)
2. Click "Billing Preferences"
3. Enable "Receive Billing Alerts"
4. Create budget:
   - Name: `Archaeology Tier 2 Budget`
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

1. **Delete unused data**: `aws s3 rm s3://$BUCKET_NAME/raw/ --recursive` after processing
2. **Use DynamoDB on-demand**: Only pay for actual reads/writes
3. **Set Lambda timeout to 5 min**: Prevents runaway costs
4. **Delete old CloudWatch logs**: After 7 days
5. **Use S3 Intelligent-Tiering**: Automatic cost optimization

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

**Cause:** Large datasets or timeout too short

**Solution:**
```bash
# Increase timeout to 10 minutes
aws lambda update-function-configuration \
  --function-name classify-artifacts \
  --timeout 600

# Or increase memory allocation
aws lambda update-function-configuration \
  --function-name classify-artifacts \
  --memory-size 512
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

### Problem: DynamoDB queries return empty results

**Cause:** Data not written or wrong table name

**Solution:**
```bash
# Verify table exists
aws dynamodb describe-table --table-name ArtifactCatalog

# Check if table has data
aws dynamodb scan --table-name ArtifactCatalog --max-items 5

# Check Lambda logs for errors
aws logs tail /aws/lambda/classify-artifacts --follow
```

### Problem: S3 trigger not working

**Cause:** Lambda permission not set or notification not configured

**Solution:**
```bash
# 1. Add Lambda permission
aws lambda add-permission \
  --function-name classify-artifacts \
  --principal s3.amazonaws.com \
  --action lambda:InvokeFunction \
  --source-arn arn:aws:s3:::$BUCKET_NAME \
  --statement-id AllowS3Invoke

# 2. Verify S3 notification
aws s3api get-bucket-notification-configuration \
  --bucket $BUCKET_NAME
```

---

## Security Best Practices

### Least Privilege Access

Use a custom IAM policy with minimal permissions:

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
      "Resource": "arn:aws:s3:::archaeology-data-*/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:PutItem",
        "dynamodb:GetItem",
        "dynamodb:Query",
        "dynamodb:Scan"
      ],
      "Resource": "arn:aws:dynamodb:*:*:table/ArtifactCatalog*"
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

1. ✅ Review `notebooks/archaeology_analysis.ipynb` for analysis workflows
2. ✅ Read `cleanup_guide.md` for resource deletion
3. ✅ Explore extending the project (more sites, image analysis, etc.)
4. ✅ Check AWS Cost Explorer regularly
5. ✅ Move to Tier 3 for production infrastructure

---

## Quick Reference

### Key Commands

```bash
# List your S3 buckets
aws s3 ls

# Upload file to S3
aws s3 cp artifact_data.csv s3://bucket-name/raw/

# Download from S3
aws s3 cp s3://bucket-name/processed/results.json ./

# Monitor Lambda logs
aws logs tail /aws/lambda/classify-artifacts --follow

# Query DynamoDB
aws dynamodb scan --table-name ArtifactCatalog --max-items 10

# Check Lambda metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/Lambda \
  --metric-name Duration \
  --dimensions Name=FunctionName,Value=classify-artifacts \
  --start-time 2025-01-14T00:00:00Z \
  --end-time 2025-01-15T00:00:00Z \
  --period 3600 \
  --statistics Average,Maximum

# Get bucket size
aws s3api list-objects-v2 \
  --bucket archaeology-data-xxxx \
  --query "sum(Contents[].Size)" \
  --output text | awk '{print $1/1024/1024 " MB"}'
```

---

## Support

- **Documentation:** See README.md
- **Issues:** https://github.com/research-jumpstart/research-jumpstart/issues
- **AWS Support:** https://console.aws.amazon.com/support/

---

**Next:** Follow the main workflow in [README.md](README.md) or clean up with [cleanup_guide.md](cleanup_guide.md)
