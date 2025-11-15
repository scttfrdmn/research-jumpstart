# AWS Setup Guide - Historical Text Analysis Tier 2

This guide walks you through setting up all AWS resources needed for the digital humanities text analysis project. Estimated time: 35-45 minutes.

---

## Prerequisites

Before starting, ensure you have:
- AWS account with billing enabled
- AWS CLI installed (`aws --version`)
- AWS credentials configured (`aws configure`)
- Python 3.8+ installed
- $20 budget available for testing

---

## Step 1: Create S3 Bucket

S3 stores your text corpus, processed results, and Lambda function logs.

### Option A: Using AWS Console (Recommended for beginners)

1. Go to [AWS S3 Console](https://console.aws.amazon.com/s3/)
2. Click "Create bucket"
3. Enter bucket name: `text-corpus-{your-name}-{date}`
   - Example: `text-corpus-alice-20250114`
   - Must be globally unique
   - Must be lowercase
   - No special characters except hyphens
4. Select region: **us-east-1** (recommended for compatibility)
5. **Block Public Access settings**: Keep all boxes checked (recommended)
6. **Bucket Versioning**: Disabled (to reduce costs)
7. **Tags** (optional): Add `Environment=tier-2`, `Project=text-analysis`
8. Keep all other default settings
9. Click "Create bucket"

### Option B: Using AWS CLI

```bash
# Create S3 bucket
BUCKET_NAME="text-corpus-$(whoami)-$(date +%s)"
aws s3 mb "s3://$BUCKET_NAME" --region us-east-1

# Verify bucket created
aws s3 ls | grep text-corpus

# Save bucket name for later
echo "BUCKET_NAME=$BUCKET_NAME" > .env
echo "Bucket created: $BUCKET_NAME"
```

### Verify Bucket Created

```bash
# List your buckets
aws s3 ls

# Should see your new text-corpus-xxxx bucket
```

**Important:** Save your bucket name! You'll use it throughout this guide.

---

## Step 2: Create S3 Folder Structure

Organize your corpus with a clear folder structure.

### Using AWS Console

1. Open your `text-corpus-xxxx` bucket
2. Create folders by clicking "Create folder":
   - `raw/` - Original text files organized by author
   - `processed/` - JSON results from Lambda processing
   - `metadata/` - Document metadata and corpus information
   - `logs/` - Lambda execution logs

3. Inside `raw/`, create author folders:
   - `raw/austen/`
   - `raw/dickens/`
   - `raw/bronte/`
   - etc.

### Using AWS CLI

```bash
# Load bucket name
source .env

# Create folder structure
aws s3api put-object --bucket "$BUCKET_NAME" --key "raw/"
aws s3api put-object --bucket "$BUCKET_NAME" --key "processed/"
aws s3api put-object --bucket "$BUCKET_NAME" --key "metadata/"
aws s3api put-object --bucket "$BUCKET_NAME" --key "logs/"

# Create sample author folders
aws s3api put-object --bucket "$BUCKET_NAME" --key "raw/austen/"
aws s3api put-object --bucket "$BUCKET_NAME" --key "raw/dickens/"
aws s3api put-object --bucket "$BUCKET_NAME" --key "raw/bronte/"

# Verify structure
aws s3 ls "s3://$BUCKET_NAME/" --recursive
```

---

## Step 3: Create DynamoDB Table

DynamoDB stores linguistic features and metadata for fast querying.

### Option A: Using AWS Console

1. Go to [DynamoDB Console](https://console.aws.amazon.com/dynamodb/)
2. Click "Create table"
3. **Table name**: `TextAnalysis`
4. **Partition key**: `document_id` (String)
5. **Sort key**: `timestamp` (Number)
6. **Table settings**: Use default settings
7. **Read/write capacity**: On-demand (pay per request)
8. **Encryption**: Default encryption
9. Click "Create table"
10. Wait 1-2 minutes for table creation

### Option B: Using AWS CLI

```bash
# Create DynamoDB table
aws dynamodb create-table \
  --table-name TextAnalysis \
  --attribute-definitions \
    AttributeName=document_id,AttributeType=S \
    AttributeName=timestamp,AttributeType=N \
  --key-schema \
    AttributeName=document_id,KeyType=HASH \
    AttributeName=timestamp,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST \
  --region us-east-1

# Wait for table to be active
aws dynamodb wait table-exists --table-name TextAnalysis

# Verify table created
aws dynamodb describe-table --table-name TextAnalysis \
  --query 'Table.TableStatus'
```

### Verify Table Created

```bash
# Check table status
aws dynamodb list-tables

# Should see TextAnalysis in the list
```

**Table Schema:**
- **Primary Key**: document_id (unique identifier for each text)
- **Sort Key**: timestamp (processing time, allows reprocessing tracking)
- **Attributes** (added by Lambda):
  - author, title, period, genre
  - word_count, unique_words, vocabulary_richness
  - top_words, named_entities, topics
  - avg_sentence_length, readability_score, sentiment_score

---

## Step 4: Create IAM Role for Lambda

Lambda needs permissions to read/write S3, access DynamoDB, and write logs.

### Option A: Using AWS Console

1. Go to [IAM Console](https://console.aws.amazon.com/iam/home)
2. Click "Roles" in left menu
3. Click "Create role"
4. **Select trusted entity**:
   - Choose "AWS service"
   - Use case: "Lambda"
   - Click "Next"
5. **Add permissions** (search and select):
   - `AWSLambdaBasicExecutionRole` (for CloudWatch logs)
   - `AmazonS3FullAccess` (for S3 read/write)
   - `AmazonDynamoDBFullAccess` (for DynamoDB read/write)
6. Click "Next"
7. **Name, review, and create**:
   - Role name: `lambda-text-processor`
   - Description: "Lambda role for NLP text processing"
   - Add tags: `Environment=tier-2`, `Project=text-analysis`
8. Click "Create role"

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
  --role-name lambda-text-processor \
  --assume-role-policy-document "$TRUST_POLICY"

# Attach required policies
aws iam attach-role-policy \
  --role-name lambda-text-processor \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam attach-role-policy \
  --role-name lambda-text-processor \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam attach-role-policy \
  --role-name lambda-text-processor \
  --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess

# Get role ARN (save this for Lambda creation)
ROLE_ARN=$(aws iam get-role --role-name lambda-text-processor \
  --query 'Role.Arn' --output text)
echo "ROLE_ARN=$ROLE_ARN" >> .env
echo "Role ARN: $ROLE_ARN"
```

### Verify Role Created

```bash
# Check role exists
aws iam get-role --role-name lambda-text-processor

# List attached policies
aws iam list-attached-role-policies --role-name lambda-text-processor
```

**Security Note:** For production use, replace `FullAccess` policies with least-privilege policies that restrict access to specific resources.

---

## Step 5: Prepare Lambda Deployment Package

Lambda needs NLP libraries (NLTK, spaCy) packaged as a deployment zip.

### Option A: Create Deployment Package Locally

```bash
# Create deployment directory
mkdir lambda-deployment
cd lambda-deployment

# Install dependencies to local directory
pip install \
  nltk==3.8.1 \
  boto3==1.34.0 \
  --target ./package

# Download NLTK data
python3 << EOF
import nltk
nltk.download('punkt', download_dir='./package/nltk_data')
nltk.download('stopwords', download_dir='./package/nltk_data')
nltk.download('averaged_perceptron_tagger', download_dir='./package/nltk_data')
nltk.download('maxent_ne_chunker', download_dir='./package/nltk_data')
nltk.download('words', download_dir='./package/nltk_data')
EOF

# Copy Lambda function code
cp ../scripts/lambda_function.py ./package/

# Create deployment zip
cd package
zip -r ../lambda-deployment.zip .
cd ..

echo "Deployment package created: lambda-deployment.zip"
ls -lh lambda-deployment.zip
```

**Note:** The deployment package will be ~50-100MB. Lambda supports packages up to 250MB unzipped.

### Option B: Use Lambda Layers (Advanced)

For reusable NLP libraries across multiple functions:

```bash
# Create layer directory
mkdir -p lambda-layer/python

# Install to layer
pip install nltk boto3 --target lambda-layer/python/

# Create layer zip
cd lambda-layer
zip -r ../nltk-layer.zip .
cd ..

# Create Lambda layer
aws lambda publish-layer-version \
  --layer-name nltk-spacy-layer \
  --zip-file fileb://nltk-layer.zip \
  --compatible-runtimes python3.9 python3.10
```

---

## Step 6: Create Lambda Function

Deploy the NLP processing function to Lambda.

### Option A: Using AWS Console

1. Go to [Lambda Console](https://console.aws.amazon.com/lambda/)
2. Click "Create function"
3. **Function name**: `process-text-document`
4. **Runtime**: Python 3.9 or Python 3.10
5. **Architecture**: x86_64
6. **Permissions**:
   - Choose "Use an existing role"
   - Select: `lambda-text-processor`
7. Click "Create function"

8. **Upload deployment package**:
   - In "Code source" section
   - Click "Upload from" → ".zip file"
   - Upload `lambda-deployment.zip` created in Step 5
   - Click "Save"

9. **Configure function**:
   - Click "Configuration" tab
   - General configuration:
     - Memory: **512 MB** (for NLP processing)
     - Timeout: **5 minutes** (300 seconds)
   - Environment variables:
     - `BUCKET_NAME`: Your S3 bucket name
     - `DYNAMODB_TABLE`: `TextAnalysis`
     - `NLTK_DATA`: `/opt/nltk_data` or `/var/task/nltk_data`
   - Click "Save"

10. **Test function**:
    - Click "Test" tab
    - Create test event with sample S3 event JSON (see below)
    - Click "Test"

### Option B: Using AWS CLI

```bash
# Load environment variables
source .env

# Create Lambda function
aws lambda create-function \
  --function-name process-text-document \
  --runtime python3.9 \
  --role "$ROLE_ARN" \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://lambda-deployment.zip \
  --timeout 300 \
  --memory-size 512 \
  --environment Variables="{
    BUCKET_NAME=$BUCKET_NAME,
    DYNAMODB_TABLE=TextAnalysis,
    NLTK_DATA=/var/task/nltk_data
  }" \
  --region us-east-1

# Wait for function to be active
sleep 10

# Verify function created
aws lambda get-function --function-name process-text-document
```

### Test Lambda Function

Create test event (`test-event.json`):

```json
{
  "Records": [
    {
      "s3": {
        "bucket": {
          "name": "text-corpus-your-id"
        },
        "object": {
          "key": "raw/austen/pride-and-prejudice.txt"
        }
      }
    }
  ]
}
```

Test with CLI:

```bash
# Invoke Lambda with test event
aws lambda invoke \
  --function-name process-text-document \
  --payload file://test-event.json \
  --output-file response.json

# Check response
cat response.json
```

---

## Step 7: Configure S3 Trigger (Optional)

Automatically process texts when uploaded to S3.

### Using AWS Console

1. In Lambda console, select `process-text-document` function
2. Click "Add trigger"
3. Select "S3"
4. **Bucket**: Your `text-corpus-xxxx` bucket
5. **Event type**: All object create events
6. **Prefix**: `raw/` (only trigger on raw texts)
7. **Suffix**: `.txt` (only process text files)
8. Acknowledge the checkbox
9. Click "Add"

### Using AWS CLI

```bash
# Add Lambda permission for S3 invocation
aws lambda add-permission \
  --function-name process-text-document \
  --statement-id s3-trigger-permission \
  --action lambda:InvokeFunction \
  --principal s3.amazonaws.com \
  --source-arn "arn:aws:s3:::$BUCKET_NAME"

# Configure S3 notification
NOTIFICATION_CONFIG=$(cat <<EOF
{
  "LambdaFunctionConfigurations": [
    {
      "LambdaFunctionArn": "$(aws lambda get-function --function-name process-text-document --query 'Configuration.FunctionArn' --output text)",
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
)

aws s3api put-bucket-notification-configuration \
  --bucket "$BUCKET_NAME" \
  --notification-configuration "$NOTIFICATION_CONFIG"

# Verify notification configured
aws s3api get-bucket-notification-configuration --bucket "$BUCKET_NAME"
```

**Note:** With S3 trigger enabled, Lambda will automatically process any `.txt` file uploaded to `raw/` folder.

---

## Step 8: Create Athena Workspace (Optional)

For SQL-based querying of text corpus results.

### Using AWS Console

1. Go to [Athena Console](https://console.aws.amazon.com/athena/)
2. **First-time setup**:
   - Click "Explore the query editor"
   - Set up query result location:
     - Click "Settings" tab
     - Query result location: `s3://text-corpus-your-id/athena-results/`
     - Click "Save"

3. **Create database**:
   ```sql
   CREATE DATABASE text_corpus_db;
   ```

4. **Create external table for DynamoDB**:
   ```sql
   CREATE EXTERNAL TABLE text_corpus_db.documents (
       document_id string,
       timestamp bigint,
       author string,
       title string,
       period string,
       genre string,
       word_count int,
       unique_words int,
       vocabulary_richness double,
       avg_sentence_length double,
       readability_score double,
       sentiment_score double
   )
   STORED BY 'org.apache.hadoop.hive.dynamodb.DynamoDBStorageHandler'
   TBLPROPERTIES (
       "dynamodb.table.name" = "TextAnalysis",
       "dynamodb.column.mapping" = "document_id:document_id,timestamp:timestamp,author:author,title:title,period:period,genre:genre,word_count:word_count,unique_words:unique_words,vocabulary_richness:vocabulary_richness,avg_sentence_length:avg_sentence_length,readability_score:readability_score,sentiment_score:sentiment_score"
   );
   ```

5. **Test query**:
   ```sql
   SELECT author, AVG(vocabulary_richness) as avg_richness
   FROM text_corpus_db.documents
   GROUP BY author
   ORDER BY avg_richness DESC;
   ```

### Using AWS CLI

```bash
# Create Athena query result location
aws s3api put-object --bucket "$BUCKET_NAME" --key "athena-results/"

# Create database (via Athena query execution)
QUERY_ID=$(aws athena start-query-execution \
  --query-string "CREATE DATABASE IF NOT EXISTS text_corpus_db" \
  --result-configuration "OutputLocation=s3://$BUCKET_NAME/athena-results/" \
  --query 'QueryExecutionId' --output text)

echo "Database creation query ID: $QUERY_ID"

# Wait for query completion
aws athena get-query-execution --query-execution-id "$QUERY_ID"
```

**Note:** Athena queries cost $5 per TB scanned. For small text corpora (<1GB), costs are minimal (~$0.50 total).

---

## Step 9: Verify Setup

Test all components to ensure proper configuration.

### 1. Test S3 Upload

```bash
# Create test file
echo "This is a test document for digital humanities analysis." > test.txt

# Upload to S3
aws s3 cp test.txt "s3://$BUCKET_NAME/raw/test/test.txt"

# Verify upload
aws s3 ls "s3://$BUCKET_NAME/raw/test/"
```

### 2. Test Lambda Processing

```bash
# Manually invoke Lambda
aws lambda invoke \
  --function-name process-text-document \
  --payload '{"Records":[{"s3":{"bucket":{"name":"'$BUCKET_NAME'"},"object":{"key":"raw/test/test.txt"}}}]}' \
  --output-file lambda-response.json

# Check response
cat lambda-response.json

# Should see: {"statusCode": 200, "body": "Processing completed"}
```

### 3. Test DynamoDB Query

```bash
# Query DynamoDB for test document
aws dynamodb scan \
  --table-name TextAnalysis \
  --filter-expression "contains(document_id, :test)" \
  --expression-attribute-values '{":test":{"S":"test"}}' \
  --limit 5

# Should see the test document with linguistic features
```

### 4. Test CloudWatch Logs

```bash
# View Lambda logs
aws logs tail /aws/lambda/process-text-document --follow

# Should see processing logs from test invocation
```

### 5. Clean Up Test Files

```bash
# Remove test files
aws s3 rm "s3://$BUCKET_NAME/raw/test/test.txt"
aws dynamodb delete-item \
  --table-name TextAnalysis \
  --key '{"document_id":{"S":"test-test.txt"},"timestamp":{"N":"'$(date +%s)'"}}'
```

---

## Step 10: Configure AWS Credentials for Python Scripts

Ensure Python scripts can access AWS services.

### Option A: Use AWS CLI Credentials

Already configured if you ran `aws configure`. Python boto3 will use these automatically.

Verify:
```bash
aws sts get-caller-identity
# Should show your AWS account ID
```

### Option B: Use Environment Variables

```bash
# Set in terminal session
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1

# Or add to .env file
echo "AWS_ACCESS_KEY_ID=your_access_key" >> .env
echo "AWS_SECRET_ACCESS_KEY=your_secret_key" >> .env
echo "AWS_DEFAULT_REGION=us-east-1" >> .env
```

### Option C: Use IAM Instance Profile (if on EC2)

If running from EC2 instance, attach IAM role with required permissions.

---

## Configuration Summary

After completing all steps, you should have:

| Resource | Name/ID | Status |
|----------|---------|--------|
| **S3 Bucket** | text-corpus-{your-id} | ✅ Created with folder structure |
| **DynamoDB Table** | TextAnalysis | ✅ Created with schema |
| **IAM Role** | lambda-text-processor | ✅ Created with policies |
| **Lambda Function** | process-text-document | ✅ Deployed with NLP libraries |
| **S3 Trigger** | (Optional) | ✅ Configured for auto-processing |
| **Athena Workspace** | text_corpus_db | ✅ (Optional) Created for SQL queries |

---

## Environment Variables Reference

Save these for use in Python scripts:

```bash
# .env file contents
BUCKET_NAME=text-corpus-your-id
DYNAMODB_TABLE=TextAnalysis
LAMBDA_FUNCTION=process-text-document
ROLE_ARN=arn:aws:iam::123456789:role/lambda-text-processor
AWS_REGION=us-east-1
```

Load in Python:
```python
import os
from dotenv import load_dotenv

load_dotenv()

bucket_name = os.getenv('BUCKET_NAME')
table_name = os.getenv('DYNAMODB_TABLE')
```

---

## Cost Estimate After Setup

**One-time setup costs:** ~$0.00 (all setup operations are free)

**Ongoing costs** (if you leave resources running):
- **S3 storage**: $0.023/GB/month
  - 1GB corpus = $0.023/month
  - 10GB corpus = $0.23/month
- **DynamoDB storage**: $0.25/GB/month (first 25GB free with free tier)
  - 100 documents metadata = ~100KB = $0.00/month
- **Lambda**: No cost when not invoked
- **Athena**: No cost when not querying

**Typical analysis run costs:** $7-12 (see README.md for detailed breakdown)

---

## Next Steps

Now that your AWS environment is configured:

1. **Test the setup**: Run verification tests above
2. **Upload corpus**: Use `scripts/upload_to_s3.py` to upload texts
3. **Process texts**: Lambda will process automatically (if trigger enabled)
4. **Analyze results**: Use `notebooks/text_analysis.ipynb` for analysis
5. **Query data**: Use `scripts/query_results.py` or Athena SQL

---

## Troubleshooting

### Issue: "Access Denied" when creating S3 bucket

**Solution:** Ensure your AWS user has S3 permissions:
```bash
# Check your IAM policies
aws iam list-attached-user-policies --user-name your-username

# Should include: AmazonS3FullAccess or custom S3 policy
```

### Issue: "Role cannot be assumed by Lambda"

**Solution:** Wait 10-30 seconds after role creation for IAM propagation:
```bash
# Verify role exists
aws iam get-role --role-name lambda-text-processor

# Wait and retry Lambda creation
sleep 30
aws lambda create-function ...
```

### Issue: Lambda deployment package too large

**Solution:**
1. Remove unnecessary files from package
2. Use Lambda layers for large libraries
3. Exclude test files and documentation

```bash
# Check package size
unzip -l lambda-deployment.zip | tail -1

# Should be < 50MB for direct upload, < 250MB unzipped
```

### Issue: "NLTK data not found" in Lambda execution

**Solution:** Ensure NLTK data is included in deployment package and path is set:
```python
# In lambda_function.py
import os
os.environ['NLTK_DATA'] = '/var/task/nltk_data'
```

### Issue: DynamoDB "Table does not exist"

**Solution:** Verify table name and region:
```bash
# List tables in region
aws dynamodb list-tables --region us-east-1

# Ensure TextAnalysis is listed
```

### Issue: S3 trigger not firing Lambda

**Solution:**
1. Check Lambda permissions for S3 invocation
2. Verify notification configuration
3. Check CloudWatch logs for errors

```bash
# Verify notification
aws s3api get-bucket-notification-configuration --bucket "$BUCKET_NAME"

# Check Lambda permissions
aws lambda get-policy --function-name process-text-document
```

---

## Security Best Practices

### 1. Use Least Privilege IAM Policies

Replace `FullAccess` policies with restricted policies:

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
      "Resource": "arn:aws:s3:::text-corpus-your-id/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:PutItem",
        "dynamodb:GetItem",
        "dynamodb:Query"
      ],
      "Resource": "arn:aws:dynamodb:us-east-1:*:table/TextAnalysis"
    }
  ]
}
```

### 2. Enable S3 Encryption

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

### 3. Enable S3 Versioning (Optional)

Protect against accidental deletions:
```bash
aws s3api put-bucket-versioning \
  --bucket "$BUCKET_NAME" \
  --versioning-configuration Status=Enabled
```

### 4. Set Up Billing Alerts

1. Go to [AWS Billing Console](https://console.aws.amazon.com/billing/)
2. Click "Budgets"
3. Create budget with alerts at $10, $25, $50

---

## AWS Resource Cleanup

When you're done with analysis, follow `cleanup_guide.md` to delete all resources and stop charges.

---

## Support Resources

- **AWS Documentation**: https://docs.aws.amazon.com/
- **boto3 Reference**: https://boto3.amazonaws.com/v1/documentation/api/latest/index.html
- **Lambda Limits**: https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html
- **DynamoDB Best Practices**: https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/best-practices.html

---

**Setup Complete!** You're ready to analyze historical texts at scale with AWS.

Proceed to `README.md` for project overview and `notebooks/text_analysis.ipynb` to begin analysis.

**Last updated:** 2025-11-14
