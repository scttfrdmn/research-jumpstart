# AWS Setup Guide - Detailed Step-by-Step

Complete guide to set up all AWS resources for the Astronomical Image Processing project.

**Total Setup Time: 45-60 minutes**

## Part 1: Prerequisites (5 minutes)

### 1.1 AWS Account

1. Sign up for an AWS account: https://aws.amazon.com
2. Verify email and add payment method
3. Wait for account activation (usually instant)

### 1.2 AWS CLI Installation

**macOS:**
```bash
brew install awscli
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install awscli
```

**Linux (pip):**
```bash
pip install awscli
```

**Windows:**
Download from: https://aws.amazon.com/cli/

### 1.3 Get AWS Credentials

1. Go to [AWS IAM Console](https://console.aws.amazon.com/iam/)
2. Click "Users" in the left menu
3. Click on your username
4. Click "Security credentials" tab
5. Click "Create access key"
6. **Save the Access Key ID and Secret Access Key somewhere safe**

### 1.4 Configure AWS CLI

```bash
aws configure

# When prompted:
# AWS Access Key ID [None]: paste_your_access_key
# AWS Secret Access Key [None]: paste_your_secret_key
# Default region name [None]: us-east-1
# Default output format [None]: json
```

**Verify configuration:**
```bash
aws sts get-caller-identity
# Should show your account info
```

## Part 2: Create IAM Role (10 minutes)

Lambda needs permissions to access S3 and write logs.

### Option A: AWS Console (Visual)

1. Go to [IAM Roles Console](https://console.aws.amazon.com/iam/home#/roles)
2. Click "Create role" button
3. Select "AWS Lambda" under "Common use cases"
4. Click "Next: Permissions"
5. Search and attach:
   - `AmazonS3FullAccess`
   - `CloudWatchLogsFullAccess`
6. Click "Next: Tags" → "Next: Review"
7. Enter name: `lambda-astronomy-role`
8. Click "Create role"
9. Click on the role you just created
10. Copy the **ARN** (starts with `arn:aws:iam::`)

**Save the ARN:**
```bash
# Copy the ARN from the console and save it
export LAMBDA_ROLE_ARN="arn:aws:iam::123456789012:role/lambda-astronomy-role"
```

### Option B: AWS CLI (Command Line)

```bash
# Create trust policy file
cat > /tmp/lambda-trust.json << 'EOF'
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

# Create the role
aws iam create-role \
  --role-name lambda-astronomy-role \
  --assume-role-policy-document file:///tmp/lambda-trust.json

# Attach S3 permissions
aws iam attach-role-policy \
  --role-name lambda-astronomy-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# Attach CloudWatch Logs permissions
aws iam attach-role-policy \
  --role-name lambda-astronomy-role \
  --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess

# Get the role ARN
aws iam get-role \
  --role-name lambda-astronomy-role \
  --query 'Role.Arn' \
  --output text

# Save it
export LAMBDA_ROLE_ARN=$(aws iam get-role \
  --role-name lambda-astronomy-role \
  --query 'Role.Arn' \
  --output text)

echo "LAMBDA_ROLE_ARN=${LAMBDA_ROLE_ARN}"
```

**Verification:**
```bash
# List created role
aws iam list-roles --query 'Roles[?RoleName==`lambda-astronomy-role`]'

# List attached policies
aws iam list-attached-role-policies --role-name lambda-astronomy-role
```

## Part 3: Create S3 Buckets (5 minutes)

### 3.1 Create Bucket Names

S3 bucket names must be globally unique across all AWS accounts. Include a timestamp:

```bash
# Generate unique bucket name
TIMESTAMP=$(date +%s)
BUCKET_RAW="astronomy-tier2-${TIMESTAMP}-raw"
BUCKET_CATALOG="astronomy-tier2-${TIMESTAMP}-catalog"

echo "Raw bucket: ${BUCKET_RAW}"
echo "Catalog bucket: ${BUCKET_CATALOG}"

# Save for later
echo "export BUCKET_RAW=${BUCKET_RAW}" >> ~/.astronomy_env
echo "export BUCKET_CATALOG=${BUCKET_CATALOG}" >> ~/.astronomy_env
```

### 3.2 Create Buckets

```bash
# Create raw images bucket
aws s3 mb s3://${BUCKET_RAW} --region us-east-1

# Create catalog results bucket
aws s3 mb s3://${BUCKET_CATALOG} --region us-east-1

# Verify buckets exist
aws s3 ls

# Set lifecycle policy to delete old logs after 30 days
cat > /tmp/lifecycle.json << 'EOF'
{
  "Rules": [
    {
      "Id": "DeleteOldLogs",
      "Status": "Enabled",
      "Prefix": "logs/",
      "Expiration": {
        "Days": 30
      }
    }
  ]
}
EOF

aws s3api put-bucket-lifecycle-configuration \
  --bucket ${BUCKET_CATALOG} \
  --lifecycle-configuration file:///tmp/lifecycle.json
```

### 3.3 Configure Bucket for Athena

Athena needs a location to store query results:

```bash
# Create a folder in the catalog bucket for Athena results
aws s3 ls s3://${BUCKET_CATALOG}/ || echo "Bucket ready"

# Create a folder structure
aws s3api put-object \
  --bucket ${BUCKET_CATALOG} \
  --key athena-results/
```

## Part 4: Deploy Lambda Function (15 minutes)

### 4.1 Prepare Lambda Package

The Lambda function code is in `scripts/lambda_function.py`. Package it for deployment:

```bash
cd scripts

# Create a deployment package
zip lambda_deployment.zip lambda_function.py

# Verify the package
unzip -l lambda_deployment.zip

cd ..
```

### 4.2 Deploy Lambda

```bash
# Load environment variables
source ~/.astronomy_env

# Get your AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "Account ID: ${ACCOUNT_ID}"

# Create the Lambda function
aws lambda create-function \
  --function-name astronomy-source-detection \
  --runtime python3.11 \
  --role ${LAMBDA_ROLE_ARN} \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://scripts/lambda_deployment.zip \
  --timeout 300 \
  --memory-size 1024 \
  --ephemeral-storage Size=2048 \
  --environment "Variables={BUCKET_RAW=${BUCKET_RAW},BUCKET_CATALOG=${BUCKET_CATALOG}}"

# Verify deployment
aws lambda get-function --function-name astronomy-source-detection
```

### 4.3 Update Lambda (if needed)

To update the function code later:

```bash
# Repackage
cd scripts
zip lambda_deployment.zip lambda_function.py
cd ..

# Update the function
aws lambda update-function-code \
  --function-name astronomy-source-detection \
  --zip-file fileb://scripts/lambda_deployment.zip
```

## Part 5: Set Up Athena (10 minutes)

### 5.1 Create Athena Workgroup

```bash
# Create a workgroup for queries
aws athena create-work-group \
  --name astronomy-workgroup \
  --description "Workgroup for astronomical catalog queries" \
  --configuration "ResultConfigurationUpdates={OutputLocation=s3://${BUCKET_CATALOG}/athena-results/}"
```

### 5.2 Create Athena Database and Table

```bash
# Create the database
aws athena start-query-execution \
  --query-string "CREATE DATABASE IF NOT EXISTS astronomy" \
  --work-group astronomy-workgroup

# Wait for query to complete (check manually via console or logs)

# Create the sources table
aws athena start-query-execution \
  --query-string """
CREATE EXTERNAL TABLE IF NOT EXISTS astronomy.sources (
  image_id STRING,
  source_id INT,
  ra DOUBLE,
  dec DOUBLE,
  x DOUBLE,
  y DOUBLE,
  flux DOUBLE,
  flux_err DOUBLE,
  peak DOUBLE,
  fwhm DOUBLE,
  a DOUBLE,
  b DOUBLE,
  theta DOUBLE,
  snr DOUBLE,
  detection_time TIMESTAMP
)
STORED AS PARQUET
LOCATION 's3://${BUCKET_CATALOG}/sources/'
TBLPROPERTIES ('classification'='parquet', 'compressionType'='snappy')
  """ \
  --work-group astronomy-workgroup
```

### 5.3 Verify Athena Setup

```bash
# List databases
aws athena start-query-execution \
  --query-string "SHOW DATABASES" \
  --work-group astronomy-workgroup

# Check for tables
aws athena start-query-execution \
  --query-string "SHOW TABLES IN astronomy" \
  --work-group astronomy-workgroup

# Simple test query
aws athena start-query-execution \
  --query-string "SELECT COUNT(*) as table_count FROM information_schema.tables WHERE table_schema = 'astronomy'" \
  --work-group astronomy-workgroup
```

## Part 6: Download Sample Data (15 minutes)

### 6.1 Download FITS Images

```bash
# Create data directory
mkdir -p data/raw

# Download sample FITS images from SDSS
python scripts/download_sample_fits.py

# This downloads ~500MB of data
# Should take 5-15 minutes depending on internet speed

# Verify download
ls -lah data/raw/
```

### 6.2 Upload to S3

```bash
# Load environment variables
source ~/.astronomy_env

# Upload FITS images
python scripts/upload_to_s3.py

# Verify upload
aws s3 ls s3://${BUCKET_RAW}/ --recursive

# Count files
aws s3 ls s3://${BUCKET_RAW}/ --recursive --summarize
```

## Part 7: Run Source Detection (20 minutes)

### 7.1 Invoke Lambda Function

```bash
# Run source detection on all uploaded images
python scripts/invoke_lambda.py

# This will:
# 1. List all FITS files in S3
# 2. Invoke Lambda for each file
# 3. Wait for completion
# 4. Report results

# Monitor progress
aws logs tail /aws/lambda/astronomy-source-detection --follow
```

### 7.2 Check Results

```bash
# List output catalogs
aws s3 ls s3://${BUCKET_CATALOG}/sources/ --recursive

# Download a catalog to inspect
aws s3 cp s3://${BUCKET_CATALOG}/sources/image_001_sources.parquet ./data/

# Check Athena for results
aws athena start-query-execution \
  --query-string "SELECT COUNT(*) as total_sources FROM astronomy.sources" \
  --work-group astronomy-workgroup
```

## Part 8: Interactive Analysis (10 minutes)

### 8.1 Start Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook notebooks/sky_analysis.ipynb

# This opens a browser window to the notebook
```

### 8.2 Query and Visualize

Use the notebook to:
1. Query Athena for results
2. Load and analyze catalogs
3. Create visualizations
4. Export results

## Verification Checklist

Confirm all components are set up:

```bash
# 1. IAM Role exists
aws iam list-roles --query 'Roles[?RoleName==`lambda-astronomy-role`]'

# 2. S3 Buckets exist
aws s3 ls

# 3. Lambda function exists
aws lambda list-functions --query 'Functions[?FunctionName==`astronomy-source-detection`]'

# 4. Lambda can access S3
aws lambda test-function \
  --function-name astronomy-source-detection \
  --payload '{"test": "connection"}'

# 5. Athena database exists
aws athena start-query-execution \
  --query-string "SHOW DATABASES" \
  --work-group astronomy-workgroup

# 6. CloudWatch logs exist
aws logs describe-log-groups --log-group-name-prefix /aws/lambda
```

## Environment Variables

Save these for future use:

```bash
# Create ~/.astronomy_env file
cat > ~/.astronomy_env << 'EOF'
export AWS_REGION=us-east-1
export BUCKET_RAW=astronomy-tier2-XXXXX-raw
export BUCKET_CATALOG=astronomy-tier2-XXXXX-catalog
export LAMBDA_FUNCTION=astronomy-source-detection
export LAMBDA_ROLE_ARN=arn:aws:iam::XXXXXXXXXX:role/lambda-astronomy-role
EOF

# Load variables
source ~/.astronomy_env
```

## Troubleshooting

### Issue: "Access Denied" when creating resources

**Solution:** Check IAM permissions
```bash
aws iam get-user
aws iam list-attached-user-policies --user-name <your-username>
```

Need to attach policies to your user (not just Lambda role).

### Issue: Lambda timeout

**Solution:** Increase timeout
```bash
aws lambda update-function-configuration \
  --function-name astronomy-source-detection \
  --timeout 600
```

### Issue: S3 bucket name already exists

**Solution:** Use a different name with timestamp
```bash
TIMESTAMP=$(date +%s%N)
BUCKET_NAME="astronomy-tier2-${TIMESTAMP}-raw"
aws s3 mb s3://${BUCKET_NAME}
```

### Issue: Cannot access S3 from Lambda

**Solution:** Verify role has correct permissions
```bash
aws iam get-role-policy \
  --role-name lambda-astronomy-role \
  --policy-name AmazonS3FullAccess
```

If missing, attach the policy:
```bash
aws iam attach-role-policy \
  --role-name lambda-astronomy-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

### Issue: Athena table not found

**Solution:** Create table manually
```bash
aws athena start-query-execution \
  --query-string """
CREATE EXTERNAL TABLE astronomy.sources (
  image_id STRING,
  source_id INT,
  ra DOUBLE,
  dec DOUBLE,
  flux DOUBLE
)
STORED AS PARQUET
LOCATION 's3://${BUCKET_CATALOG}/sources/'
  """ \
  --work-group astronomy-workgroup
```

## Next Steps

1. Complete all 8 setup sections above
2. Run `jupyter notebook notebooks/sky_analysis.ipynb`
3. Follow the notebook for analysis
4. See README.md for cost tracking and cleanup

**When finished, see cleanup_guide.md to delete all resources**
