# AWS Setup Guide - Materials Property Prediction Tier 2

This guide walks you through setting up all AWS resources needed for materials property prediction. Estimated time: 30-40 minutes.

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

S3 stores your crystal structure files and reference data.

### Option A: Using AWS Console (Recommended for beginners)

1. Go to [AWS S3 Console](https://console.aws.amazon.com/s3/)
2. Click "Create bucket"
3. Enter bucket name: `materials-data-{your-name}-{date}`
   - Example: `materials-data-alice-20250114`
   - Must be globally unique
   - Must be lowercase
   - No special characters
4. Select region: **us-east-1**
5. Keep all default settings
6. Click "Create bucket"

### Option B: Using AWS CLI

```bash
# Create S3 bucket
BUCKET_NAME="materials-data-$(whoami)-$(date +%s)"
aws s3 mb "s3://$BUCKET_NAME" --region us-east-1

# Verify bucket created
aws s3 ls | grep materials-data

# Save bucket name for later
echo "BUCKET_NAME=$BUCKET_NAME" > .env
```

### Verify Bucket Created

```bash
# List your buckets
aws s3 ls

# Should see your new materials-data-xxxx bucket
```

**Save your bucket name!** You'll use it in later steps.

---

## Step 2: Create S3 Folder Structure

Lambda will expect data organized in specific folders.

### Using AWS Console

1. Open your materials-data bucket
2. Create folders:
   - Click "Create folder" → name it `structures` → Create
   - Click "Create folder" → name it `reference` → Create
   - Click "Create folder" → name it `logs` → Create

### Using AWS CLI

```bash
# Create folder structure
aws s3api put-object --bucket "$BUCKET_NAME" --key "structures/"
aws s3api put-object --bucket "$BUCKET_NAME" --key "reference/"
aws s3api put-object --bucket "$BUCKET_NAME" --key "logs/"

# Verify
aws s3 ls "s3://$BUCKET_NAME/" --recursive
```

---

## Step 3: Create DynamoDB Table

DynamoDB stores computed material properties for fast queries.

### Option A: Using AWS Console

1. Go to [DynamoDB Console](https://console.aws.amazon.com/dynamodb/)
2. Click "Create table"
3. Table name: `MaterialsProperties`
4. Partition key: `material_id` (String)
5. Table settings: Choose "On-demand" capacity mode
6. Keep all other defaults
7. Click "Create table"

Wait 1-2 minutes for table to become active.

### Option B: Using AWS CLI

```bash
# Create DynamoDB table
aws dynamodb create-table \
  --table-name MaterialsProperties \
  --attribute-definitions \
    AttributeName=material_id,AttributeType=S \
  --key-schema \
    AttributeName=material_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region us-east-1

# Wait for table to be active
aws dynamodb wait table-exists --table-name MaterialsProperties

# Verify table created
aws dynamodb describe-table --table-name MaterialsProperties
```

### Add Global Secondary Index (Optional)

For querying by density or space group:

```bash
# Add density index
aws dynamodb update-table \
  --table-name MaterialsProperties \
  --attribute-definitions \
    AttributeName=density,AttributeType=N \
  --global-secondary-index-updates \
    "[{\"Create\":{\"IndexName\":\"density-index\",\"KeySchema\":[{\"AttributeName\":\"density\",\"KeyType\":\"HASH\"}],\"Projection\":{\"ProjectionType\":\"ALL\"},\"ProvisionedThroughput\":{\"ReadCapacityUnits\":5,\"WriteCapacityUnits\":5}}}]"
```

**Note:** On-demand mode doesn't require capacity units, so this is optional.

---

## Step 4: Create IAM Role for Lambda

Lambda needs permissions to read/write S3, write to DynamoDB, and log to CloudWatch.

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
8. Add tag: Key=`Project` Value=`materials-tier-2`
9. Role name: `lambda-materials-processor`
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
  --role-name lambda-materials-processor \
  --assume-role-policy-document "$ROLE_JSON"

# Attach policies
aws iam attach-role-policy \
  --role-name lambda-materials-processor \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam attach-role-policy \
  --role-name lambda-materials-processor \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam attach-role-policy \
  --role-name lambda-materials-processor \
  --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess

# Get role ARN (you'll need this)
ROLE_ARN=$(aws iam get-role --role-name lambda-materials-processor \
  --query 'Role.Arn' --output text)
echo "ROLE_ARN=$ROLE_ARN" >> .env
```

### Verify Role Created

```bash
aws iam get-role --role-name lambda-materials-processor
```

**Save your role ARN!** Format: `arn:aws:iam::ACCOUNT_ID:role/lambda-materials-processor`

---

## Step 5: Create Lambda Function

Lambda processes crystal structures and calculates properties.

### Option A: Using AWS Console

1. Go to [Lambda Console](https://console.aws.amazon.com/lambda/)
2. Click "Create function"
3. Function name: `process-crystal-structure`
4. Runtime: **Python 3.11**
5. Architecture: **x86_64**
6. Permissions: Choose existing role
   - Select `lambda-materials-processor`
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
3. Timeout: Change to **60 seconds**
4. Memory: Set to **512 MB** (includes more CPU)
5. Click "Save"

#### Set Environment Variables:

1. In Configuration tab
2. Scroll to "Environment variables"
3. Click "Edit"
4. Add variable: `BUCKET_NAME` = `materials-data-xxxx`
5. Add variable: `DYNAMODB_TABLE` = `MaterialsProperties`
6. Add variable: `AWS_REGION` = `us-east-1`
7. Click "Save"

### Option B: Using AWS CLI

```bash
# Create deployment package
cd scripts
zip lambda_function.zip lambda_function.py
cd ..

# Wait for role to be available (usually 10-30 seconds)
sleep 15

# Create function
LAMBDA_ARN=$(aws lambda create-function \
  --function-name process-crystal-structure \
  --runtime python3.11 \
  --role "$ROLE_ARN" \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://scripts/lambda_function.zip \
  --timeout 60 \
  --memory-size 512 \
  --environment Variables="{BUCKET_NAME=$BUCKET_NAME,DYNAMODB_TABLE=MaterialsProperties,AWS_REGION=us-east-1}" \
  --query 'FunctionArn' \
  --output text)

echo "LAMBDA_ARN=$LAMBDA_ARN" >> .env

# Verify function created
aws lambda get-function --function-name process-crystal-structure
```

### Test Lambda Function

```bash
# Create test event
TEST_EVENT=$(cat <<'EOF'
{
  "Records": [
    {
      "s3": {
        "bucket": {
          "name": "materials-data-xxxx"
        },
        "object": {
          "key": "structures/Si.cif"
        }
      }
    }
  ]
}
EOF
)

# Note: Replace materials-data-xxxx with your actual bucket name

# Invoke function with test event
aws lambda invoke \
  --function-name process-crystal-structure \
  --payload "$TEST_EVENT" \
  --cli-binary-format raw-in-base64-out \
  response.json

# View response
cat response.json
```

---

## Step 6: Upload Sample Data

Upload crystal structures to test the pipeline.

### Option A: Using Python Script (Recommended)

```bash
# Configure bucket name
export AWS_S3_BUCKET="materials-data-xxxx"

# Run upload script
python scripts/upload_to_s3.py --bucket $AWS_S3_BUCKET

# Verify upload
aws s3 ls "s3://$AWS_S3_BUCKET/structures/" --recursive
```

### Option B: Using AWS Console

1. Go to your S3 bucket
2. Open `structures/` folder
3. Click "Upload"
4. Select CIF/POSCAR files from `sample_data/`
5. Click "Upload"

### Option C: Using AWS CLI

```bash
# Upload sample structures
aws s3 cp sample_data/structures/ \
  "s3://$BUCKET_NAME/structures/" \
  --recursive

# Verify upload
aws s3 ls "s3://$BUCKET_NAME/structures/" --recursive
```

---

## Step 7: Configure S3 Event Trigger (Optional)

Auto-trigger Lambda when new structures are uploaded to S3.

### Option A: Using AWS Console

1. Open S3 bucket
2. Go to "Properties" tab
3. Scroll to "Event notifications"
4. Click "Create event notification"
5. Name: `lambda-trigger`
6. Prefix: `structures/`
7. Events: Select "All object create events"
8. Destination: Lambda function
9. Function: `process-crystal-structure`
10. Click "Save changes"

### Option B: Using AWS CLI

```bash
# Create Lambda permission for S3
aws lambda add-permission \
  --function-name process-crystal-structure \
  --principal s3.amazonaws.com \
  --statement-id AllowS3Invoke \
  --action lambda:InvokeFunction \
  --source-arn "arn:aws:s3:::$BUCKET_NAME"

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
              "Value": "structures/"
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

## Step 8: Test the Pipeline

Test that data flows through successfully.

### Test 1: Manual Lambda Invocation

```bash
# Upload test file
aws s3 cp sample_data/structures/Si.cif \
  "s3://$BUCKET_NAME/structures/Si.cif"

# Invoke Lambda manually
aws lambda invoke \
  --function-name process-crystal-structure \
  --payload "{\"Records\":[{\"s3\":{\"bucket\":{\"name\":\"$BUCKET_NAME\"},\"object\":{\"key\":\"structures/Si.cif\"}}}]}" \
  response.json

# Check response
cat response.json

# Check DynamoDB for results
aws dynamodb get-item \
  --table-name MaterialsProperties \
  --key '{"material_id": {"S": "Si"}}'
```

### Test 2: Check CloudWatch Logs

```bash
# Find log group
LOG_GROUP="/aws/lambda/process-crystal-structure"

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
```

### Test 3: Query DynamoDB

```bash
# Scan entire table (small datasets only)
aws dynamodb scan --table-name MaterialsProperties

# Query specific material
aws dynamodb get-item \
  --table-name MaterialsProperties \
  --key '{"material_id": {"S": "Si"}}' \
  | python -m json.tool

# Count items in table
aws dynamodb describe-table \
  --table-name MaterialsProperties \
  --query 'Table.ItemCount'
```

**Expected result:** DynamoDB entry with computed properties:
```json
{
  "Item": {
    "material_id": {"S": "Si"},
    "formula": {"S": "Si2"},
    "space_group": {"S": "Fd-3m (227)"},
    "density": {"N": "2.33"},
    "volume": {"N": "40.88"},
    "lattice_a": {"N": "3.867"},
    "num_atoms": {"N": "2"},
    "crystal_system": {"S": "cubic"},
    "processed_at": {"S": "2025-01-14T10:30:00Z"}
  }
}
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

# Verify installations
python -c "import boto3; print('boto3:', boto3.__version__)"
python -c "import pymatgen; print('pymatgen:', pymatgen.__version__)"
python -c "import pandas; print('pandas:', pandas.__version__)"
```

### Configure .env File

```bash
# Create .env file
cat > .env << EOF
AWS_REGION=us-east-1
AWS_S3_BUCKET=materials-data-xxxx
AWS_LAMBDA_FUNCTION=process-crystal-structure
AWS_DYNAMODB_TABLE=MaterialsProperties
AWS_PROFILE=default
EOF

# Load environment variables
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
# Upload all structures to S3
python scripts/upload_to_s3.py --bucket $BUCKET_NAME

# Monitor upload
watch -n 2 "aws s3 ls s3://$BUCKET_NAME/structures/ --recursive | wc -l"
```

### Script 2: Process Structures

```bash
# Structures process automatically via S3 trigger
# Or manually trigger for specific files:
for file in $(aws s3 ls s3://$BUCKET_NAME/structures/ | awk '{print $NF}'); do
  aws lambda invoke \
    --function-name process-crystal-structure \
    --payload "{\"Records\":[{\"s3\":{\"bucket\":{\"name\":\"$BUCKET_NAME\"},\"object\":{\"key\":\"structures/$file\"}}}]}" \
    response_$file.json
done

# Monitor Lambda logs
aws logs tail /aws/lambda/process-crystal-structure --follow
```

### Script 3: Query Results

```bash
# Query materials by density range
python scripts/query_results.py \
  --property density \
  --min 2.0 \
  --max 5.0

# Query by space group
python scripts/query_results.py \
  --property space_group \
  --value "Fd-3m"

# Export all results to CSV
python scripts/query_results.py --export results.csv
```

### Script 4: Jupyter Analysis

```bash
# Launch Jupyter notebook
jupyter notebook notebooks/materials_analysis.ipynb

# Run all cells for full analysis
```

---

## Step 11: Monitor Costs

Track spending to avoid surprises.

### Set Billing Alerts

1. Go to [AWS Billing Console](https://console.aws.amazon.com/billing/)
2. Click "Billing Preferences"
3. Enable "Receive Billing Alerts"
4. Create budget:
   - Name: `Materials Tier 2 Budget`
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

1. **Delete unused structures**: Remove from S3 after processing
2. **Use DynamoDB on-demand**: Pay only for what you use
3. **Set Lambda timeout to 1 min**: Prevents runaway costs
4. **Clean up Lambda versions**: Only keep latest
5. **Enable S3 Intelligent-Tiering**: Automatic cost optimization

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

**Cause:** Complex structures or timeout too short

**Solution:**
```bash
# Increase timeout to 2 minutes
aws lambda update-function-configuration \
  --function-name process-crystal-structure \
  --timeout 120

# Or increase memory allocation (includes more CPU)
aws lambda update-function-configuration \
  --function-name process-crystal-structure \
  --memory-size 1024
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

### Problem: S3 trigger not working

**Cause:** Lambda permission not set or S3 notification not configured

**Solution:**
```bash
# 1. Add Lambda permission
aws lambda add-permission \
  --function-name process-crystal-structure \
  --principal s3.amazonaws.com \
  --statement-id AllowS3Invoke \
  --action lambda:InvokeFunction \
  --source-arn arn:aws:s3:::$BUCKET_NAME

# 2. Verify S3 notification
aws s3api get-bucket-notification-configuration \
  --bucket $BUCKET_NAME
```

### Problem: CIF parsing errors in Lambda

**Cause:** Invalid CIF format or pymatgen not available

**Solution:**
- Lambda function is standalone and doesn't include pymatgen (would require layers)
- Uses basic CIF parsing built into the function
- For complex structures, validate CIF files locally before upload

```python
# Validate locally
from pymatgen.core import Structure
try:
    struct = Structure.from_file("structure.cif")
    print("Valid CIF")
except Exception as e:
    print(f"Invalid CIF: {e}")
```

---

## Security Best Practices

### Least Privilege Access

Create a custom IAM policy with minimal permissions:

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
      "Resource": "arn:aws:s3:::materials-data-*/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:PutItem",
        "dynamodb:GetItem",
        "dynamodb:Query",
        "dynamodb:Scan"
      ],
      "Resource": "arn:aws:dynamodb:*:*:table/MaterialsProperties"
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

1. ✅ Review `notebooks/materials_analysis.ipynb` for analysis workflows
2. ✅ Read `cleanup_guide.md` for resource deletion
3. ✅ Explore extending the project (more properties, ML models)
4. ✅ Check AWS Cost Explorer regularly
5. ✅ Move to Tier 3 for production infrastructure

---

## Quick Reference

### Key Commands

```bash
# List S3 buckets
aws s3 ls

# Upload file to S3
aws s3 cp structure.cif s3://bucket-name/structures/

# Download from S3
aws s3 cp s3://bucket-name/structures/file.cif ./

# Monitor Lambda logs
aws logs tail /aws/lambda/process-crystal-structure --follow

# Query DynamoDB
aws dynamodb scan --table-name MaterialsProperties

# Get specific item
aws dynamodb get-item \
  --table-name MaterialsProperties \
  --key '{"material_id": {"S": "Si"}}'

# Count DynamoDB items
aws dynamodb describe-table \
  --table-name MaterialsProperties \
  --query 'Table.ItemCount'
```

---

## Support

- **Documentation:** See README.md
- **Issues:** https://github.com/research-jumpstart/research-jumpstart/issues
- **AWS Support:** https://console.aws.amazon.com/support/

---

**Next:** Follow the main workflow in [README.md](README.md) or clean up with [cleanup_guide.md](cleanup_guide.md)
