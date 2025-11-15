# AWS Setup Guide - Molecular Analysis Tier 2

This guide walks you through setting up all AWS resources needed for the molecular property analysis project. Estimated time: 30-45 minutes.

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

S3 stores your molecular structures and processing logs.

### Option A: Using AWS Console (Recommended for beginners)

1. Go to [AWS S3 Console](https://console.aws.amazon.com/s3/)
2. Click "Create bucket"
3. Enter bucket name: `molecular-data-{your-name}-{date}`
   - Example: `molecular-data-alice-20250114`
   - Must be globally unique
   - Must be lowercase
   - No special characters except hyphens
4. Select region: **us-east-1** (or your preferred region)
5. Keep all default settings:
   - Block all public access: **Enabled**
   - Bucket versioning: **Disabled**
   - Server-side encryption: **Amazon S3 managed keys (SSE-S3)**
6. Click "Create bucket"

### Option B: Using AWS CLI

```bash
# Create S3 bucket with unique name
BUCKET_NAME="molecular-data-$(whoami)-$(date +%s)"
aws s3 mb "s3://$BUCKET_NAME" --region us-east-1

# Verify bucket created
aws s3 ls | grep molecular-data

# Save bucket name for later
echo "BUCKET_NAME=$BUCKET_NAME" > .env
echo "Bucket created: $BUCKET_NAME"
```

### Verify Bucket Created

```bash
# List your buckets
aws s3 ls

# Should see your new molecular-data-xxxx bucket
```

**Save your bucket name!** You'll use it throughout this setup.

---

## Step 2: Create S3 Folder Structure

Organize molecular data in logical folders.

### Using AWS Console

1. Open your molecular-data bucket
2. Create folders:
   - Click "Create folder" → name it `molecules` → Create
   - Click "Create folder" → name it `molecules/drugs` → Create
   - Click "Create folder" → name it `molecules/natural_products` → Create
   - Click "Create folder" → name it `molecules/screening_library` → Create
   - Click "Create folder" → name it `logs` → Create

### Using AWS CLI

```bash
# Create folder structure (S3 doesn't have true folders, but we can simulate them)
aws s3api put-object --bucket "$BUCKET_NAME" --key "molecules/"
aws s3api put-object --bucket "$BUCKET_NAME" --key "molecules/drugs/"
aws s3api put-object --bucket "$BUCKET_NAME" --key "molecules/natural_products/"
aws s3api put-object --bucket "$BUCKET_NAME" --key "molecules/screening_library/"
aws s3api put-object --bucket "$BUCKET_NAME" --key "logs/"

# Verify structure
aws s3 ls "s3://$BUCKET_NAME/" --recursive
```

---

## Step 3: Create DynamoDB Table

DynamoDB stores calculated molecular properties.

### Option A: Using AWS Console

1. Go to [DynamoDB Console](https://console.aws.amazon.com/dynamodb/)
2. Click "Create table"
3. Table settings:
   - **Table name**: `MolecularProperties`
   - **Partition key**: `molecule_id` (String)
   - **Sort key**: `compound_class` (String)
4. Table settings:
   - **Table class**: DynamoDB Standard
   - **Capacity mode**: On-demand (pay per request)
   - **Encryption**: Amazon DynamoDB encryption at rest (default)
5. Click "Create table"
6. Wait 1-2 minutes for table to become active

### Option B: Using AWS CLI

```bash
# Create DynamoDB table
aws dynamodb create-table \
  --table-name MolecularProperties \
  --attribute-definitions \
    AttributeName=molecule_id,AttributeType=S \
    AttributeName=compound_class,AttributeType=S \
  --key-schema \
    AttributeName=molecule_id,KeyType=HASH \
    AttributeName=compound_class,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST \
  --region us-east-1

# Wait for table to become active
aws dynamodb wait table-exists --table-name MolecularProperties

# Verify table created
aws dynamodb describe-table --table-name MolecularProperties
```

### Table Schema Explanation

```
MolecularProperties Table:
├── molecule_id (Partition Key) - Unique identifier (e.g., "CHEM000001")
├── compound_class (Sort Key) - Category (e.g., "drug", "natural_product")
├── smiles (String) - SMILES notation
├── name (String) - Molecule name
├── molecular_weight (Number) - MW in Da
├── logp (Number) - Lipophilicity
├── tpsa (Number) - Topological polar surface area
├── hbd (Number) - Hydrogen bond donors
├── hba (Number) - Hydrogen bond acceptors
├── rotatable_bonds (Number) - Flexibility measure
├── aromatic_rings (Number) - Aromaticity
├── lipinski_compliant (Boolean) - Drug-likeness
└── timestamp (String) - ISO 8601 datetime
```

---

## Step 4: Create IAM Role for Lambda

Lambda needs permissions to read/write S3, DynamoDB, and write logs.

### Option A: Using AWS Console

1. Go to [IAM Console](https://console.aws.amazon.com/iam/home)
2. Click "Roles" in left menu
3. Click "Create role"
4. **Trusted entity type**: AWS service
5. **Use case**: Lambda
6. Click "Next"
7. **Permissions policies** - Search and select:
   - `AWSLambdaBasicExecutionRole` (CloudWatch logs)
   - `AmazonS3FullAccess` (S3 read/write)
   - `AmazonDynamoDBFullAccess` (DynamoDB read/write)
8. Click "Next"
9. **Role name**: `lambda-molecular-analyzer`
10. **Description**: "Lambda role for molecular property analysis"
11. **Tags** (optional):
    - Key: `Project`, Value: `molecular-analysis-tier2`
    - Key: `Environment`, Value: `development`
12. Click "Create role"

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
  --role-name lambda-molecular-analyzer \
  --assume-role-policy-document "$TRUST_POLICY" \
  --description "Lambda role for molecular property analysis"

# Attach required policies
aws iam attach-role-policy \
  --role-name lambda-molecular-analyzer \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam attach-role-policy \
  --role-name lambda-molecular-analyzer \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam attach-role-policy \
  --role-name lambda-molecular-analyzer \
  --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess

# Get role ARN (you'll need this)
ROLE_ARN=$(aws iam get-role --role-name lambda-molecular-analyzer \
  --query 'Role.Arn' --output text)
echo "ROLE_ARN=$ROLE_ARN" >> .env
echo "Role ARN: $ROLE_ARN"
```

### Security Note: Least Privilege Alternative

For production, use a custom policy with minimal permissions:

```bash
# Create custom policy (more secure)
POLICY_JSON=$(cat <<'EOF'
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
        "arn:aws:s3:::molecular-data-*",
        "arn:aws:s3:::molecular-data-*/*"
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
      "Resource": "arn:aws:dynamodb:*:*:table/MolecularProperties"
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
)

# Create and attach custom policy
aws iam create-policy \
  --policy-name MolecularAnalysisPolicy \
  --policy-document "$POLICY_JSON"

# Get policy ARN
POLICY_ARN=$(aws iam list-policies --query 'Policies[?PolicyName==`MolecularAnalysisPolicy`].Arn' --output text)

# Attach to role
aws iam attach-role-policy \
  --role-name lambda-molecular-analyzer \
  --policy-arn "$POLICY_ARN"
```

---

## Step 5: Create Lambda Function

Lambda processes molecular structures and calculates properties.

### Option A: Using AWS Console

1. Go to [Lambda Console](https://console.aws.amazon.com/lambda/)
2. Click "Create function"
3. Function settings:
   - **Function name**: `analyze-molecule`
   - **Runtime**: Python 3.11
   - **Architecture**: x86_64
   - **Permissions**: Use existing role
     - Select `lambda-molecular-analyzer`
4. Click "Create function"

#### Upload Function Code:

1. In function page, scroll to "Code source"
2. Click "Upload from" → ".zip file"
3. Create deployment package:
   ```bash
   cd scripts
   zip lambda_deployment.zip lambda_function.py
   ```
4. Upload `lambda_deployment.zip`
5. Click "Save"

#### Configure Function Settings:

1. Go to "Configuration" tab
2. Click "General configuration" → "Edit"
   - **Timeout**: 30 seconds
   - **Memory**: 256 MB
   - **Ephemeral storage**: 512 MB
3. Click "Save"

#### Set Environment Variables:

1. In Configuration tab, click "Environment variables"
2. Click "Edit"
3. Add variables:
   - `BUCKET_NAME` = `molecular-data-xxxx` (your bucket name)
   - `DYNAMODB_TABLE` = `MolecularProperties`
   - `AWS_REGION` = `us-east-1`
4. Click "Save"

### Option B: Using AWS CLI

```bash
# Navigate to scripts directory
cd scripts

# Create Lambda deployment package
zip lambda_deployment.zip lambda_function.py

# Wait for IAM role to propagate (10-30 seconds)
echo "Waiting for IAM role to be ready..."
sleep 20

# Create Lambda function
aws lambda create-function \
  --function-name analyze-molecule \
  --runtime python3.11 \
  --role "$ROLE_ARN" \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://lambda_deployment.zip \
  --timeout 30 \
  --memory-size 256 \
  --environment Variables="{BUCKET_NAME=$BUCKET_NAME,DYNAMODB_TABLE=MolecularProperties,AWS_REGION=us-east-1}" \
  --description "Molecular property calculation function" \
  --region us-east-1

# Get function ARN
LAMBDA_ARN=$(aws lambda get-function --function-name analyze-molecule \
  --query 'Configuration.FunctionArn' --output text)
echo "LAMBDA_ARN=$LAMBDA_ARN" >> .env
echo "Lambda function created: $LAMBDA_ARN"

cd ..
```

### Test Lambda Function

```bash
# Create test event
TEST_EVENT=$(cat <<'EOF'
{
  "Records": [{
    "s3": {
      "bucket": {"name": "molecular-data-xxxx"},
      "object": {"key": "molecules/drugs/aspirin.smi"}
    }
  }]
}
EOF
)

# Replace bucket name with yours
TEST_EVENT=$(echo "$TEST_EVENT" | sed "s/molecular-data-xxxx/$BUCKET_NAME/")

# Invoke function
aws lambda invoke \
  --function-name analyze-molecule \
  --payload "$TEST_EVENT" \
  --cli-binary-format raw-in-base64-out \
  response.json

# View response
cat response.json | python -m json.tool
```

---

## Step 6: Configure S3 Event Trigger (Optional)

Auto-trigger Lambda when new molecules are uploaded to S3.

### Option A: Using AWS Console

1. Open S3 bucket `molecular-data-xxxx`
2. Go to "Properties" tab
3. Scroll to "Event notifications"
4. Click "Create event notification"
5. Event notification settings:
   - **Name**: `molecule-upload-trigger`
   - **Event types**: Select "All object create events"
   - **Prefix**: `molecules/`
   - **Suffix**: `.smi`
   - **Destination**: Lambda function
   - **Lambda function**: `analyze-molecule`
6. Click "Save changes"

### Option B: Using AWS CLI

```bash
# Add Lambda permission for S3 invocation
aws lambda add-permission \
  --function-name analyze-molecule \
  --principal s3.amazonaws.com \
  --statement-id s3-invoke-permission \
  --action lambda:InvokeFunction \
  --source-arn "arn:aws:s3:::$BUCKET_NAME" \
  --source-account $(aws sts get-caller-identity --query Account --output text)

# Create S3 event notification configuration
NOTIFICATION_JSON=$(cat <<EOF
{
  "LambdaFunctionConfigurations": [
    {
      "LambdaFunctionArn": "$LAMBDA_ARN",
      "Events": ["s3:ObjectCreated:*"],
      "Filter": {
        "Key": {
          "FilterRules": [
            {"Name": "prefix", "Value": "molecules/"},
            {"Name": "suffix", "Value": ".smi"}
          ]
        }
      }
    }
  ]
}
EOF
)

# Apply notification configuration
aws s3api put-bucket-notification-configuration \
  --bucket "$BUCKET_NAME" \
  --notification-configuration "$NOTIFICATION_JSON"

# Verify configuration
aws s3api get-bucket-notification-configuration \
  --bucket "$BUCKET_NAME"
```

**Note:** S3 event triggers can take 2-3 minutes to activate.

---

## Step 7: Upload Sample Data

Upload sample molecular structures to test the pipeline.

### Option A: Using Python Script (Recommended)

```bash
# Set environment variables
export AWS_S3_BUCKET="$BUCKET_NAME"
export AWS_REGION="us-east-1"

# Run upload script
python scripts/upload_to_s3.py \
  --bucket "$BUCKET_NAME" \
  --data-dir sample_data/

# Verify upload
aws s3 ls "s3://$BUCKET_NAME/molecules/" --recursive
```

### Option B: Using AWS Console

1. Go to your S3 bucket
2. Open `molecules/drugs/` folder
3. Click "Upload"
4. Select files from `sample_data/drugs/`
5. Click "Upload"
6. Repeat for `natural_products/` and `screening_library/`

### Option C: Using AWS CLI

```bash
# Upload all sample data
aws s3 cp sample_data/ \
  "s3://$BUCKET_NAME/molecules/" \
  --recursive \
  --exclude "*.md" \
  --exclude "README*"

# Verify upload
aws s3 ls "s3://$BUCKET_NAME/molecules/" --recursive

# Check file count
aws s3 ls "s3://$BUCKET_NAME/molecules/" --recursive | wc -l
```

### Create Sample SMILES Files (if not provided)

```bash
# Create sample drug molecules
cat > sample_data/drugs/aspirin.smi << 'EOF'
CC(=O)Oc1ccccc1C(=O)O aspirin
EOF

cat > sample_data/drugs/ibuprofen.smi << 'EOF'
CC(C)Cc1ccc(cc1)C(C)C(=O)O ibuprofen
EOF

cat > sample_data/drugs/caffeine.smi << 'EOF'
CN1C=NC2=C1C(=O)N(C(=O)N2C)C caffeine
EOF

# Upload to S3
aws s3 cp sample_data/drugs/ "s3://$BUCKET_NAME/molecules/drugs/" --recursive
```

---

## Step 8: Test the Complete Pipeline

Verify that data flows through successfully.

### Test 1: Manual Lambda Invocation

```bash
# Upload test molecule
echo "CC(=O)Oc1ccccc1C(=O)O aspirin" > /tmp/test_molecule.smi
aws s3 cp /tmp/test_molecule.smi "s3://$BUCKET_NAME/molecules/drugs/test_molecule.smi"

# Invoke Lambda manually
aws lambda invoke \
  --function-name analyze-molecule \
  --payload "{\"Records\":[{\"s3\":{\"bucket\":{\"name\":\"$BUCKET_NAME\"},\"object\":{\"key\":\"molecules/drugs/test_molecule.smi\"}}}]}" \
  --cli-binary-format raw-in-base64-out \
  response.json

# Check response
cat response.json | python -m json.tool
```

### Test 2: Check DynamoDB Data

```bash
# Scan DynamoDB table for results
aws dynamodb scan \
  --table-name MolecularProperties \
  --max-items 5

# Query specific molecule
aws dynamodb get-item \
  --table-name MolecularProperties \
  --key '{"molecule_id":{"S":"aspirin"},"compound_class":{"S":"drug"}}'
```

### Test 3: Check CloudWatch Logs

```bash
# Find log group
LOG_GROUP="/aws/lambda/analyze-molecule"

# Get latest log streams
aws logs describe-log-streams \
  --log-group-name "$LOG_GROUP" \
  --order-by LastEventTime \
  --descending \
  --max-items 1

# Get log stream name
LOG_STREAM=$(aws logs describe-log-streams \
  --log-group-name "$LOG_GROUP" \
  --order-by LastEventTime \
  --descending \
  --max-items 1 \
  --query 'logStreams[0].logStreamName' \
  --output text)

# View logs
aws logs get-log-events \
  --log-group-name "$LOG_GROUP" \
  --log-stream-name "$LOG_STREAM" \
  --limit 50
```

### Test 4: Verify Results with Python

```python
import boto3
import pandas as pd

# Initialize DynamoDB client
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table('MolecularProperties')

# Scan table
response = table.scan(Limit=10)
items = response['Items']

# Convert to DataFrame
df = pd.DataFrame(items)
print(df)

# Check molecular properties
print("\nMolecular Weight Range:", df['molecular_weight'].min(), "-", df['molecular_weight'].max())
print("Lipinski Compliant:", df['lipinski_compliant'].sum(), "out of", len(df))
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
python -c "import pandas; print('pandas:', pandas.__version__)"
python -c "import rdkit; print('rdkit installed')"  # Optional
```

### Configure Environment Variables

```bash
# Create .env file
cat > .env << EOF
AWS_REGION=us-east-1
AWS_S3_BUCKET=$BUCKET_NAME
AWS_LAMBDA_FUNCTION=analyze-molecule
AWS_DYNAMODB_TABLE=MolecularProperties
AWS_PROFILE=default
EOF

# Load environment variables (optional)
export $(cat .env | xargs)
```

### Verify AWS Access

```bash
# Check AWS credentials
aws sts get-caller-identity

# Test S3 access
aws s3 ls "s3://$BUCKET_NAME/"

# Test DynamoDB access
aws dynamodb describe-table --table-name MolecularProperties

# Test Lambda access
aws lambda get-function --function-name analyze-molecule
```

---

## Step 10: Run Analysis Notebook

Launch Jupyter and run the analysis workflow.

### Start Jupyter

```bash
# Start Jupyter notebook server
jupyter notebook notebooks/molecular_analysis.ipynb

# Or use JupyterLab
jupyter lab
```

### Notebook Workflow

The notebook will guide you through:

1. **Connect to AWS** - Load credentials and connect to services
2. **Query Molecules** - Fetch molecular properties from DynamoDB
3. **Analyze Properties** - Statistical analysis and distributions
4. **Filter Drug-Like** - Apply Lipinski's Rule of Five
5. **Visualize** - Create plots and chemical space maps
6. **Export Results** - Save hit lists and figures

---

## Step 11: Set Up Cost Monitoring

Track spending to avoid surprises.

### Set Billing Alerts

1. Go to [AWS Billing Console](https://console.aws.amazon.com/billing/)
2. Click "Billing Preferences"
3. Enable "Receive Billing Alerts"
4. Click "Save preferences"
5. Go to [CloudWatch Console](https://console.aws.amazon.com/cloudwatch/)
6. Navigate to "Alarms" → "Billing"
7. Create alarm:
   - Metric: EstimatedCharges
   - Condition: Greater than $10
   - Actions: Send email notification
8. Create additional alarms for $15 and $20

### Create Budget

```bash
# Create monthly budget with alerts
aws budgets create-budget \
  --account-id $(aws sts get-caller-identity --query Account --output text) \
  --budget '{
    "BudgetName": "MolecularAnalysisBudget",
    "BudgetLimit": {"Amount": "20", "Unit": "USD"},
    "TimeUnit": "MONTHLY",
    "BudgetType": "COST"
  }' \
  --notifications-with-subscribers '[
    {
      "Notification": {
        "NotificationType": "ACTUAL",
        "ComparisonOperator": "GREATER_THAN",
        "Threshold": 80
      },
      "Subscribers": [{
        "SubscriptionType": "EMAIL",
        "Address": "your-email@example.com"
      }]
    }
  ]'
```

### Check Current Costs

```bash
# View costs for current month
aws ce get-cost-and-usage \
  --time-period Start=$(date -d "$(date +%Y-%m-01)" +%Y-%m-%d),End=$(date +%Y-%m-%d) \
  --granularity MONTHLY \
  --metrics "UnblendedCost" \
  --group-by Type=DIMENSION,Key=SERVICE

# Or use Cost Explorer in console:
# https://console.aws.amazon.com/cost-management/
```

---

## Troubleshooting

### Problem: "Access Denied" creating S3 bucket

**Cause:** IAM permissions issue

**Solution:**
```bash
# Check your AWS user permissions
aws iam get-user

# Attach S3 full access (or create custom policy)
aws iam attach-user-policy \
  --user-name your-username \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

### Problem: Lambda timeout errors

**Cause:** Processing takes longer than timeout setting

**Solution:**
```bash
# Increase timeout to 60 seconds
aws lambda update-function-configuration \
  --function-name analyze-molecule \
  --timeout 60

# Or increase memory (also increases CPU)
aws lambda update-function-configuration \
  --function-name analyze-molecule \
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

### Problem: DynamoDB "ProvisionedThroughputExceededException"

**Cause:** Too many requests for on-demand capacity

**Solution:**
```bash
# On-demand mode should auto-scale, but check table mode
aws dynamodb describe-table --table-name MolecularProperties

# If using provisioned capacity, switch to on-demand
aws dynamodb update-table \
  --table-name MolecularProperties \
  --billing-mode PAY_PER_REQUEST
```

### Problem: S3 event trigger not working

**Cause:** Lambda permission not set or notification not configured

**Solution:**
```bash
# Check Lambda permissions
aws lambda get-policy --function-name analyze-molecule

# Re-add S3 permission
aws lambda add-permission \
  --function-name analyze-molecule \
  --principal s3.amazonaws.com \
  --statement-id s3-invoke-permission \
  --action lambda:InvokeFunction \
  --source-arn "arn:aws:s3:::$BUCKET_NAME"

# Verify S3 notification
aws s3api get-bucket-notification-configuration \
  --bucket "$BUCKET_NAME"
```

---

## Security Best Practices

### Enable S3 Bucket Encryption

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

### Block Public S3 Access

```bash
# Block all public access (should be default)
aws s3api put-public-access-block \
  --bucket "$BUCKET_NAME" \
  --public-access-block-configuration \
  "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
```

### Enable DynamoDB Point-in-Time Recovery

```bash
# Enable continuous backups
aws dynamodb update-continuous-backups \
  --table-name MolecularProperties \
  --point-in-time-recovery-specification PointInTimeRecoveryEnabled=true
```

---

## Next Steps

After successful setup:

1. Review `notebooks/molecular_analysis.ipynb` for analysis workflows
2. Read `cleanup_guide.md` for resource deletion
3. Explore extending the project (more properties, larger datasets)
4. Check AWS Cost Explorer regularly
5. Move to Tier 3 for production infrastructure

---

## Quick Reference

### Key AWS Resources

```bash
# S3 Bucket
BUCKET_NAME="molecular-data-xxxx"
aws s3 ls "s3://$BUCKET_NAME/"

# DynamoDB Table
TABLE_NAME="MolecularProperties"
aws dynamodb describe-table --table-name $TABLE_NAME

# Lambda Function
FUNCTION_NAME="analyze-molecule"
aws lambda get-function --function-name $FUNCTION_NAME

# IAM Role
ROLE_NAME="lambda-molecular-analyzer"
aws iam get-role --role-name $ROLE_NAME
```

### Common Commands

```bash
# Upload molecule to S3
aws s3 cp molecule.smi s3://$BUCKET_NAME/molecules/drugs/

# Invoke Lambda
aws lambda invoke --function-name analyze-molecule \
  --payload '{"test":"data"}' response.json

# Query DynamoDB
aws dynamodb scan --table-name MolecularProperties --max-items 10

# Check Lambda logs
aws logs tail /aws/lambda/analyze-molecule --follow

# Check costs
aws ce get-cost-and-usage \
  --time-period Start=$(date -d "7 days ago" +%Y-%m-%d),End=$(date +%Y-%m-%d) \
  --granularity DAILY \
  --metrics "UnblendedCost"
```

---

## Support

- **Documentation:** See README.md
- **Issues:** https://github.com/research-jumpstart/research-jumpstart/issues
- **AWS Support:** https://console.aws.amazon.com/support/

---

**Next:** Follow the main workflow in [README.md](README.md) or start analyzing in [notebooks/molecular_analysis.ipynb](notebooks/molecular_analysis.ipynb)
