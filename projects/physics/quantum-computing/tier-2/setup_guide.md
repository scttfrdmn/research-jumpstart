# AWS Setup Guide - Quantum Circuit Simulation Tier 2

This guide walks you through setting up all AWS resources needed for the quantum circuit simulation project. Estimated time: 30-40 minutes.

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

S3 stores your quantum circuit definitions, simulation results, and logs.

### Option A: Using AWS Console (Recommended for beginners)

1. Go to [AWS S3 Console](https://console.aws.amazon.com/s3/)
2. Click "Create bucket"
3. Enter bucket name: `quantum-circuits-{your-name}-{date}`
   - Example: `quantum-circuits-alice-20250114`
   - Must be globally unique
   - Must be lowercase
   - No special characters except hyphens
4. Select region: **us-east-1** (lowest latency for most users)
5. Keep all default settings (versioning: disabled, encryption: default)
6. Click "Create bucket"

### Option B: Using AWS CLI

```bash
# Create S3 bucket with unique name
BUCKET_NAME="quantum-circuits-$(whoami)-$(date +%s)"
aws s3 mb "s3://$BUCKET_NAME" --region us-east-1

# Verify bucket created
aws s3 ls | grep quantum-circuits

# Save bucket name for later
echo "BUCKET_NAME=$BUCKET_NAME" > .env
```

### Verify Bucket Created

```bash
# List your buckets
aws s3 ls

# Should see your new quantum-circuits-xxxx bucket
```

**Save your bucket name!** You'll use it in later steps.

---

## Step 2: Create S3 Folder Structure

Organize circuits and results in specific folders.

### Using AWS Console

1. Open your quantum-circuits bucket
2. Create folders:
   - Click "Create folder" → name it `circuits` → Create
   - Click "Create folder" → name it `results` → Create
   - Click "Create folder" → name it `logs` → Create

### Using AWS CLI

```bash
# Create folder structure (optional for S3, but helps organization)
aws s3api put-object --bucket "$BUCKET_NAME" --key "circuits/"
aws s3api put-object --bucket "$BUCKET_NAME" --key "circuits/bell/"
aws s3api put-object --bucket "$BUCKET_NAME" --key "circuits/grover/"
aws s3api put-object --bucket "$BUCKET_NAME" --key "circuits/ghz/"
aws s3api put-object --bucket "$BUCKET_NAME" --key "circuits/vqe/"
aws s3api put-object --bucket "$BUCKET_NAME" --key "results/"
aws s3api put-object --bucket "$BUCKET_NAME" --key "logs/"

# Verify structure
aws s3 ls "s3://$BUCKET_NAME/" --recursive
```

---

## Step 3: Create DynamoDB Table

DynamoDB stores circuit metadata and simulation results for fast queries.

### Option A: Using AWS Console

1. Go to [DynamoDB Console](https://console.aws.amazon.com/dynamodb/)
2. Click "Create table"
3. Table settings:
   - **Table name**: `QuantumResults`
   - **Partition key**: `CircuitID` (String)
   - **Sort key**: `Timestamp` (String)
4. Table settings:
   - Keep default settings
   - **Table class**: DynamoDB Standard
   - **Capacity mode**: On-demand (pay per request)
5. Click "Create table"
6. Wait for table status to become "Active" (~30 seconds)

### Option B: Using AWS CLI

```bash
# Create DynamoDB table
aws dynamodb create-table \
  --table-name QuantumResults \
  --attribute-definitions \
    AttributeName=CircuitID,AttributeType=S \
    AttributeName=Timestamp,AttributeType=S \
  --key-schema \
    AttributeName=CircuitID,KeyType=HASH \
    AttributeName=Timestamp,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST \
  --region us-east-1

# Wait for table to become active
aws dynamodb wait table-exists --table-name QuantumResults

# Verify table created
aws dynamodb describe-table --table-name QuantumResults
```

### Table Schema

The table stores these attributes:
- **CircuitID** (Partition Key): Unique circuit identifier
- **Timestamp** (Sort Key): Simulation timestamp
- **AlgorithmType**: bell, grover, ghz, vqe, etc.
- **NumQubits**: Number of qubits (1-10)
- **NumGates**: Total gate count
- **Fidelity**: Simulation fidelity (0-1)
- **ExecutionTimeMs**: Lambda execution time
- **MeasurementProbabilities**: JSON of measurement outcomes
- **Entanglement**: Entanglement entropy
- **S3ResultsKey**: Path to detailed results in S3

---

## Step 4: Create IAM Role for Lambda

Lambda needs permissions to read/write S3, DynamoDB, and CloudWatch logs.

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
8. Add tag: Key=`Project` Value=`quantum-tier-2`
9. Role name: `lambda-quantum-simulator`
10. Click "Create role"

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
  --role-name lambda-quantum-simulator \
  --assume-role-policy-document "$TRUST_POLICY"

# Attach required policies
aws iam attach-role-policy \
  --role-name lambda-quantum-simulator \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam attach-role-policy \
  --role-name lambda-quantum-simulator \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam attach-role-policy \
  --role-name lambda-quantum-simulator \
  --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess

# Get role ARN (you'll need this for Lambda)
ROLE_ARN=$(aws iam get-role --role-name lambda-quantum-simulator \
  --query 'Role.Arn' --output text)
echo "ROLE_ARN=$ROLE_ARN" >> .env
echo "Role ARN: $ROLE_ARN"
```

### Security Best Practices

For production, use least-privilege custom policies instead of full access:

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
      "Resource": "arn:aws:s3:::quantum-circuits-*/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:PutItem",
        "dynamodb:GetItem",
        "dynamodb:Query"
      ],
      "Resource": "arn:aws:dynamodb:*:*:table/QuantumResults"
    }
  ]
}
```

**Save your role ARN!** Format: `arn:aws:iam::ACCOUNT_ID:role/lambda-quantum-simulator`

---

## Step 5: Create Lambda Function

Lambda processes quantum circuits using numpy-based simulation.

### Option A: Using AWS Console

1. Go to [Lambda Console](https://console.aws.amazon.com/lambda/)
2. Click "Create function"
3. Function settings:
   - **Function name**: `simulate-quantum-circuit`
   - **Runtime**: **Python 3.11**
   - **Architecture**: **x86_64**
   - **Permissions**: Choose existing role → `lambda-quantum-simulator`
4. Click "Create function"

#### Upload Function Code:

1. In function page, scroll to "Code source"
2. Delete default code
3. Copy entire contents of `scripts/lambda_function.py`
4. Paste into editor
5. Click "Deploy"

#### Configure Function:

1. Click "Configuration" tab
2. Click "General configuration" → "Edit"
3. Settings:
   - **Memory**: **1024 MB** (for up to 10 qubits)
   - **Timeout**: **120 seconds** (2 minutes)
   - **Ephemeral storage**: 512 MB (default)
4. Click "Save"

#### Set Environment Variables:

1. In Configuration tab → "Environment variables"
2. Click "Edit"
3. Add variables:
   - `BUCKET_NAME` = `quantum-circuits-xxxx` (your bucket)
   - `DYNAMODB_TABLE` = `QuantumResults`
   - `MAX_QUBITS` = `10`
   - `AWS_REGION` = `us-east-1`
4. Click "Save"

### Option B: Using AWS CLI

```bash
# Create deployment package
cd scripts
zip lambda_function.zip lambda_function.py
cd ..

# Wait for IAM role to propagate (10-30 seconds)
echo "Waiting for IAM role to be available..."
sleep 15

# Create Lambda function
LAMBDA_ARN=$(aws lambda create-function \
  --function-name simulate-quantum-circuit \
  --runtime python3.11 \
  --role "$ROLE_ARN" \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://scripts/lambda_function.zip \
  --timeout 120 \
  --memory-size 1024 \
  --environment Variables="{BUCKET_NAME=$BUCKET_NAME,DYNAMODB_TABLE=QuantumResults,MAX_QUBITS=10,AWS_REGION=us-east-1}" \
  --query 'FunctionArn' \
  --output text)

echo "LAMBDA_ARN=$LAMBDA_ARN" >> .env
echo "Lambda ARN: $LAMBDA_ARN"

# Verify function created
aws lambda get-function --function-name simulate-quantum-circuit
```

### Lambda Memory Guidelines

Choose memory based on maximum qubits:
- **256 MB**: Up to 6 qubits (64-dimensional state vector)
- **512 MB**: Up to 8 qubits (256-dimensional)
- **1024 MB**: Up to 10 qubits (1024-dimensional)
- **2048 MB**: Up to 11 qubits (2048-dimensional)
- **3008 MB**: Up to 12 qubits (4096-dimensional)

**Note:** State vector size = 2^n complex numbers × 16 bytes each

---

## Step 6: Test Lambda Function

Verify Lambda can simulate a simple quantum circuit.

### Create Test Event

1. In Lambda console, click "Test" tab
2. Create new test event:
   - **Event name**: `test-bell-state`
   - **Event JSON**:

```json
{
  "Records": [
    {
      "s3": {
        "bucket": {
          "name": "quantum-circuits-xxxx"
        },
        "object": {
          "key": "circuits/bell/bell_state.qasm"
        }
      }
    }
  ]
}
```

3. Click "Save"
4. Click "Test"
5. Check execution results (should succeed with status 200)

### Using AWS CLI

```bash
# Create test event file
cat > test_event.json << 'EOF'
{
  "Records": [{
    "s3": {
      "bucket": {"name": "quantum-circuits-test"},
      "object": {"key": "circuits/test.qasm"}
    }
  }]
}
EOF

# Replace bucket name with your actual bucket
sed -i '' "s/quantum-circuits-test/$BUCKET_NAME/g" test_event.json

# Invoke Lambda function
aws lambda invoke \
  --function-name simulate-quantum-circuit \
  --payload file://test_event.json \
  --cli-binary-format raw-in-base64-out \
  response.json

# View response
cat response.json | python -m json.tool
```

Expected response:
```json
{
  "statusCode": 200,
  "body": {
    "circuit_id": "bell_state",
    "qubits": 2,
    "gates": 2,
    "fidelity": 1.0,
    "message": "Circuit simulated successfully"
  }
}
```

---

## Step 7: Upload Sample Circuits

Upload sample quantum circuits to test the pipeline.

### Option A: Using Python Script (Recommended)

```bash
# Configure environment
export AWS_S3_BUCKET="quantum-circuits-xxxx"

# Run upload script
python scripts/upload_to_s3.py

# Verify upload
aws s3 ls "s3://$AWS_S3_BUCKET/circuits/" --recursive
```

### Option B: Manual Upload via Console

1. Go to S3 bucket
2. Open `circuits/` folder
3. Click "Upload"
4. Select QASM files from `sample_circuits/`
5. Click "Upload"

### Option C: Using AWS CLI

```bash
# Upload sample circuits (if you have them locally)
aws s3 cp sample_circuits/ \
  "s3://$BUCKET_NAME/circuits/" \
  --recursive

# Verify upload
aws s3 ls "s3://$BUCKET_NAME/circuits/" --recursive
```

---

## Step 8: Configure S3 Event Trigger (Optional)

Auto-trigger Lambda when new circuits are uploaded to S3.

### Option A: Using AWS Console

1. Go to S3 bucket
2. Click "Properties" tab
3. Scroll to "Event notifications"
4. Click "Create event notification"
5. Settings:
   - **Event name**: `quantum-circuit-trigger`
   - **Prefix**: `circuits/`
   - **Suffix**: `.qasm`
   - **Event types**: Select "All object create events"
   - **Destination**: Lambda function
   - **Lambda function**: `simulate-quantum-circuit`
6. Click "Save changes"

### Option B: Using AWS CLI

```bash
# Add Lambda permission for S3 to invoke
aws lambda add-permission \
  --function-name simulate-quantum-circuit \
  --principal s3.amazonaws.com \
  --statement-id s3-trigger-permission \
  --action lambda:InvokeFunction \
  --source-arn "arn:aws:s3:::$BUCKET_NAME" \
  --source-account $(aws sts get-caller-identity --query Account --output text)

# Create S3 event notification configuration
NOTIFICATION_CONFIG=$(cat <<EOF
{
  "LambdaFunctionConfigurations": [
    {
      "Id": "quantum-circuit-trigger",
      "LambdaFunctionArn": "$LAMBDA_ARN",
      "Events": ["s3:ObjectCreated:*"],
      "Filter": {
        "Key": {
          "FilterRules": [
            {"Name": "prefix", "Value": "circuits/"},
            {"Name": "suffix", "Value": ".qasm"}
          ]
        }
      }
    }
  ]
}
EOF
)

# Apply notification configuration
echo "$NOTIFICATION_CONFIG" | \
aws s3api put-bucket-notification-configuration \
  --bucket "$BUCKET_NAME" \
  --notification-configuration file:///dev/stdin

# Verify notification configured
aws s3api get-bucket-notification-configuration --bucket "$BUCKET_NAME"
```

**Note:** S3 triggers can take 1-2 minutes to activate after configuration.

---

## Step 9: Test Complete Pipeline

Test that data flows through the entire pipeline.

### Test 1: Upload and Simulate

```bash
# Create a simple test circuit (Bell state)
cat > test_bell.qasm << 'EOF'
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q -> c;
EOF

# Upload to S3 (triggers Lambda automatically if trigger configured)
aws s3 cp test_bell.qasm "s3://$BUCKET_NAME/circuits/bell/test_bell.qasm"

# Wait for Lambda to process (5-10 seconds)
sleep 10

# Check DynamoDB for results
aws dynamodb query \
  --table-name QuantumResults \
  --key-condition-expression "CircuitID = :cid" \
  --expression-attribute-values '{":cid":{"S":"test_bell"}}' \
  --limit 1
```

### Test 2: Check CloudWatch Logs

```bash
# Find log group
LOG_GROUP="/aws/lambda/simulate-quantum-circuit"

# Get latest log stream
LATEST_STREAM=$(aws logs describe-log-streams \
  --log-group-name "$LOG_GROUP" \
  --order-by LastEventTime \
  --descending \
  --max-items 1 \
  --query 'logStreams[0].logStreamName' \
  --output text)

# View logs
aws logs get-log-events \
  --log-group-name "$LOG_GROUP" \
  --log-stream-name "$LATEST_STREAM" \
  --limit 50
```

### Test 3: Verify DynamoDB Data

```bash
# Scan DynamoDB table for all results
aws dynamodb scan \
  --table-name QuantumResults \
  --limit 10

# Query specific circuit
aws dynamodb get-item \
  --table-name QuantumResults \
  --key '{"CircuitID":{"S":"test_bell"},"Timestamp":{"S":"2025-01-14T10:00:00Z"}}'
```

### Test 4: Download Results from S3

```bash
# List result files
aws s3 ls "s3://$BUCKET_NAME/results/" --recursive

# Download a result file
aws s3 cp "s3://$BUCKET_NAME/results/test_bell_result.json" ./

# View result
cat test_bell_result.json | python -m json.tool
```

Expected result structure:
```json
{
  "circuit_id": "test_bell",
  "algorithm_type": "bell",
  "num_qubits": 2,
  "num_gates": 2,
  "state_vector": [0.707, 0, 0, 0.707],
  "measurement_probs": {
    "00": 0.5,
    "11": 0.5
  },
  "fidelity": 1.0,
  "entanglement": 1.0,
  "execution_time_ms": 125
}
```

---

## Step 10: Set Up Local Environment

Configure your local machine for analysis.

### Install Dependencies

```bash
# Navigate to project directory
cd tier-2/

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt

# Verify installations
python -c "import boto3; print('boto3:', boto3.__version__)"
python -c "import numpy; print('numpy:', numpy.__version__)"
python -c "import pandas; print('pandas:', pandas.__version__)"
```

### Configure Environment Variables

```bash
# Create .env file
cat > .env << EOF
AWS_REGION=us-east-1
AWS_S3_BUCKET=$BUCKET_NAME
AWS_LAMBDA_FUNCTION=simulate-quantum-circuit
AWS_DYNAMODB_TABLE=QuantumResults
AWS_PROFILE=default
EOF

# Load environment variables (optional)
export $(cat .env | xargs)
```

### Verify AWS Access

```bash
# Check AWS identity
aws sts get-caller-identity

# Test S3 access
aws s3 ls "s3://$BUCKET_NAME/"

# Test DynamoDB access
aws dynamodb describe-table --table-name QuantumResults

# Test Lambda access
aws lambda get-function --function-name simulate-quantum-circuit
```

---

## Step 11: (Optional) Set Up Athena

Use Athena for SQL queries on circuit results.

### Create Athena Database

```bash
# Athena requires a query results location
ATHENA_RESULTS="s3://$BUCKET_NAME/athena-results/"

# Create Athena database
aws athena start-query-execution \
  --query-string "CREATE DATABASE IF NOT EXISTS quantum_circuits;" \
  --result-configuration "OutputLocation=$ATHENA_RESULTS"
```

### Create Athena Table

```sql
-- Run this query in Athena console or via CLI
CREATE EXTERNAL TABLE IF NOT EXISTS quantum_circuits.results (
  CircuitID string,
  Timestamp string,
  AlgorithmType string,
  NumQubits int,
  NumGates int,
  Fidelity double,
  ExecutionTimeMs int,
  Entanglement double
)
ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
LOCATION 's3://quantum-circuits-xxxx/results/';
```

### Query Examples

```sql
-- Average fidelity by algorithm type
SELECT AlgorithmType, AVG(Fidelity) as avg_fidelity, COUNT(*) as count
FROM quantum_circuits.results
GROUP BY AlgorithmType;

-- Circuits with high fidelity (>0.95)
SELECT CircuitID, Fidelity, NumQubits, ExecutionTimeMs
FROM quantum_circuits.results
WHERE Fidelity > 0.95
ORDER BY NumQubits DESC;

-- Execution time vs. qubit count
SELECT NumQubits, AVG(ExecutionTimeMs) as avg_time
FROM quantum_circuits.results
GROUP BY NumQubits
ORDER BY NumQubits;
```

---

## Step 12: Monitor Costs

Track spending to avoid surprises.

### Set Billing Alerts

1. Go to [AWS Billing Console](https://console.aws.amazon.com/billing/)
2. Click "Billing Preferences"
3. Enable "Receive Billing Alerts"
4. Click "Manage Billing Alerts" → CloudWatch
5. Create alarm:
   - **Metric**: BillingEstimatedCharges
   - **Threshold**: $15 USD
   - **Notification**: Email me at [your-email]

### Create Budget

```bash
# Create budget via CLI
aws budgets create-budget \
  --account-id $(aws sts get-caller-identity --query Account --output text) \
  --budget '{
    "BudgetName": "quantum-tier2-budget",
    "BudgetLimit": {"Amount": "20", "Unit": "USD"},
    "TimeUnit": "MONTHLY",
    "BudgetType": "COST"
  }' \
  --notifications-with-subscribers '[{
    "Notification": {
      "NotificationType": "ACTUAL",
      "ComparisonOperator": "GREATER_THAN",
      "Threshold": 80,
      "ThresholdType": "PERCENTAGE"
    },
    "Subscribers": [{
      "SubscriptionType": "EMAIL",
      "Address": "your-email@example.com"
    }]
  }]'
```

### Check Current Costs

```bash
# View costs for last 30 days
aws ce get-cost-and-usage \
  --time-period Start=$(date -d '30 days ago' +%Y-%m-%d),End=$(date +%Y-%m-%d) \
  --granularity DAILY \
  --metrics "UnblendedCost" \
  --group-by Type=DIMENSION,Key=SERVICE

# Or use Cost Explorer GUI:
# https://console.aws.amazon.com/cost-management/
```

---

## Troubleshooting

### Problem: "Access Denied" creating S3 bucket

**Cause:** IAM user lacks S3 permissions

**Solution:**
```bash
# Check your user permissions
aws iam get-user

# Attach S3 full access (or use least-privilege policy)
aws iam attach-user-policy \
  --user-name your-username \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

### Problem: Lambda timeout errors

**Cause:** Circuit too complex or timeout too short

**Solution:**
```bash
# Increase timeout to 5 minutes
aws lambda update-function-configuration \
  --function-name simulate-quantum-circuit \
  --timeout 300

# Increase memory if simulating 10+ qubit circuits
aws lambda update-function-configuration \
  --function-name simulate-quantum-circuit \
  --memory-size 2048
```

### Problem: "Module not found" in Lambda

**Cause:** Lambda doesn't include numpy by default (actually it does in Python 3.11+)

**Solution:**
```bash
# If using older runtime, create Lambda layer with numpy
# For Python 3.11, numpy is included by default
# Verify by checking Lambda console under "Layers"
```

### Problem: DynamoDB access denied

**Cause:** IAM role lacks DynamoDB permissions

**Solution:**
```bash
# Attach DynamoDB access to Lambda role
aws iam attach-role-policy \
  --role-name lambda-quantum-simulator \
  --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess
```

### Problem: S3 trigger not firing

**Cause:** Permission issue or trigger misconfigured

**Solution:**
```bash
# Verify Lambda permission exists
aws lambda get-policy --function-name simulate-quantum-circuit

# Re-add permission if missing
aws lambda add-permission \
  --function-name simulate-quantum-circuit \
  --principal s3.amazonaws.com \
  --statement-id s3-trigger-permission \
  --action lambda:InvokeFunction \
  --source-arn "arn:aws:s3:::$BUCKET_NAME"

# Verify S3 notification configuration
aws s3api get-bucket-notification-configuration --bucket "$BUCKET_NAME"
```

---

## Security Best Practices

### Least Privilege IAM Policies

Replace full access policies with custom least-privilege policies:

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
      "Resource": "arn:aws:s3:::quantum-circuits-*/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:PutItem",
        "dynamodb:Query",
        "dynamodb:GetItem"
      ],
      "Resource": "arn:aws:dynamodb:us-east-1:*:table/QuantumResults"
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:us-east-1:*:log-group:/aws/lambda/simulate-quantum-circuit:*"
    }
  ]
}
```

### Enable S3 Encryption

```bash
# Enable default encryption
aws s3api put-bucket-encryption \
  --bucket "$BUCKET_NAME" \
  --server-side-encryption-configuration '{
    "Rules": [{
      "ApplyServerSideEncryptionByDefault": {
        "SSEAlgorithm": "AES256"
      },
      "BucketKeyEnabled": true
    }]
  }'
```

### Block Public S3 Access

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

1. ✅ Run `python scripts/upload_to_s3.py` to upload sample circuits
2. ✅ Open `notebooks/quantum_analysis.ipynb` for analysis
3. ✅ Review `cleanup_guide.md` for resource deletion
4. ✅ Monitor costs in AWS Cost Explorer
5. ✅ Explore extending with more quantum algorithms

---

## Quick Reference Commands

```bash
# List S3 buckets
aws s3 ls

# Upload file to S3
aws s3 cp circuit.qasm s3://bucket-name/circuits/

# Monitor Lambda logs
aws logs tail /aws/lambda/simulate-quantum-circuit --follow

# Query DynamoDB
aws dynamodb scan --table-name QuantumResults --limit 10

# Check Lambda metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/Lambda \
  --metric-name Duration \
  --dimensions Name=FunctionName,Value=simulate-quantum-circuit \
  --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 300 \
  --statistics Average,Maximum

# Get current AWS costs
aws ce get-cost-and-usage \
  --time-period Start=$(date -d '7 days ago' +%Y-%m-%d),End=$(date +%Y-%m-%d) \
  --granularity DAILY \
  --metrics BlendedCost
```

---

## Support

- **Documentation:** See [README.md](README.md)
- **Issues:** https://github.com/research-jumpstart/research-jumpstart/issues
- **AWS Support:** https://console.aws.amazon.com/support/

---

**Next:** Follow the main workflow in [README.md](README.md) or start analyzing with [notebooks/quantum_analysis.ipynb](notebooks/quantum_analysis.ipynb)

**Last updated:** 2025-11-14
