# AWS Setup Guide for Medical Image Processing Pipeline

This guide provides step-by-step instructions to set up the AWS environment for the Medical Image Processing Tier 2 project.

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

The S3 bucket will store raw and processed medical images.

### Using AWS Management Console

1. Go to **S3** > **Buckets** > **Create Bucket**
2. Enter bucket name: `medical-images-{your-user-id}`
   - **Important:** Bucket names must be globally unique
   - Use lowercase letters and hyphens only
   - Example: `medical-images-scttfrdmn-001`
3. Region: Select your region (e.g., `us-east-1`)
4. **Block Public Access:** Keep all settings enabled (default)
5. Click **Create Bucket**

### Using AWS CLI

```bash
# Replace {your-user-id} with your name/ID
aws s3 mb s3://medical-images-{your-user-id} --region us-east-1

# Verify bucket creation
aws s3 ls | grep medical-images
```

### Create Bucket Folders

Create folder structure:

```bash
# Raw images folder
aws s3api put-object --bucket medical-images-{your-user-id} --key raw-images/

# Processed images folder
aws s3api put-object --bucket medical-images-{your-user-id} --key processed-images/

# Logs folder
aws s3api put-object --bucket medical-images-{your-user-id} --key logs/
```

---

## Step 2: Create DynamoDB Table

The DynamoDB table stores metadata about predictions.

### Using AWS Management Console

1. Go to **DynamoDB** > **Tables** > **Create Table**
2. Table name: `medical-predictions`
3. Partition Key: `image_id` (String)
4. Sort Key: `timestamp` (Number) - optional but recommended
5. Billing Mode: **On-demand** (auto-scales, easier for learning)
6. Click **Create Table**
7. Wait for table to become ACTIVE (1-2 minutes)

### Using AWS CLI

```bash
aws dynamodb create-table \
    --table-name medical-predictions \
    --attribute-definitions \
        AttributeName=image_id,AttributeType=S \
        AttributeName=timestamp,AttributeType=N \
    --key-schema \
        AttributeName=image_id,KeyType=HASH \
        AttributeName=timestamp,KeyType=RANGE \
    --billing-mode PAY_PER_REQUEST \
    --region us-east-1

# Verify table creation
aws dynamodb list-tables --region us-east-1
```

### Verify Table

```bash
aws dynamodb describe-table --table-name medical-predictions --region us-east-1
```

Check that status is `ACTIVE`.

---

## Step 3: Create IAM Role for Lambda

Lambda needs permissions to access S3 and DynamoDB.

### Using AWS Management Console

1. Go to **IAM** > **Roles** > **Create Role**
2. Trusted entity type: **AWS service**
3. Use case: **Lambda**
4. Click **Next**
5. Add permissions:
   - Search for and select: `AmazonS3FullAccess`
   - Search for and select: `AmazonDynamoDBFullAccess`
   - Search for and select: `CloudWatchLogsFullAccess`
   - Click **Next**
6. Role name: `lambda-medical-processor`
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
    --role-name lambda-medical-processor \
    --assume-role-policy-document file://trust-policy.json

# Attach policies
aws iam attach-role-policy \
    --role-name lambda-medical-processor \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam attach-role-policy \
    --role-name lambda-medical-processor \
    --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess

aws iam attach-role-policy \
    --role-name lambda-medical-processor \
    --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess

# Get role ARN (you'll need this for Lambda)
aws iam get-role --role-name lambda-medical-processor --query 'Role.Arn'
```

**Save the role ARN** - you'll need it for Lambda deployment.

Example: `arn:aws:iam::123456789012:role/lambda-medical-processor`

---

## Step 4: Deploy Lambda Function

### Create Lambda Function

1. Go to **Lambda** > **Functions** > **Create Function**
2. Function name: `process-medical-images`
3. Runtime: **Python 3.11** (or latest)
4. Execution role: **Use an existing role**
5. Existing role: Select `lambda-medical-processor` from dropdown
6. Click **Create Function**

### Upload Code

In the Lambda console:

1. Go to the **Code** tab
2. Copy the code from `scripts/lambda_function.py`
3. Paste into the Lambda editor
4. Click **Deploy**

Or using AWS CLI:

```bash
# Zip the lambda function
zip lambda_function.zip scripts/lambda_function.py

# Deploy to Lambda
aws lambda create-function \
    --function-name process-medical-images \
    --runtime python3.11 \
    --role arn:aws:iam::YOUR_ACCOUNT_ID:role/lambda-medical-processor \
    --handler lambda_function.lambda_handler \
    --zip-file fileb://lambda_function.zip \
    --timeout 300 \
    --memory-size 256

# If function already exists, update it
aws lambda update-function-code \
    --function-name process-medical-images \
    --zip-file fileb://lambda_function.zip
```

### Configure Lambda Settings

1. Go to **Configuration** tab
2. Set:
   - **Timeout:** 300 seconds (5 minutes)
   - **Memory:** 256 MB (sufficient for image processing)
   - **Ephemeral storage:** 512 MB

```bash
# Using AWS CLI
aws lambda update-function-configuration \
    --function-name process-medical-images \
    --timeout 300 \
    --memory-size 256 \
    --ephemeral-storage Size=512
```

### Test Lambda Function

Create a test event in the Lambda console or use CLI:

```bash
# Create test event file
cat > test-event.json << 'EOF'
{
  "Records": [
    {
      "s3": {
        "bucket": {
          "name": "medical-images-{your-user-id}"
        },
        "object": {
          "key": "raw-images/sample.png"
        }
      }
    }
  ]
}
EOF

# Test function (if test images exist)
# aws lambda invoke --function-name process-medical-images --payload file://test-event.json response.json
```

---

## Step 5: Set Up S3 Event Notifications (Optional)

Configure S3 to automatically trigger Lambda when images are uploaded.

### Using AWS Management Console

1. Go to **S3** > Your bucket > **Properties** > **Event notifications**
2. Click **Create event notification**
3. Event name: `trigger-image-processing`
4. Events: Select **s3:ObjectCreated:***
5. Destination: **Lambda function**
6. Lambda function: Select `process-medical-images`
7. Click **Save**

### Using AWS CLI

```bash
# Create Lambda permission to be invoked by S3
aws lambda add-permission \
    --function-name process-medical-images \
    --statement-id AllowS3Invoke \
    --action lambda:InvokeFunction \
    --principal s3.amazonaws.com \
    --source-arn arn:aws:s3:::medical-images-{your-user-id}

# Configure S3 bucket notification
cat > s3-event-config.json << 'EOF'
{
  "LambdaFunctionConfigurations": [
    {
      "LambdaFunctionArn": "arn:aws:lambda:us-east-1:YOUR_ACCOUNT_ID:function:process-medical-images",
      "Events": ["s3:ObjectCreated:*"],
      "Filter": {
        "Key": {
          "FilterRules": [
            {
              "Name": "prefix",
              "Value": "raw-images/"
            }
          ]
        }
      }
    }
  ]
}
EOF

aws s3api put-bucket-notification-configuration \
    --bucket medical-images-{your-user-id} \
    --notification-configuration file://s3-event-config.json
```

---

## Step 6: Install Python Dependencies

### Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Verify installation:
```bash
python -c "import boto3; print(f'boto3 version: {boto3.__version__}')"
```

---

## Step 7: Environment Variables Setup

Create a `.env` file with your AWS configuration:

```bash
cd /path/to/tier-2/

cat > .env << 'EOF'
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here

# S3 Configuration
S3_BUCKET_NAME=medical-images-{your-user-id}
S3_RAW_PREFIX=raw-images/
S3_PROCESSED_PREFIX=processed-images/

# DynamoDB Configuration
DYNAMODB_TABLE_NAME=medical-predictions
DYNAMODB_REGION=us-east-1

# Lambda Configuration
LAMBDA_FUNCTION_NAME=process-medical-images
LAMBDA_REGION=us-east-1

# Image Processing
IMAGE_SIZE=224
IMAGE_FORMAT=png
EOF
```

**Important:** Never commit `.env` to version control (add to `.gitignore`)

---

## Step 8: Verify Setup

Run verification script to ensure everything is configured:

```bash
python -c "
import boto3
import os
from dotenv import load_dotenv

load_dotenv()

# Check S3
s3 = boto3.client('s3', region_name=os.getenv('AWS_REGION'))
try:
    s3.head_bucket(Bucket=os.getenv('S3_BUCKET_NAME'))
    print('✓ S3 bucket accessible')
except:
    print('✗ S3 bucket not accessible')

# Check DynamoDB
dynamodb = boto3.client('dynamodb', region_name=os.getenv('DYNAMODB_REGION'))
try:
    dynamodb.describe_table(TableName=os.getenv('DYNAMODB_TABLE_NAME'))
    print('✓ DynamoDB table accessible')
except:
    print('✗ DynamoDB table not accessible')

# Check Lambda
lambda_client = boto3.client('lambda', region_name=os.getenv('LAMBDA_REGION'))
try:
    lambda_client.get_function(FunctionName=os.getenv('LAMBDA_FUNCTION_NAME'))
    print('✓ Lambda function accessible')
except:
    print('✗ Lambda function not accessible')
"
```

---

## Troubleshooting Setup Issues

### "NoCredentialsError"
- Solution: Run `aws configure` and enter your access keys
- Verify with `aws sts get-caller-identity`

### "Access Denied" for S3
- Check IAM user has S3FullAccess policy
- Check S3 bucket policy (should be open to your account)

### "Access Denied" for DynamoDB
- Ensure Lambda role has DynamoDB permissions
- Verify role is attached to Lambda function

### "Bucket already exists"
- Bucket names are globally unique across AWS
- Choose a different name with a unique suffix

### Lambda timeout issues
- Increase timeout to 300 seconds in configuration
- Check Lambda logs in CloudWatch for errors

### S3 event notification not triggering
- Verify Lambda permission was added with `add-permission`
- Check S3 event configuration with `get-bucket-notification-configuration`
- Ensure filter prefix matches uploaded objects

---

## Cost Monitoring

To monitor costs as you use the pipeline:

### Using AWS Console
1. Go to **Billing** > **Cost Explorer**
2. Set date range for your project
3. Filter by service (S3, Lambda, DynamoDB)
4. Check daily and hourly breakdowns

### Using AWS CLI
```bash
aws ce get-cost-and-usage \
    --time-period Start=2024-01-01,End=2024-01-08 \
    --granularity DAILY \
    --metrics BlendedCost \
    --group-by Type=DIMENSION,Key=SERVICE \
    --filter file://cost-filter.json
```

---

## Next Steps

After setup verification:

1. **Prepare test data:** Use sample X-rays or public datasets
2. **Run upload script:** `python scripts/upload_to_s3.py`
3. **Process images:** Trigger Lambda function
4. **Query results:** `python scripts/query_results.py`
5. **Analyze in notebook:** Open `notebooks/image_analysis.ipynb`

See `README.md` for workflow instructions.

---

## Cleanup

After completing the project, follow `cleanup_guide.md` to delete AWS resources and avoid charges.

**Important:** Incomplete cleanup can result in ongoing charges!
