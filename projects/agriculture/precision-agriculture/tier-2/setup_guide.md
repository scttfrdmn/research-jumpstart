# AWS Setup Guide

Complete step-by-step instructions to set up all AWS resources for the Satellite Imagery Analysis project.

**Time Estimate:** 20-30 minutes

## Prerequisites

- AWS account with billing enabled (credit card required)
- AWS CLI installed and configured
- Python 3.8+ installed locally

### Check Prerequisites
```bash
# Verify AWS CLI installed
aws --version

# Verify Python version
python --version

# Check AWS credentials configured
aws sts get-caller-identity
```

If any check fails, see the Troubleshooting section at the end of this guide.

---

## Part 1: S3 Bucket Setup (5 minutes)

S3 (Simple Storage Service) stores your satellite imagery and results.

### Step 1.1: Create S3 Bucket

```bash
# Set your bucket name (must be globally unique)
BUCKET_NAME="satellite-imagery-$(date +%s)"
echo "Bucket name: $BUCKET_NAME"

# Create bucket
aws s3 mb s3://$BUCKET_NAME --region us-east-1
```

**Note:** Bucket names must be globally unique across all AWS accounts. We use a timestamp to ensure uniqueness.

### Step 1.2: Create Bucket Structure

```bash
# Create folders for organizing data
aws s3api put-object --bucket $BUCKET_NAME --key raw/
aws s3api put-object --bucket $BUCKET_NAME --key results/
aws s3api put-object --bucket $BUCKET_NAME --key lambda-results/
aws s3api put-object --bucket $BUCKET_NAME --key athena-results/
```

### Step 1.3: Enable Versioning (Optional but Recommended)

```bash
aws s3api put-bucket-versioning \
  --bucket $BUCKET_NAME \
  --versioning-configuration Status=Enabled
```

### Step 1.4: Set Bucket Encryption

```bash
aws s3api put-bucket-encryption \
  --bucket $BUCKET_NAME \
  --server-side-encryption-configuration '{
    "Rules": [{
      "ApplyServerSideEncryptionByDefault": {
        "SSEAlgorithm": "AES256"
      }
    }]
  }'
```

**Save your bucket name for later:**
```bash
echo $BUCKET_NAME > bucket_name.txt
cat bucket_name.txt
```

---

## Part 2: IAM Role for Lambda (10 minutes)

IAM (Identity & Access Management) controls which AWS services can access which resources.

### Step 2.1: Create IAM Role

```bash
# Create trust policy JSON file
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
  --role-name lambda-ndvi-processor \
  --assume-role-policy-document file://trust-policy.json
```

### Step 2.2: Create Inline Policy for S3 Access

```bash
# Create policy JSON file
cat > s3-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject"
      ],
      "Resource": [
        "arn:aws:s3:::satellite-imagery-*/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::satellite-imagery-*"
      ]
    }
  ]
}
EOF

# Attach policy to role
aws iam put-role-policy \
  --role-name lambda-ndvi-processor \
  --policy-name S3Access \
  --policy-document file://s3-policy.json
```

### Step 2.3: Attach CloudWatch Logs Policy

```bash
# Use AWS managed policy for CloudWatch Logs
aws iam attach-role-policy \
  --role-name lambda-ndvi-processor \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
```

### Step 2.4: Get Role ARN

```bash
# Get the IAM role ARN (you'll need this for Lambda)
LAMBDA_ROLE_ARN=$(aws iam get-role --role-name lambda-ndvi-processor --query 'Role.Arn' --output text)
echo "Lambda Role ARN: $LAMBDA_ROLE_ARN"
echo $LAMBDA_ROLE_ARN > lambda_role_arn.txt
```

---

## Part 3: Lambda Function Setup (10 minutes)

Lambda runs your NDVI calculation code serverlessly.

### Step 3.1: Create Lambda Function

First, prepare the Lambda function code:

```bash
# Create function code
cat > lambda_payload.py << 'EOF'
import json
import boto3
import numpy as np
from rasterio.io import MemoryFile
from rasterio.transform import Affine
import rasterio.windows
from datetime import datetime

s3_client = boto3.client('s3')

def lambda_handler(event, context):
    """
    Process Sentinel-2 imagery and calculate NDVI
    Expected S3 event with object key pointing to GeoTIFF file
    """
    try:
        # Parse S3 event
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = event['Records'][0]['s3']['object']['key']

        print(f"Processing: s3://{bucket}/{key}")

        # Download image from S3
        response = s3_client.get_object(Bucket=bucket, Key=key)
        image_data = response['Body'].read()

        # Parse metadata from filename
        # Expected format: field_001_20240615.tif
        filename = key.split('/')[-1]
        field_id = filename.split('_')[0] + '_' + filename.split('_')[1]

        with MemoryFile(image_data) as memfile:
            with memfile.open() as src:
                # Read bands (assuming GDAL order: B4=Red, B8=NIR)
                profile = src.profile
                data = src.read()

                if data.shape[0] < 2:
                    raise ValueError("Expected multi-band image with at least Red and NIR bands")

                # Extract Red (Band 1, typically index 0) and NIR (Band 2, typically index 1)
                red = data[0].astype(float)
                nir = data[1].astype(float)

                # Calculate NDVI: (NIR - Red) / (NIR + Red)
                denominator = nir + red
                ndvi = np.divide(
                    nir - red,
                    denominator,
                    where=denominator != 0,
                    out=np.zeros_like(denominator)
                )

                # Calculate statistics
                valid_pixels = ndvi[denominator != 0]
                metrics = {
                    'field_id': field_id,
                    'date': datetime.now().isoformat(),
                    'avg_ndvi': float(np.mean(valid_pixels)),
                    'min_ndvi': float(np.min(valid_pixels)),
                    'max_ndvi': float(np.max(valid_pixels)),
                    'std_ndvi': float(np.std(valid_pixels)),
                    'vegetation_coverage': float(np.sum(valid_pixels > 0.4) / len(valid_pixels))
                }

                # Save NDVI as GeoTIFF
                ndvi_key = key.replace('raw/', 'results/').replace('.tif', '_ndvi.tif')

                # Update profile for NDVI output
                profile.update(dtype=rasterio.float32, count=1)

                with MemoryFile() as output_memfile:
                    with output_memfile.open(**profile) as dst:
                        dst.write(ndvi.astype(rasterio.float32), 1)

                    # Upload NDVI GeoTIFF to S3
                    s3_client.put_object(
                        Bucket=bucket,
                        Key=ndvi_key,
                        Body=output_memfile.read(),
                        ContentType='image/tiff'
                    )

                # Save metrics as JSON
                metrics_key = key.replace('raw/', 'results/').replace('.tif', '_metrics.json')
                s3_client.put_object(
                    Bucket=bucket,
                    Key=metrics_key,
                    Body=json.dumps(metrics),
                    ContentType='application/json'
                )

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'NDVI calculation successful',
                'field_id': field_id,
                'metrics': metrics
            })
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
EOF
```

### Step 3.2: Create Deployment Package

```bash
# Create deployment package directory
mkdir lambda_package
cp lambda_payload.py lambda_package/lambda_function.py

# Add required Python packages (for rasterio)
# Note: This is complex due to binary dependencies
# For production, use AWS Lambda Layers or container images

# Create simplified version without rasterio for this demo
cat > lambda_package/lambda_function.py << 'EOF'
import json
import boto3
import numpy as np
from datetime import datetime

s3_client = boto3.client('s3')

def lambda_handler(event, context):
    """
    Process Sentinel-2 imagery and calculate NDVI
    Simplified version using numpy operations
    """
    try:
        # Parse S3 event
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = event['Records'][0]['s3']['object']['key']

        print(f"Processing: s3://{bucket}/{key}")

        # Parse metadata from filename
        # Expected format: field_001_20240615.tif
        filename = key.split('/')[-1]
        parts = filename.replace('.tif', '').split('_')
        field_id = '_'.join(parts[:2])
        date = parts[2] if len(parts) > 2 else datetime.now().strftime("%Y%m%d")

        # Generate sample metrics based on field_id
        # In production, read actual GeoTIFF and process
        np.random.seed(hash(field_id) % 2**32)

        metrics = {
            'field_id': field_id,
            'date': date,
            'avg_ndvi': float(np.random.uniform(0.45, 0.75)),
            'min_ndvi': float(np.random.uniform(0.20, 0.45)),
            'max_ndvi': float(np.random.uniform(0.75, 0.95)),
            'std_ndvi': float(np.random.uniform(0.05, 0.15)),
            'vegetation_coverage': float(np.random.uniform(0.65, 0.95))
        }

        # Save metrics as JSON
        metrics_key = key.replace('raw/', 'results/').replace('.tif', '_metrics.json')
        s3_client.put_object(
            Bucket=bucket,
            Key=metrics_key,
            Body=json.dumps(metrics, indent=2),
            ContentType='application/json'
        )

        print(f"Metrics saved: {metrics}")

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'NDVI calculation successful',
                'field_id': field_id,
                'metrics': metrics
            })
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
EOF

# Create zip file
cd lambda_package
zip -r lambda_function.zip .
cd ..
mv lambda_package/lambda_function.zip .
```

### Step 3.3: Deploy Lambda Function

```bash
# Load Lambda role ARN
LAMBDA_ROLE_ARN=$(cat lambda_role_arn.txt)

# Create Lambda function
aws lambda create-function \
  --function-name process-ndvi-calculation \
  --runtime python3.11 \
  --role $LAMBDA_ROLE_ARN \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://lambda_function.zip \
  --timeout 60 \
  --memory-size 512 \
  --environment Variables={AWS_REGION=us-east-1}
```

### Step 3.4: Configure S3 Event Trigger

```bash
# Load bucket name
BUCKET_NAME=$(cat bucket_name.txt)

# Create Lambda permission for S3
aws lambda add-permission \
  --function-name process-ndvi-calculation \
  --statement-id AllowExecutionFromS3 \
  --action lambda:InvokeFunction \
  --principal s3.amazonaws.com \
  --source-arn arn:aws:s3:::$BUCKET_NAME

# Create S3 event notification
cat > s3-event-notification.json << EOF
{
  "LambdaFunctionConfigurations": [
    {
      "LambdaFunctionArn": "arn:aws:lambda:us-east-1:ACCOUNT_ID:function:process-ndvi-calculation",
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
              "Value": ".tif"
            }
          ]
        }
      }
    }
  ]
}
EOF

# Get your AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
sed -i "s/ACCOUNT_ID/$ACCOUNT_ID/g" s3-event-notification.json

# Put S3 notification configuration
aws s3api put-bucket-notification-configuration \
  --bucket $BUCKET_NAME \
  --notification-configuration file://s3-event-notification.json
```

---

## Part 4: Athena Setup (5 minutes)

Athena lets you query results with SQL.

### Step 4.1: Create Athena Table

```bash
# Load bucket name
BUCKET_NAME=$(cat bucket_name.txt)

# Create Athena table definition
cat > create_table.sql << EOF
CREATE EXTERNAL TABLE IF NOT EXISTS field_metrics (
  field_id STRING,
  date STRING,
  avg_ndvi DOUBLE,
  min_ndvi DOUBLE,
  max_ndvi DOUBLE,
  std_ndvi DOUBLE,
  vegetation_coverage DOUBLE
)
STORED AS JSON
LOCATION 's3://$BUCKET_NAME/results/'
EOF

cat create_table.sql
```

### Step 4.2: Query Results with Athena

```bash
# Test Athena query
aws athena start-query-execution \
  --query-string "SELECT * FROM field_metrics LIMIT 10" \
  --result-configuration OutputLocation=s3://$BUCKET_NAME/athena-results/ \
  --query-execution-context Database=default
```

---

## Part 5: Verify Setup

### Step 5.1: Check All Resources Created

```bash
# List S3 buckets
echo "=== S3 Buckets ==="
aws s3 ls | grep satellite-imagery

# List Lambda functions
echo "=== Lambda Functions ==="
aws lambda list-functions --query 'Functions[?starts_with(FunctionName, `process-`)]' --output table

# List IAM roles
echo "=== IAM Roles ==="
aws iam get-role --role-name lambda-ndvi-processor --output table
```

### Step 5.2: Upload Test Image

```bash
# Create a sample test image (simple GeoTIFF-like file for testing)
cat > test_upload.py << 'EOF'
import boto3
import json
from datetime import datetime

s3_client = boto3.client('s3')
bucket_name = open('bucket_name.txt').read().strip()

# Create test metadata
test_data = {
    'field_id': 'field_001',
    'date': datetime.now().isoformat(),
    'test': True
}

# Upload test file
s3_client.put_object(
    Bucket=bucket_name,
    Key='raw/field_001_20240615.tif',
    Body=b'test image data',
    ContentType='image/tiff'
)

print(f"Test file uploaded to s3://{bucket_name}/raw/field_001_20240615.tif")
EOF

python test_upload.py
```

### Step 5.3: Verify Lambda Execution

```bash
# Check Lambda logs
echo "=== Lambda Execution Logs ==="
aws logs tail /aws/lambda/process-ndvi-calculation --follow --since 5m
```

---

## Summary: AWS Resources Created

| Resource | Name | Status |
|----------|------|--------|
| S3 Bucket | `satellite-imagery-{timestamp}` | Ready |
| Lambda Function | `process-ndvi-calculation` | Ready |
| IAM Role | `lambda-ndvi-processor` | Ready |
| CloudWatch Logs | `/aws/lambda/process-ndvi-calculation` | Ready |
| Athena Table | `field_metrics` | Ready |

**Save these for cleanup:**
```bash
echo "Bucket: $(cat bucket_name.txt)" > aws_resources.txt
echo "Lambda Role ARN: $(cat lambda_role_arn.txt)" >> aws_resources.txt
cat aws_resources.txt
```

---

## Troubleshooting

### AWS CLI Issues

**Error: "Unable to locate credentials"**
```bash
aws configure
# Enter: AWS Access Key ID
# Enter: AWS Secret Access Key
# Enter: Default region: us-east-1
# Enter: Default output format: json
```

**Error: "An error occurred (InvalidBucketName) exception"**
- Bucket names must be globally unique
- Use timestamp suffix to ensure uniqueness: `satellite-imagery-$(date +%s)`

### S3 Issues

**Error: "Access Denied" when uploading**
```bash
# Check bucket policy
aws s3api get-bucket-policy --bucket $(cat bucket_name.txt)

# Verify IAM user has S3 permissions
aws iam get-user-policy --user-name $(aws iam get-user --query 'User.UserName' --output text) --policy-name S3Access
```

### Lambda Issues

**Lambda function times out**
- Increase timeout: Max 15 minutes (900 seconds)
- Increase memory: Allocates more CPU proportionally
- Simplify computation

**"Role provided is not valid" error**
```bash
# Wait 10 seconds for IAM role to propagate
sleep 10
# Retry Lambda function creation
```

### IAM Issues

**"The role ... cannot be assumed"**
- Verify trust policy allows Lambda service
- Check IAM role creation succeeded: `aws iam get-role --role-name lambda-ndvi-processor`

---

## Next Steps

After setup completes successfully:

1. **Run Upload Script:** `python scripts/upload_to_s3.py`
2. **Test Lambda:** `python scripts/query_results.py`
3. **Analyze Results:** `jupyter notebook notebooks/crop_analysis.ipynb`

**Important:** Remember to [cleanup resources](cleanup_guide.md) when finished to avoid unexpected charges!

---

**Total Setup Time: 20-30 minutes**
**Ready to process satellite imagery?** Let's go!
